"""Command-line entry point.

Exit codes: 0 ok (pass/warn/skip only), 1 at least one FAIL, 2 hwb error/crash, 130 interrupted.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from types import FrameType

from . import __version__, bench, serve, update
from . import config as hwb_config
from .core.context import RunContext
from .core.errors import HwbError
from .core.log import setup_logging
from .core.platform import is_linux, os_info
from .core.report import render_console, write_report
from .core.report_html import INDEX_HTML, collect_runs
from .detect.gpu import run_detect
from .install import plan as install
from .metrics.collect import run_info
from .share import client as share_client
from .share import command as share

log = logging.getLogger("hwb")

EXIT_ERROR = 2
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hwb", description="Acceptance tests for fresh NVIDIA GPU servers."
    )
    p.add_argument("--version", action="version", version=f"hwb {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging on stderr")
    p.add_argument("--out", type=Path, default=Path("hwb-runs"), help="base dir for run output")
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    p.set_defaults(command="info")

    sub.add_parser(
        "info", help="check the GPU driver/CUDA stack and collect the hardware inventory (default)"
    )

    ins = sub.add_parser("install", help="plan (default) or apply driver/CUDA/tool installation")
    ins.add_argument("target", choices=("driver", "cuda", "tools"))
    ins.add_argument("--apply", action="store_true", help="execute the plan (root required)")
    ins.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    ins.add_argument(
        "--branch", default=install.DEFAULT_DRIVER_BRANCH, help="driver branch (Ubuntu)"
    )
    ins.add_argument("--cuda-version", help="e.g. 12.8 (default: latest in repo)")

    b = sub.add_parser("bench", help="run benchmarks")
    b.add_argument("kind", choices=(*bench.KINDS, "all"))
    b.add_argument("--duration", type=int, default=60, help="seconds of load per benchmark")
    b.add_argument(
        "--path", type=Path, default=bench.BenchOptions().path, help="scratch dir for disk tests"
    )
    b.add_argument(
        "--interval",
        type=float,
        default=bench.BenchOptions().interval_s,
        help="sampling interval (s)",
    )

    m = sub.add_parser(
        "monitor",
        help="record temperatures, clocks, power and PCIe links while the machine runs its own workload",
        description="Record the benchmark time series (CPU/GPU temperature, clocks, power, GPU "
        "PCIe link generation/width and error counters) without adding load. Flags thermal "
        "limits, throttling and PCIe links dropping below their max while busy. Ctrl-C stops "
        "early and keeps what was recorded.",
    )
    m.add_argument("--duration", type=int, default=600, help="seconds to record (default 600)")
    m.add_argument(
        "--interval",
        type=float,
        default=bench.BenchOptions().interval_s,
        help="sampling interval (s)",
    )

    s = sub.add_parser(
        "serve",
        help="host a model with vLLM (a preset for this hardware) or test a running server",
        description="Launch vLLM with a preset matched to the detected GPUs (or --endpoint to test a "
        "server you already run), check its answers (arithmetic, JSON, code, long-context recall, tool "
        "call), then load-test it over several concurrency levels while recording GPU temperature, "
        "power, clocks and PCIe links. vLLM is always stopped afterwards.",
    )
    s.add_argument(
        "--list", action="store_true", help="list the presets and which fit this machine"
    )
    s.add_argument("--preset", help="preset name (default: the verified preset for this hardware)")
    s.add_argument(
        "--endpoint", metavar="URL", help="test this running OpenAI-compatible server instead"
    )
    s.add_argument(
        "--model",
        help="with --endpoint: the served model name (default: the first one it serves); "
        "otherwise: launch this model from the machine (HF id in a cache, or a directory) without a preset",
    )
    s.add_argument(
        "--port", type=int, default=8000, help="port for the launched vLLM (default 8000)"
    )
    s.add_argument(
        "--concurrency", default="1,4,16", help="load levels, comma-separated (default 1,4,16)"
    )
    s.add_argument(
        "--requests", type=int, default=4, help="requests per level, times its concurrency"
    )
    s.add_argument("--max-tokens", type=int, default=256, help="output tokens per load request")
    s.add_argument(
        "--ready-timeout", type=int, default=3600, help="seconds to wait for vLLM to load"
    )
    s.add_argument("--interval", type=float, default=1.0, help="sampling interval (s)")
    s.add_argument("--vllm", metavar="PATH", help="vllm executable to use")
    s.add_argument(
        "--install-vllm",
        action="store_true",
        help=f"install {serve.presets.VLLM_SPEC} into an isolated uv venv under $HWB_HOME",
    )
    s.add_argument("--dry-run", action="store_true", help="print the vllm command and stop")
    s.add_argument("--force", action="store_true", help="run a preset meant for other hardware")

    a = sub.add_parser("all", help="info + all benchmarks (+ serve with --preset)")
    a.add_argument("--duration", type=int, default=60, help="seconds of load per benchmark")
    a.add_argument(
        "--interval",
        type=float,
        default=bench.BenchOptions().interval_s,
        help="sampling interval (s)",
    )
    a.add_argument("--preset", help="also run hwb serve with this preset")

    sb = sub.add_parser(
        "submit",
        help="share the runs in --out through an hwb gateway (sanitized, after confirmation)",
        description="Upload the runs in --out to an hwb gateway and print the link and a QR code. "
        "Host names, addresses, UUIDs, user paths and serials other than the GPUs' are removed "
        "first; you see a summary and confirm before anything is sent.",
    )
    sb.add_argument(
        "--gateway",
        help=f"gateway URL (default: ${share_client.ENV_GATEWAY}, else {share_client.DEFAULT_GATEWAY})",
    )
    sb.add_argument(
        "--label",
        help="short label shown with the report, e.g. a rack position (default: hwb config label)",
    )
    sb.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="DIR",
        help="only this run directory (repeatable)",
    )
    sb.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")
    sb.add_argument(
        "--dry-run", action="store_true", help="write the exact upload to --out and send nothing"
    )
    sb.add_argument("--delete", metavar="ID", help="delete a report you submitted from this --out")

    up = sub.add_parser(
        "update",
        help="check for a newer hwb release; --apply installs it over this hwb",
        description="Show the installed and latest release. With --apply, download the release's "
        "hwb.pyz, verify it against SHA256SUMS, test-run it and replace this hwb in place "
        f"(after confirmation). Interactive runs check daily; {update.ENV_NO_CHECK}=1 disables that.",
    )
    up.add_argument("--apply", action="store_true", help="install the newer release")
    up.add_argument("-y", "--yes", action="store_true", help="skip the confirmation prompt")

    cf = sub.add_parser(
        "config",
        help="change saved settings in a menu, e.g. submit every run automatically",
        description="Without arguments in a terminal: a menu to change the settings. "
        "Settings are kept in "
        f"{hwb_config.FILE} under $XDG_CONFIG_HOME (~/.config) or ${hwb_config.ENV_CONFIG}; "
        "`show`, `set` and `unset` work in scripts. "
        "`hwb config set submit auto` uploads each finished run (with the latest info run for "
        "context) to the gateway without asking, also when nobody watches the terminal.",
    )
    cf.add_argument("action", nargs="?", choices=("show", "set", "unset"))
    cf.add_argument("key", nargs="?", choices=tuple(hwb_config.KEYS))
    cf.add_argument("value", nargs="?")
    return p


def _cmd_info(ctx: RunContext, args: argparse.Namespace) -> None:
    run_detect(ctx)
    run_info(ctx)


def _cmd_install(ctx: RunContext, args: argparse.Namespace) -> None:
    osi = os_info()
    if args.target == "driver":
        steps = install.driver_plan(osi, ctx.run_dir, args.branch)
    elif args.target == "cuda":
        steps = install.cuda_plan(osi, ctx.run_dir, args.cuda_version)
    else:
        steps = install.tools_plan(osi)
    install.execute(ctx, args.target, steps, apply=args.apply, assume_yes=args.yes)


def _cmd_bench(ctx: RunContext, args: argparse.Namespace) -> None:
    opts = bench.BenchOptions(duration_s=args.duration, path=args.path, interval_s=args.interval)
    bench.run(ctx, args.kind, opts)


def _cmd_monitor(ctx: RunContext, args: argparse.Namespace) -> None:
    bench.monitor.run(ctx, bench.BenchOptions(duration_s=args.duration, interval_s=args.interval))


def _serve_options(args: argparse.Namespace) -> serve.ServeOptions:
    try:
        levels = tuple(int(c) for c in args.concurrency.split(",") if c.strip())
    except ValueError:
        raise HwbError(
            f"--concurrency must be numbers like 1,4,16, not {args.concurrency!r}"
        ) from None
    return serve.ServeOptions(
        preset=args.preset,
        endpoint=args.endpoint,
        model=args.model,
        port=args.port,
        ready_timeout_s=args.ready_timeout,
        concurrency=levels or (1,),
        requests_per_level=args.requests,
        max_tokens=args.max_tokens,
        interval_s=args.interval,
        vllm=args.vllm,
        install_vllm=args.install_vllm,
        dry_run=args.dry_run,
        force=args.force,
    )


def _cmd_serve(ctx: RunContext, args: argparse.Namespace) -> None:
    serve.run(ctx, _serve_options(args))


def _serve_list() -> int:
    hw = serve.presets.matching_hardware(serve.detected_gpus())
    found = serve.weights.find_all()
    print(f"This machine: {hw.label if hw else 'no preset hardware profile matches'}\n")
    print(f"{'preset':<28} {'fits':<5} {'on disk':<8} {'status':<9} {'weights':>8}  model")
    for p in serve.presets.PRESETS:
        fits = "yes" if hw and p.hardware == hw.id else "-"
        cached = "yes" if serve.weights.find(found, p.model, p.revision) else "-"
        star = " (default)" if p.default else ""
        print(
            f"{p.name:<28} {fits:<5} {cached:<8} {p.status:<9} {p.weights_gb:>6.0f}GB  {p.model}{star}"
        )
    print(f"\nModel weights on this machine ({len(found)}):")
    for w in found:
        preset = serve.presets.match_model(w.id)
        if not w.complete:
            how = "incomplete download"
        elif not w.vllm_runnable:
            how = f"{w.format}: run it with {w.source}, test with --endpoint"
        else:
            how = (
                f"preset {preset.name}"
                if preset
                else f"hwb serve --model {w.id if w.source == 'huggingface' else w.path}"
            )
        print(
            f"  {w.id:<44} {w.source:<12} {w.size_gb:>6.1f} GB  {w.architecture or '?':<34} {how}"
        )
    if not found:
        print(
            "  none found (Hugging Face, ModelScope, Ollama, LM Studio caches; $HWB_MODEL_DIRS, /models)"
        )
    print(
        f"\nNew downloads go to the shared Hugging Face cache: {serve.weights.default_hub()} (set HF_HOME to move it)"
    )
    print("hwb serve --preset NAME [--dry-run] | --model ID-OR-PATH | --endpoint http://host:8000")
    return 0


def _cmd_all(ctx: RunContext, args: argparse.Namespace) -> None:
    _cmd_info(ctx, args)
    bench.run(ctx, "all", bench.BenchOptions(duration_s=args.duration, interval_s=args.interval))
    if args.preset:
        serve.run(ctx, serve.ServeOptions(preset=args.preset))


COMMANDS: dict[str, Callable[[RunContext, argparse.Namespace], None]] = {
    "info": _cmd_info,
    "install": _cmd_install,
    "bench": _cmd_bench,
    "monitor": _cmd_monitor,
    "serve": _cmd_serve,
    "all": _cmd_all,
}


def _on_sigterm(signum: int, frame: FrameType | None) -> None:
    # Route SIGTERM through the KeyboardInterrupt path so cleanup and the report still run.
    raise KeyboardInterrupt


def _cmd_submit(args: argparse.Namespace) -> int:
    setup_logging(None, verbose=args.verbose)
    token = os.environ.get(share_client.ENV_TOKEN)
    try:
        settings = hwb_config.load()
        if args.delete:
            return share.delete_submission(
                args.out, args.delete, gateway=args.gateway, assume_yes=args.yes
            )
        return share.submit_runs(
            args.out,
            gateway=args.gateway or os.environ.get(share_client.ENV_GATEWAY) or settings.gateway,
            token=token,
            label=args.label or settings.label,
            only=args.run,
            assume_yes=args.yes,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except HwbError as exc:
        log.error("%s", exc)
        return EXIT_ERROR


def _cmd_config(args: argparse.Namespace) -> int:
    setup_logging(None, verbose=args.verbose)
    try:
        if args.action is None and sys.stdin.isatty() and sys.stdout.isatty():
            try:
                hwb_config.menu()
            except (EOFError, KeyboardInterrupt):
                print()
            return 0
        if args.action in (None, "show"):
            settings = hwb_config.load()
            print(f"config file: {hwb_config.path()}")
            for key, about in hwb_config.KEYS.items():
                value = getattr(settings, key)
                print(f"  {key:<8} {value if value is not None else '-':<40} {about}")
            if os.environ.get(hwb_config.ENV_SUBMIT):
                print(f"  ({hwb_config.ENV_SUBMIT} overrides submit)")
            return 0
        if not args.key or (args.action == "set") != (args.value is not None):
            usage = "set KEY VALUE" if args.action == "set" else "unset KEY"
            raise hwb_config.ConfigError(f"usage: hwb config {usage}")
        path = hwb_config.write(args.key, args.value if args.action == "set" else None)
    except hwb_config.ConfigError as exc:
        log.error("%s", exc)
        return EXIT_ERROR
    print(f"{args.key} {'= ' + args.value if args.action == 'set' else 'removed'} in {path}")
    if args.key == "submit" and args.value and args.value.strip().lower() == "auto":
        print(f"{hwb_config.AUTO_WARNING} Undo: hwb config set submit ask")
    return 0


def _cmd_update(args: argparse.Namespace) -> int:
    setup_logging(None, verbose=args.verbose)
    try:
        return update.self_update(apply=args.apply, assume_yes=args.yes)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except HwbError as exc:
        log.error("%s", exc)
        return EXIT_ERROR


def _update_notice() -> None:
    """After an interactive run: one line if a newer release is out (checked at most daily)."""
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return
    try:
        line = update.notice()
    except (HwbError, OSError, ValueError) as exc:  # a notice must never break a run
        log.debug("update notice failed: %s", exc)
        return
    if line:
        print(f"\n{line}", file=sys.stderr)


def _context_runs(out: Path, run_id: str, command: str) -> list[str]:
    """The run just made, plus the newest info run before it for the hardware context."""
    if command in ("info", "all"):
        return [run_id]
    infos = [
        r["id"]
        for r in collect_runs(out)
        if r["report"].get("command") == "info" and r["id"] != run_id
    ]
    return [*infos[-1:], run_id]


def _submit_after_run(args: argparse.Namespace, ctx: RunContext, code: int | None) -> None:
    """After a run: offer to share it ("ask", interactive only), or share it right away
    ("auto": `hwb config set submit auto`). HWB_GATEWAY=off turns both off. Sharing is
    optional and never changes the run's own exit code."""
    if not share_client.sharing_enabled() or code == EXIT_INTERRUPTED:
        return
    try:
        settings = hwb_config.load()
    except hwb_config.ConfigError as exc:
        print(f"not submitted: {exc}", file=sys.stderr)
        return
    gateway = os.environ.get(share_client.ENV_GATEWAY) or settings.gateway
    token = os.environ.get(share_client.ENV_TOKEN)
    try:
        if settings.submit == "auto":
            # A plan that was not applied or a dry run is not a test result worth publishing.
            if args.command == "install" or getattr(args, "dry_run", False):
                return
            print("\nSubmitting (hwb config: submit auto)")
            share.submit_runs(
                args.out,
                gateway=share_client.resolve_gateway(gateway),
                token=token,
                label=settings.label,
                only=_context_runs(args.out, ctx.run_dir.name, args.command),
                assume_yes=True,
            )
        elif settings.submit == "ask" and code is None:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                return
            print()
            share.submit_runs(
                args.out,
                gateway=share_client.resolve_gateway(gateway),
                token=token,
                label=settings.label,
            )
    except (HwbError, KeyboardInterrupt) as exc:
        print(f"not submitted: {exc or 'interrupted'}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "submit":  # works on existing runs; must not create a run directory
        return _cmd_submit(args)
    if args.command == "update":
        return _cmd_update(args)
    if args.command == "config":
        return _cmd_config(args)
    if args.command == "serve" and args.list:  # informational: no run directory
        return _serve_list()
    try:
        ctx = RunContext.create(args.out, args.command)
    except OSError as exc:
        print(f"hwb: cannot create run directory under {args.out}: {exc}", file=sys.stderr)
        return EXIT_ERROR
    setup_logging(ctx.run_dir, verbose=args.verbose)
    signal.signal(signal.SIGTERM, _on_sigterm)
    log.info("hwb %s: %s -> %s", __version__, args.command, ctx.run_dir)
    if not is_linux():
        log.warning("hwb targets Linux; most checks will be skipped or fail on %s", sys.platform)

    code: int | None = None
    try:
        COMMANDS[args.command](ctx, args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        code = EXIT_INTERRUPTED
    except HwbError as exc:
        log.error("%s", exc)
        code = EXIT_ERROR
    except Exception:  # last-resort handler: log traceback, still write the report
        log.exception("unexpected error (please report with %s)", ctx.run_dir / "hwb.log.jsonl")
        code = EXIT_ERROR
    finally:
        report = write_report(ctx)
        if ctx.results:
            print()
            render_console(ctx.results)
        print(f"\nreport: {report}")
        if (index := args.out / INDEX_HTML).exists():
            print(f"view:   {index}  (all runs in {args.out})")
    _submit_after_run(args, ctx, code)
    _update_notice()
    return code if code is not None else ctx.exit_code()


def entry() -> None:
    """For launchers that ignore main()'s return value (zipapp's generated __main__)."""
    sys.exit(main())
