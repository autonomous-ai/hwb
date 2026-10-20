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

from . import __version__, bench, serve
from .core.context import RunContext
from .core.errors import HwbError
from .core.log import setup_logging
from .core.platform import is_linux, os_info
from .core.report import render_console, write_report
from .core.report_html import INDEX_HTML
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
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="collect hardware inventory")
    sub.add_parser("detect", help="check GPU driver / CUDA health")

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
    b.add_argument("--duration", type=int, default=60, help="seconds per sustained test")
    b.add_argument(
        "--path", type=Path, default=bench.BenchOptions().path, help="scratch dir for disk tests"
    )

    s = sub.add_parser("serve", help="host a real model with vLLM and load-test it")
    s.add_argument("--model", default=serve.DEFAULT_MODEL)
    s.add_argument("--tp", type=int, help="tensor parallel size (default: all GPUs)")
    s.add_argument("--port", type=int, default=8000)

    a = sub.add_parser("all", help="info + detect + all benchmarks (+ serve with --model)")
    a.add_argument("--duration", type=int, default=60)
    a.add_argument("--model", help="also run the vLLM test with this model")

    sb = sub.add_parser(
        "submit",
        help="share the runs in --out through an hwb gateway (sanitized, after confirmation)",
        description="Upload the runs in --out to an hwb gateway and print the link and a QR code. "
        "Host names, addresses, serials, UUIDs and user paths are removed first; you see a "
        "summary and confirm before anything is sent.",
    )
    sb.add_argument(
        "--gateway",
        help=f"gateway URL (default: ${share_client.ENV_GATEWAY}, else {share_client.DEFAULT_GATEWAY})",
    )
    sb.add_argument("--label", help="short label shown with the report, e.g. a rack position")
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
    return p


def _cmd_info(ctx: RunContext, args: argparse.Namespace) -> None:
    run_info(ctx)


def _cmd_detect(ctx: RunContext, args: argparse.Namespace) -> None:
    run_detect(ctx)


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
    bench.run(ctx, args.kind, bench.BenchOptions(duration_s=args.duration, path=args.path))


def _cmd_serve(ctx: RunContext, args: argparse.Namespace) -> None:
    serve.run(ctx, serve.ServeOptions(model=args.model, tensor_parallel=args.tp, port=args.port))


def _cmd_all(ctx: RunContext, args: argparse.Namespace) -> None:
    run_info(ctx)
    run_detect(ctx)
    bench.run(ctx, "all", bench.BenchOptions(duration_s=args.duration))
    if args.model:
        serve.run(ctx, serve.ServeOptions(model=args.model))


COMMANDS: dict[str, Callable[[RunContext, argparse.Namespace], None]] = {
    "info": _cmd_info,
    "detect": _cmd_detect,
    "install": _cmd_install,
    "bench": _cmd_bench,
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
        if args.delete:
            return share.delete_submission(
                args.out, args.delete, gateway=args.gateway, assume_yes=args.yes
            )
        return share.submit_runs(
            args.out,
            gateway=args.gateway,
            token=token,
            label=args.label,
            only=args.run,
            assume_yes=args.yes,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except HwbError as exc:
        log.error("%s", exc)
        return EXIT_ERROR


def _offer_submit(out: Path) -> None:
    """After an interactive run, offer to share it (the operator still confirms).
    HWB_GATEWAY=off turns the offer off."""
    if not share_client.sharing_enabled() or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    gateway = share_client.resolve_gateway()
    print()
    try:
        share.submit_runs(
            out, gateway=gateway, token=os.environ.get(share_client.ENV_TOKEN), label=None
        )
    except (HwbError, KeyboardInterrupt) as exc:
        # Sharing is optional; it never changes the run's own exit code.
        print(f"not submitted: {exc or 'interrupted'}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "submit":  # works on existing runs; must not create a run directory
        return _cmd_submit(args)
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
    if code is None:
        _offer_submit(args.out)
    return code if code is not None else ctx.exit_code()


def entry() -> None:
    """For launchers that ignore main()'s return value (zipapp's generated __main__)."""
    sys.exit(main())
