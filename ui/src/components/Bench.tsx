// Benchmarks tab: headline numbers, findings, time-series charts and a per-phase table for
// each benchmark in sections.bench (cpu, thermal, and whatever hwb adds later).

import type { ReactNode } from "react";
import { label as keyLabel } from "../format";
import type { Row } from "../types";
import { LineChart, SLOTS, type Line, type Phase } from "./Chart";
import { Dash, KV } from "./Data";

interface Series {
  key: string;
  label: string;
  group: string;
  unit: string;
  values: (number | null)[];
}

interface Timeseries {
  t: number[];
  phases: Phase[];
  series: Series[];
  interval_s?: number;
  cpu_crit_c?: number | null;
  downsampled_from?: number;
  csv?: string[];
}

const ORDER = ["serve", "monitor", "cpu", "thermal", "memory", "disk", "gpu"];
const TITLES: Record<string, string> = {
  serve: "Model serving",
  monitor: "Monitor",
  cpu: "CPU",
  thermal: "Thermal",
  memory: "Memory",
  disk: "Disk",
  gpu: "GPU",
};

// One chart per (domain, quantity): never two units on one axis, never more than 8 lines.
const CHARTS: { title: string; match: (k: string) => boolean; limit?: boolean }[] = [
  { title: "CPU temperature", match: (k) => k.startsWith("cpu.temp."), limit: true },
  { title: "CPU and memory utilization", match: (k) => k === "cpu.util" || k === "mem.used" },
  { title: "CPU clock (average)", match: (k) => k === "cpu.mhz" },
  { title: "CPU package power", match: (k) => k === "cpu.power" },
  { title: "Load average (1 min)", match: (k) => k === "load1" },
  { title: "GPU temperature", match: (k) => /^gpu\d+\.temp$/.test(k) },
  { title: "GPU utilization", match: (k) => /^gpu\d+\.util$/.test(k) },
  { title: "GPU power", match: (k) => /^gpu\d+\.power$/.test(k) },
  { title: "GPU SM clock", match: (k) => /^gpu\d+\.sm_mhz$/.test(k) },
  { title: "GPU PCIe link generation", match: (k) => /^gpu\d+\.pcie_gen$/.test(k) },
  { title: "PCIe correctable errors (since start)", match: (k) => /^gpu\d+\.pcie_errors$/.test(k) },
];

/** Colour follows the entity: CPU (and its first sensor) slot 1, further sockets the next
 * slots, memory slot 7; GPU n slot n+1. Stable across every chart of the page. */
export function entityColor(key: string, cpuSensors: string[]): string {
  const gpu = key.match(/^gpu(\d+)\./);
  if (gpu) return SLOTS[Number(gpu[1]) % SLOTS.length]!;
  if (key === "mem.used") return SLOTS[6]!;
  if (key.startsWith("cpu.temp.")) return SLOTS[Math.max(0, cpuSensors.indexOf(key)) % 6]!;
  return SLOTS[0]!;
}

function shortLabel(s: Series): string {
  const gpu = s.key.match(/^gpu(\d+)\./);
  if (gpu) return `GPU${gpu[1]}`;
  if (s.key.startsWith("cpu.temp.")) return s.key.slice("cpu.temp.".length);
  if (s.key === "cpu.util") return "CPU";
  if (s.key === "mem.used") return "Memory";
  return s.label;
}

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const fmt = (v: unknown, digits = 0, unit = "") => {
  const n = num(v);
  return n === null ? null : `${n.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: digits })}${unit}`;
};

function Tiles({ items }: { items: [string, string | null, string?][] }) {
  const shown = items.filter(([, v]) => v !== null);
  if (!shown.length) return null;
  return (
    <dl className="stat-tiles">
      {shown.map(([k, v, sub]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
          {sub && <dd className="stat-sub">{sub}</dd>}
        </div>
      ))}
    </dl>
  );
}

function PcieTable({ pcie }: { pcie: unknown }) {
  if (!pcie || typeof pcie !== "object" || !Object.keys(pcie).length) return null;
  const rows = Object.entries(pcie as Record<string, Row>);
  const val = (v: unknown, prefix = "") => (v === null || v === undefined ? <Dash /> : `${prefix}${String(v)}`);
  return (
    <div className="table-wrap">
      <table className="pcie-table">
        <caption>GPU PCIe links</caption>
        <thead>
          <tr>
            <th>GPU</th>
            <th>Address</th>
            <th>Upstream port</th>
            <th className="num">Max</th>
            <th className="num">Narrowest</th>
            <th className="num">Correctable errors</th>
            <th className="num">Uncorrectable</th>
            <th className="num">Replays</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([gpu, p]) => {
            const narrowed = num(p.min_width) !== null && num(p.max_width) !== null && num(p.min_width)! < num(p.max_width)!;
            return (
              <tr key={gpu}>
                <td>GPU{gpu}</td>
                <td>
                  <code>{String(p.addr ?? "?")}</code>
                </td>
                <td>{p.upstream ? <code>{String(p.upstream)}</code> : <Dash />}</td>
                <td className="num">
                  {val(p.max_gen, "Gen")} x{String(p.max_width ?? "?")}
                </td>
                <td className={`num${narrowed ? " bad" : ""}`}>{val(p.min_width, "x")}</td>
                <td className={`num${num(p.correctable_errors) ? " bad" : ""}`}>{val(p.correctable_errors)}</td>
                <td className={`num${num(p.uncorrectable_errors) ? " bad" : ""}`}>{val(p.uncorrectable_errors)}</td>
                <td className={`num${num(p.replays) ? " bad" : ""}`}>{val(p.replays)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ServeTables({ b }: { b: Row }) {
  const checks = Array.isArray(b.checks) ? (b.checks as Row[]) : [];
  const levels = Array.isArray(b.load) ? (b.load as Row[]) : [];
  const ms = (v: unknown) => fmt(v, 0, " ms") ?? <Dash />;
  return (
    <>
      {checks.length > 0 && (
        <div className="table-wrap">
          <table className="serve-checks">
            <caption>Answer checks (reasoning off, temperature 0)</caption>
            <thead>
              <tr>
                <th>Check</th>
                <th>Result</th>
                <th>Answer</th>
                <th className="num">TTFT</th>
              </tr>
            </thead>
            <tbody>
              {checks.map((c) => (
                <tr key={String(c.name)}>
                  <td>{String(c.name).replace("_", " ")}</td>
                  <td>
                    <span className={`pill ${String(c.status)}`}>{String(c.status)}</span>
                    {c.detail ? <span className="finding-hint"> {String(c.detail)}</span> : null}
                  </td>
                  <td>
                    <code className="answer">{String(c.answer ?? "")}</code>
                  </td>
                  <td className="num">{ms(num(c.ttft_s) === null ? null : num(c.ttft_s)! * 1000)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {levels.length > 0 && (
        <div className="table-wrap">
          <table className="serve-load">
            <caption>Load test ({String(levels[0]?.requests ?? "")}+ requests per level, fixed output length)</caption>
            <thead>
              <tr>
                <th className="num">Concurrent</th>
                <th className="num">OK / errors</th>
                <th className="num">Output tok/s</th>
                <th className="num">Per request tok/s</th>
                <th className="num">TTFT p50 / p99</th>
                <th className="num">TPOT p50</th>
                <th className="num">Longest stall</th>
              </tr>
            </thead>
            <tbody>
              {levels.map((lv) => (
                <tr key={String(lv.concurrency)}>
                  <td className="num">{String(lv.concurrency)}</td>
                  <td className={`num${num(lv.errors) ? " bad" : ""}`}>
                    {String(lv.ok)} / {String(lv.errors)}
                  </td>
                  <td className="num">
                    <strong>{fmt(lv.output_tok_per_s, 0) ?? <Dash />}</strong>
                  </td>
                  <td className="num">{fmt(lv.per_request_tok_per_s, 0) ?? <Dash />}</td>
                  <td className="num">
                    {ms(lv.ttft_ms_p50)} / {ms(lv.ttft_ms_p99)}
                  </td>
                  <td className="num">{ms(lv.tpot_ms_p50)}</td>
                  <td className="num">{ms(lv.max_stall_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {Array.isArray(b.command) && (
        <details className="data">
          <summary>vLLM command</summary>
          <pre className="cmd">{(b.command as string[]).join(" ")}</pre>
        </details>
      )}
    </>
  );
}

function headline(kind: string, b: Row): ReactNode {
  if (kind === "cpu") {
    const method = b.method === "stress-ng" ? "stress-ng matrixprod" : "Python fallback (install stress-ng)";
    return (
      <Tiles
        items={[
          ["All-core", fmt(b.multi_ops_per_s), `${String(b.unit ?? "")} · ${String(b.workers ?? "?")} threads`],
          ["Single-core", fmt(b.single_ops_per_s), String(b.unit ?? "")],
          ["Scaling", fmt(b.scaling_pct, 0, "%"), "all-core ÷ (single × threads)"],
          ["Clock under load", fmt(b.load_mhz, 0, " MHz")],
          ["Peak CPU temp", fmt(b.cpu_temp_peak_c, 0, " °C")],
          ["Package power", fmt(b.load_power_w, 0, " W"), "average under load"],
          ["Load generator", method],
        ]}
      />
    );
  }
  if (kind === "serve") {
    const levels = Array.isArray(b.load) ? (b.load as Row[]) : [];
    const checks = Array.isArray(b.checks) ? (b.checks as Row[]) : [];
    const best = levels.reduce<Row | null>((a, lv) => (num(lv.output_tok_per_s) ?? 0) > (num(a?.output_tok_per_s) ?? -1) ? lv : a, null);
    const single = levels.find((lv) => lv.concurrency === 1);
    const preset = (b.preset ?? null) as Row | null;
    const passed = checks.filter((c) => c.status === "pass").length;
    const counted = checks.filter((c) => c.status !== "skip").length;
    return (
      <Tiles
        items={[
          ["Model", String(b.model ?? preset?.model ?? "?"), preset ? `preset ${String(preset.name)} (${String(preset.status)})` : String(b.endpoint ?? "")],
          ["Best throughput", fmt(best?.output_tok_per_s, 0, " tok/s"), best ? `at ${String(best.concurrency)} concurrent` : undefined],
          ["TTFT, 1 request", fmt(single?.ttft_ms_p50, 0, " ms"), "p50"],
          ["Speed, 1 request", fmt(single?.per_request_tok_per_s, 0, " tok/s"), "per request"],
          ["Answer checks", counted ? `${passed} / ${counted}` : null],
          ["Load time", fmt(b.ready_s, 0, " s"), "until /health"],
        ]}
      />
    );
  }
  if (kind === "monitor") {
    return (
      <Tiles
        items={[
          ["Recorded", fmt(b.seconds, 0, " s"), b.interrupted ? "stopped early" : undefined],
          ["CPU busy", fmt(b.cpu_util_pct, 0, "%"), "average"],
          ["Peak CPU temp", fmt(b.cpu_temp_peak_c, 0, " °C"), num(b.cpu_crit_c) ? `limit ${fmt(b.cpu_crit_c)} °C` : undefined],
          ["Package power", fmt(b.load_power_w, 0, " W"), "average"],
          ["Throttle events", b.cpu_throttle_events === null || b.cpu_throttle_events === undefined ? null : String(b.cpu_throttle_events)],
        ]}
      />
    );
  }
  if (kind === "thermal") {
    return (
      <Tiles
        items={[
          ["Peak CPU temp", fmt(b.cpu_temp_peak_c, 0, " °C"), num(b.cpu_crit_c) ? `limit ${fmt(b.cpu_crit_c)} °C` : undefined],
          ["Headroom", fmt(b.cpu_temp_margin_c, 0, " °C")],
          ["Idle temp", fmt(b.idle_temp_c, 0, " °C")],
          ["Clock sag", fmt(b.cpu_clock_sag_pct, 1, "%"), "start vs end of load"],
          ["End slope", fmt(b.cpu_temp_end_slope_c_per_min, 1, " °C/min")],
          ["Throttle events", b.cpu_throttle_events === null || b.cpu_throttle_events === undefined ? null : String(b.cpu_throttle_events)],
          ["After cooldown", fmt(b.cooldown_end_temp_c, 0, " °C")],
        ]}
      />
    );
  }
  return null;
}

function Findings({ items }: { items: unknown }) {
  if (!Array.isArray(items) || !items.length) return null;
  return (
    <ul className="findings">
      {(items as Row[]).map((f, i) => (
        <li key={i} className={`finding ${String(f.level)}`}>
          <span className={`pill ${f.level === "note" ? "skip" : String(f.level)}`}>{f.level === "note" ? "Note" : String(f.level)}</span>
          <span>
            {String(f.text ?? "")}
            {f.hint ? <span className="finding-hint"> {String(f.hint)}</span> : null}
          </span>
        </li>
      ))}
    </ul>
  );
}

function PhaseTable({ phases }: { phases: unknown }) {
  if (!phases || typeof phases !== "object") return null;
  const rows = Object.entries(phases as Record<string, Row>);
  const cell = (v: unknown, d: number, unit: string) => fmt(v, d, unit) ?? <Dash />;
  return (
    <div className="table-wrap">
      <table className="phase-table">
        <caption>Per-phase summary</caption>
        <thead>
          <tr>
            <th>Phase</th>
            <th className="num">CPU util</th>
            <th className="num">Clock</th>
            <th className="num">Package power</th>
            <th className="num">CPU temp (mean)</th>
            <th className="num">CPU temp (peak)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, p]) => (
            <tr key={name}>
              <td>{name}</td>
              <td className="num">{cell(p.util, 0, "%")}</td>
              <td className="num">{cell(p.mhz, 0, " MHz")}</td>
              <td className="num">{cell(p.power, 0, " W")}</td>
              <td className="num">{cell(p.temp_mean, 1, " °C")}</td>
              <td className="num">{cell(p.temp_peak, 1, " °C")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function toCsv(ts: Timeseries): string {
  const head = ["t_s", "phase", ...ts.series.map((s) => `${s.key}${s.unit ? ` (${s.unit})` : ""}`)];
  const phase = (s: number) => ts.phases.find((p) => s >= p.start && s <= p.end)?.name ?? "";
  const rows = ts.t.map((s, i) => [s, phase(s), ...ts.series.map((x) => x.values[i] ?? "")].join(","));
  return [head.map((h) => (h.includes(",") ? `"${h}"` : h)).join(","), ...rows].join("\n") + "\n";
}

function download(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export function Charts({ ts }: { ts: Timeseries }) {
  if (!ts.t?.length || !ts.series?.length) return <p className="muted">No samples were recorded.</p>;
  const cpuSensors = ts.series.filter((s) => s.key.startsWith("cpu.temp.")).map((s) => s.key);
  const crit = num(ts.cpu_crit_c);
  return (
    <div className="charts">
      {CHARTS.map((c) => {
        const series = ts.series.filter((s) => c.match(s.key));
        if (!series.length) return null;
        const lines: Line[] = series.map((s) => ({
          key: s.key,
          label: shortLabel(s),
          values: s.values,
          color: entityColor(s.key, cpuSensors),
        }));
        return (
          <LineChart
            key={c.title}
            title={c.title}
            unit={series[0]!.unit}
            t={ts.t}
            lines={lines}
            phases={ts.phases}
            limit={c.limit && crit ? { value: crit, label: `limit ${crit} °C` } : null}
          />
        );
      })}
    </div>
  );
}

function BenchCard({ kind, b, runId }: { kind: string; b: Row; runId: string }) {
  const ts = b.timeseries as Timeseries | undefined;
  const { timeseries: _ts, phases, findings, pcie, ...rest } = b;
  const known = ["cpu", "thermal", "monitor", "serve"].includes(kind);
  const gpusWatched = ts?.series?.some((s) => s.key.startsWith("gpu"));
  return (
    <section className="bench" aria-label={`${TITLES[kind] ?? kind} benchmark`}>
      <header className="bench-head">
        <h3>{TITLES[kind] ?? keyLabel(kind)}</h3>
        {ts?.t?.length ? (
          <button type="button" onClick={() => download(`hwb-${runId}-${kind}.csv`, toCsv(ts))}>
            Download CSV
          </button>
        ) : null}
      </header>
      {known ? headline(kind, b) : <KV data={rest} />}
      <Findings items={findings} />
      {kind === "thermal" && gpusWatched && (
        <p className="muted small">GPUs are monitored here, not loaded; GPU load comes with bench gpu.</p>
      )}
      {kind === "monitor" && (
        <p className="muted small">
          Recorded while the machine ran its own workload; hwb added no load. A GPU link below its max generation only counts
          while that GPU is busy: idle GPUs lower it on purpose.
        </p>
      )}
      {ts && <Charts ts={ts} />}
      {ts && (
        <p className="muted small">
          {ts.downsampled_from
            ? `${ts.downsampled_from} samples shown as ${ts.t.length} points (${ts.interval_s} s each; peaks kept for temperatures). `
            : `${ts.t.length} samples every ${ts.interval_s ?? "?"} s. `}
          Full resolution: {(ts.csv ?? []).join(", ") || "CSV files"} in the run directory.
        </p>
      )}
      {kind === "serve" && <ServeTables b={b} />}
      <PcieTable pcie={pcie} />
      <PhaseTable phases={phases} />
    </section>
  );
}

const rank = (kind: string) => (ORDER.includes(kind) ? ORDER.indexOf(kind) : ORDER.length);

export function BenchPanel({ bench, runId }: { bench: Row; runId: string }) {
  const kinds = Object.keys(bench).sort((a, b) => rank(a) - rank(b));
  return (
    <div className="benches">
      {kinds.map((k) => (
        <BenchCard key={k} kind={k} b={bench[k] as Row} runId={runId} />
      ))}
    </div>
  );
}
