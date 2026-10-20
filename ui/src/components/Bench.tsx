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

const ORDER = ["cpu", "thermal", "memory", "disk", "gpu"];
const TITLES: Record<string, string> = { cpu: "CPU", thermal: "Thermal", memory: "Memory", disk: "Disk", gpu: "GPU" };

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
  const { timeseries: _ts, phases, findings, ...rest } = b;
  const known = kind === "cpu" || kind === "thermal";
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
      {ts && <Charts ts={ts} />}
      {ts && (
        <p className="muted small">
          {ts.downsampled_from
            ? `${ts.downsampled_from} samples shown as ${ts.t.length} points (${ts.interval_s} s each; peaks kept for temperatures). `
            : `${ts.t.length} samples every ${ts.interval_s ?? "?"} s. `}
          Full resolution: {(ts.csv ?? []).join(", ") || "CSV files"} in the run directory.
        </p>
      )}
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
