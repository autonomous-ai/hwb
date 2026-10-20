import type { CheckResult, Status } from "./types";

// Severity order: drives the summary tiles and the "needs attention" list.
export const STATUSES: readonly Status[] = ["fail", "error", "warn", "pass", "skip"];

export const LABELS: Record<Status, string> = {
  fail: "Fail",
  error: "Error",
  warn: "Warn",
  pass: "Pass",
  skip: "Skip",
};

export const GROUPS: Record<string, string> = {
  gpu: "GPU driver & CUDA",
  info: "Hardware inventory",
  bench: "Benchmarks",
  serve: "Model serving",
  install: "Installation",
};

// Key suffix -> display unit; the suffix is dropped from the label.
const UNITS: readonly [string, string][] = [
  ["_gib", " GiB"],
  ["_mbps", " Mb/s"],
  ["_mhz", " MHz"],
  ["_c", " °C"],
];

export const TEMP_WARM_C = 70;
export const TEMP_HOT_C = 85;

export function label(key: string): string {
  for (const suffix of [...UNITS.map(([s]) => s), "_s"]) {
    if (key.endsWith(suffix) && key.length > suffix.length) {
      key = key.slice(0, -suffix.length);
      break;
    }
  }
  return key.replace(/_/g, " ");
}

/** Decimal units, matching how disk vendors label capacity (3.84 TB). */
export function siBytes(n: number): string {
  const units = ["B", "kB", "MB", "GB", "TB", "PB"];
  let i = 0;
  while (Math.abs(n) >= 1000 && i < units.length - 1) {
    n /= 1000;
    i++;
  }
  return i === 0 ? `${n.toFixed(0)} ${units[i]}` : `${n.toFixed(2)} ${units[i]}`;
}

export function humanSeconds(s: number): string {
  if (s < 120) return `${s.toFixed(1)} s`;
  let m = Math.floor(s / 60);
  let h = Math.floor(m / 60);
  const d = Math.floor(h / 24);
  m %= 60;
  h %= 24;
  const parts = d ? [`${d}d`] : [];
  if (d || h) parts.push(`${h}h`);
  parts.push(`${m}m`);
  return parts.join(" ");
}

/** Number formatted by what its key says it is; null if the key carries no unit. */
export function withUnit(key: string, value: number): string | null {
  if (key === "size") return siBytes(value);
  if (key.endsWith("_s")) return humanSeconds(value);
  for (const [suffix, unit] of UNITS) if (key.endsWith(suffix)) return `${value}${unit}`;
  return null;
}

export function num(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

export function tempClass(c: number): "hot" | "warm" | "ok" {
  return c >= TEMP_HOT_C ? "hot" : c >= TEMP_WARM_C ? "warm" : "ok";
}

export function countByStatus(results: CheckResult[]): Record<Status, number> {
  const counts: Record<Status, number> = { fail: 0, error: 0, warn: 0, pass: 0, skip: 0 };
  for (const r of results) if (r.status in counts) counts[r.status]++;
  return counts;
}

export interface Verdict {
  status: Status;
  title: string;
  detail: string;
}

export function verdict(counts: Record<Status, number>, exitCode: number): Verdict {
  if (counts.fail)
    return {
      status: "fail",
      title: "Not acceptable",
      detail: `${counts.fail} check(s) failed. Fix them before the rig goes into service.`,
    };
  if (counts.error || exitCode === 2)
    return {
      status: "error",
      title: "Incomplete",
      detail: "hwb itself hit errors, so some checks did not run. See the log.",
    };
  if (counts.warn)
    return {
      status: "warn",
      title: "Acceptable with warnings",
      detail: `${counts.warn} item(s) need an operator's attention.`,
    };
  if (counts.pass)
    return { status: "pass", title: "Acceptable", detail: "Every check that ran passed." };
  return {
    status: "skip",
    title: "Nothing verified",
    detail: "No check produced a verdict; everything was skipped.",
  };
}

export function formatWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
}

export function duration(start: string, end: string): string | null {
  const ms = new Date(end).getTime() - new Date(start).getTime();
  return Number.isFinite(ms) ? humanSeconds(ms / 1000) : null;
}

export function groupOf(name: string): string {
  return name.split(".")[0] ?? name;
}
