// The page holds every run in one hwb output directory. This module turns that into the
// views the page shows: one Overview per host (latest result of each check), then each run.

import { countByStatus, verdict } from "./format";
import type { CheckResult, Report, Status } from "./types";

export interface Run {
  id: string; // run directory name, relative to report.html
  report: Report;
}

/** Set by the hwb gateway when it serves a submitted report on the web. */
export interface Shared {
  id: string;
  submitted_at: string | null;
  label: string | null;
}

export interface Bundle {
  runs: Run[];
  shared?: Shared;
}

export interface View {
  id: string;
  label: string;
  kind: "overview" | "run";
  report: Report;
  status: Status;
  /** Run directory for links to report.json etc.; absent for an overview. */
  dir?: string;
  /** Overview only: the runs merged into it, and which run each check came from. */
  runs?: Run[];
  source?: Map<string, Run>;
}

const startedMs = (r: Report) => new Date(r.started_at).getTime() || 0;

/** Accepts the bundle hwb writes, or a bare report.json (older pages, `npm run dev`). */
export function normalize(data: unknown): Bundle {
  if (data && typeof data === "object" && Array.isArray((data as Bundle).runs)) return data as Bundle;
  return { runs: [{ id: ".", report: data as Report }] };
}

export function shortTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;
}

export function runLabel(run: Run): string {
  return `${run.report.command} · ${shortTime(run.report.started_at)}`;
}

function exitCodeFor(results: CheckResult[]): number {
  const c = countByStatus(results);
  return c.fail ? 1 : c.error ? 2 : 0;
}

/** Latest result of every check across one host's runs; later runs win. */
export function mergeRuns(runs: Run[], host: string): { report: Report; source: Map<string, Run> } {
  const own = runs.filter((r) => r.report.host === host).sort((a, b) => startedMs(a.report) - startedMs(b.report));
  const latest = new Map<string, CheckResult>();
  const source = new Map<string, Run>();
  const sections: Record<string, unknown> = {};
  for (const run of own) {
    for (const r of run.report.results ?? []) {
      latest.set(r.name, r);
      source.set(r.name, run);
    }
    Object.assign(sections, run.report.sections ?? {});
  }
  const results = [...latest.values()];
  const first = own[0]?.report;
  const last = own[own.length - 1]?.report;
  return {
    source,
    report: {
      schema: last?.schema ?? 1,
      tool: last?.tool ?? { name: "hwb", version: "?" },
      command: "overview",
      host,
      started_at: first?.started_at ?? "",
      finished_at: last?.finished_at ?? "",
      exit_code: exitCodeFor(results),
      results,
      sections,
    },
  };
}

export function buildViews(bundle: Bundle): View[] {
  const runs = [...bundle.runs].sort((a, b) => startedMs(b.report) - startedMs(a.report)); // newest first
  const status = (r: Report) => verdict(countByStatus(r.results ?? []), r.exit_code).status;
  const views: View[] = [];
  if (runs.length > 1) {
    const hosts = [...new Set(runs.map((r) => r.report.host))];
    for (const host of hosts) {
      const { report, source } = mergeRuns(runs, host);
      views.push({
        id: hosts.length > 1 ? `overview-${host}` : "overview",
        label: hosts.length > 1 ? `Overview · ${host}` : "Overview",
        kind: "overview",
        report,
        status: status(report),
        runs: runs.filter((r) => r.report.host === host),
        source,
      });
    }
  }
  for (const run of runs)
    views.push({
      id: run.id,
      label: runLabel(run),
      kind: "run",
      report: run.report,
      status: status(run.report),
      dir: bundle.shared ? undefined : run.id, // no run directories next to a shared page
    });
  return views;
}
