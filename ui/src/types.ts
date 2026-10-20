// Mirrors report.json (REPORT_SCHEMA 1) as written by hwb/core/report.py.

export type Status = "pass" | "warn" | "fail" | "skip" | "error";

export interface CheckResult {
  name: string;
  status: Status;
  summary: string;
  hint: string | null;
  data: Record<string, unknown>;
}

export interface Report {
  schema: number;
  tool: { name: string; version: string };
  command: string;
  host: string;
  started_at: string;
  finished_at: string;
  exit_code: number;
  results: CheckResult[];
  sections: Record<string, unknown>;
}

export type Row = Record<string, unknown>;
