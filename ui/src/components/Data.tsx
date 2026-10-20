// Generic rendering of arbitrary report data. Used for check details and any section the
// UI has no dedicated view for, so new hwb sections show up without a UI change.

import type { ReactNode } from "react";
import { label, withUnit } from "../format";
import type { Row } from "../types";

const MAX_DEPTH = 5;
const TAG_MAX_LEN = 40;

function isEmpty(v: unknown): boolean {
  if (v === null || v === undefined || v === "") return true;
  if (Array.isArray(v)) return v.length === 0;
  if (typeof v === "object") return Object.keys(v as object).length === 0;
  return false;
}

function isRow(v: unknown): v is Row {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

export function Dash() {
  return <span className="muted">—</span>;
}

export function Value({ value, k = "", depth = 0 }: { value: unknown; k?: string; depth?: number }): ReactNode {
  if (isEmpty(value)) return <Dash />;
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return withUnit(k, value) ?? String(value);
  if (typeof value === "string") return value;
  if (depth >= MAX_DEPTH) return <pre>{JSON.stringify(value, null, 2)}</pre>;
  if (Array.isArray(value)) {
    if (value.every(isRow)) return <Table rows={value} depth={depth} />;
    if (value.every((v) => (typeof v === "string" || typeof v === "number") && String(v).length <= TAG_MAX_LEN))
      return (
        <div className="tags">
          {value.map((v, i) => (
            <span key={i}>{String(v)}</span>
          ))}
        </div>
      );
    return (
      <ul className="lines">
        {value.map((v, i) => (
          <li key={i}>
            <Value value={v} k={k} depth={depth + 1} />
          </li>
        ))}
      </ul>
    );
  }
  if (isRow(value)) return <KV data={value} depth={depth} />;
  return String(value);
}

export function KV({ data, depth = 0 }: { data: Row; depth?: number }) {
  return (
    <dl className="kv">
      {Object.entries(data).map(([k, v]) => (
        <div key={k} className="kv-row">
          <dt>{label(k)}</dt>
          <dd>
            <Value value={v} k={k} depth={depth + 1} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

export type Columns = readonly (readonly [key: string, title: string])[];

export function Table({ rows, columns, depth = 0 }: { rows: Row[]; columns?: Columns; depth?: number }) {
  const cols: Columns =
    columns ?? [...new Set(rows.flatMap((r) => Object.keys(r)))].map((k) => [k, label(k)] as const);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {cols.map(([k, title]) => (
              <th key={k}>{title}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {cols.map(([k]) => (
                <td key={k}>
                  <Value value={r[k]} k={k} depth={depth + 1} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
