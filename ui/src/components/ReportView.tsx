import { useEffect, useMemo, useState, type ReactNode } from "react";
import { runLabel, type Run, type Shared, type View } from "../bundle";
import {
  GROUPS,
  LABELS,
  STATUSES,
  countByStatus,
  duration,
  formatWhen,
  groupOf,
  label,
  verdict,
} from "../format";
import type { CheckResult, Row, Status } from "../types";
import { BenchPanel } from "./Bench";
import { KV, TableWrap, Value } from "./Data";
import { Card, DriverFacts, InstallPlan, Inventory } from "./Sections";
import { TabPanel, Tabs, type TabItem } from "./Tabs";

const COMPANY_URL = "https://autonomous.ai";
const REPO_URL = "https://github.com/autonomous-ai/hwb";

const KNOWN_SECTIONS = new Set(["inventory", "gpu_facts", "install_plan", "bench"]);
const ALL = () => new Set<Status>(STATUSES);

export type Go = (view: string, tab: string, check?: string) => void;

export function href(view: string, tab: string, check?: string): string {
  return "#/" + [view, tab, ...(check ? [check] : [])].map(encodeURIComponent).join("/");
}

function hasData(r: CheckResult): boolean {
  return r.data !== null && typeof r.data === "object" && Object.keys(r.data).length > 0;
}

export function checkAnchor(name: string): string {
  return `check-${name.replace(/[^\w.-]/g, "_")}`;
}

export function Pill({ status }: { status: Status }) {
  return <span className={`pill ${status}`}>{LABELS[status] ?? status}</span>;
}

function Header({ view }: { view: View }) {
  const { report } = view;
  const v = verdict(countByStatus(report.results ?? []), report.exit_code);
  const system = ((report.sections.inventory as Row | undefined)?.system ?? {}) as Row;
  const meta: [string, ReactNode][] =
    view.kind === "overview"
      ? [
          ["Runs", String(view.runs?.length ?? 0)],
          ["Commands", <code key="c">{[...new Set(view.runs?.map((r) => r.report.command))].join(", ")}</code>],
          ["First run", formatWhen(report.started_at)],
          ["Last run", formatWhen(report.finished_at)],
          ["hwb", report.tool?.version ?? "?"],
        ]
      : [
          ["Command", <code key="c">hwb {report.command}</code>],
          ["Started", formatWhen(report.started_at)],
          ["Duration", duration(report.started_at, report.finished_at) ?? "—"],
          ["hwb", report.tool?.version ?? "?"],
          ["Exit code", String(report.exit_code)],
        ];
  if (typeof system.os === "string")
    meta.splice(1, 0, [
      "OS",
      <>
        {system.os}
        <br />
        <small>{String(system.kernel ?? "")}</small>
      </>,
    ]);
  return (
    <header className="hero">
      <div>
        <p className="kicker">
          {view.kind === "overview"
            ? `hwb acceptance report · latest result of each check across ${view.runs?.length ?? 0} runs`
            : "hwb acceptance report · single run"}
        </p>
        <h1>{report.host}</h1>
      </div>
      <dl className="meta">
        {meta.map(([k, val]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{val}</dd>
          </div>
        ))}
      </dl>
      <div className={`verdict ${v.status}`} role="status">
        <strong>{v.title}</strong>
        <span>{v.detail}</span>
      </div>
    </header>
  );
}

function StatusBar({ counts }: { counts: Record<Status, number> }) {
  const total = STATUSES.reduce((n, s) => n + counts[s], 0) || 1;
  return (
    <div className="bar">
      {STATUSES.filter((s) => counts[s]).map((s) => (
        <span key={s} className={`seg ${s}`} style={{ width: `${(counts[s] / total) * 100}%` }} title={`${counts[s]} ${s}`} />
      ))}
    </div>
  );
}

function Attention({ view, results }: { view: View; results: CheckResult[] }) {
  const rank = (s: Status) => STATUSES.indexOf(s);
  const items = results
    .filter((r) => r.status === "fail" || r.status === "error" || r.status === "warn")
    .sort((a, b) => rank(a.status) - rank(b.status));
  if (!items.length) return <p className="all-clear">Nothing needs attention: no failures, errors or warnings.</p>;
  return (
    <TableWrap>
      <table className="attention">
        <thead>
          <tr>
            <th>Status</th>
            <th>Check</th>
            <th>Finding</th>
            <th>What to do</th>
          </tr>
        </thead>
        <tbody>
          {items.map((r) => (
            <tr key={r.name} className={`row-${r.status}`}>
              <td>
                <Pill status={r.status} />
              </td>
              <td>
                <a href={href(view.id, "checks", r.name)}>
                  <code>{r.name}</code>
                </a>
              </td>
              <td className="text">{r.summary}</td>
              <td className="text todo">{r.hint ?? "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableWrap>
  );
}

function RunsTable({ runs }: { runs: Run[] }) {
  return (
    <TableWrap>
      <table className="runs">
        <thead>
          <tr>
            <th>Run</th>
            <th>Verdict</th>
            {STATUSES.map((s) => (
              <th key={s} className="num">
                {LABELS[s]}
              </th>
            ))}
            <th>Directory</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const c = countByStatus(run.report.results ?? []);
            return (
              <tr key={run.id}>
                <td>
                  <a href={href(run.id, "summary")}>{runLabel(run)}</a>
                </td>
                <td>
                  <Pill status={verdict(c, run.report.exit_code).status} />
                </td>
                {STATUSES.map((s) => (
                  <td key={s} className={`num${c[s] ? "" : " muted"}`}>
                    {c[s]}
                  </td>
                ))}
                <td>
                  <code>{run.id}</code>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </TableWrap>
  );
}

function SummaryPanel({ view, onShowStatus }: { view: View; onShowStatus: (s: Status) => void }) {
  const results = view.report.results ?? [];
  const counts = countByStatus(results);
  return (
    <>
      <StatusBar counts={counts} />
      <div className="tiles" aria-label="Show checks with this status">
        {STATUSES.map((s) => (
          <button key={s} type="button" className={`tile ${s}${counts[s] ? "" : " zero"}`} onClick={() => onShowStatus(s)}>
            <b>{counts[s]}</b>
            <span>{LABELS[s]}</span>
          </button>
        ))}
      </div>
      <h2>Needs attention</h2>
      <Attention view={view} results={results} />
      {view.kind === "overview" && view.runs && (
        <>
          <h2>Runs in this report</h2>
          <RunsTable runs={view.runs} />
        </>
      )}
    </>
  );
}

function CheckRow({
  r,
  open,
  focused,
  source,
  onToggle,
}: {
  r: CheckResult;
  open: boolean;
  focused: boolean;
  source?: Run;
  onToggle: (open: boolean) => void;
}) {
  return (
    <article className={`check${focused ? " focus" : ""}`} id={checkAnchor(r.name)} data-status={r.status}>
      <Pill status={r.status} />
      <code className="name">{r.name}</code>
      <div className="check-body">
        <span className="summary">{r.summary}</span>
        {r.hint && <p className="hint">{r.hint}</p>}
        {source && (
          <p className="source">
            from <a href={href(source.id, "checks", r.name)}>{runLabel(source)}</a>
          </p>
        )}
        {hasData(r) && (
          <details className="data" open={open} onToggle={(e) => onToggle(e.currentTarget.open)}>
            <summary>Details</summary>
            <div className="data-body">
              <KV data={r.data} />
            </div>
          </details>
        )}
      </div>
    </article>
  );
}

function ChecksPanel({
  view,
  active,
  setActive,
  query,
  setQuery,
  focus,
}: {
  view: View;
  active: Set<Status>;
  setActive: (s: Set<Status>) => void;
  query: string;
  setQuery: (q: string) => void;
  focus?: string;
}) {
  const results = view.report.results ?? [];
  const counts = countByStatus(results);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const q = query.trim().toLowerCase();
  const visible = results.filter(
    (r) => active.has(r.status) && (!q || `${r.name} ${r.summary} ${r.hint ?? ""}`.toLowerCase().includes(q)),
  );
  const groups = new Map<string, CheckResult[]>();
  for (const r of visible) groups.set(groupOf(r.name), [...(groups.get(groupOf(r.name)) ?? []), r]);

  const toggleStatus = (s: Status) => {
    const next = new Set(active);
    if (next.has(s)) next.delete(s);
    else next.add(s);
    setActive(next);
  };
  const setOne = (name: string, isOpen: boolean) =>
    setOpen((prev) => {
      if (prev.has(name) === isOpen) return prev;
      const next = new Set(prev);
      if (isOpen) next.add(name);
      else next.delete(name);
      return next;
    });

  return (
    <>
      <div className="toolbar">
        <div className="chips" aria-label="Filter by status">
          {STATUSES.map((s) => (
            <button
              key={s}
              type="button"
              className={`tile ${s}${counts[s] ? "" : " zero"}`}
              aria-pressed={active.has(s)}
              onClick={() => toggleStatus(s)}
            >
              <b>{counts[s]}</b>
              <span>{LABELS[s]}</span>
            </button>
          ))}
        </div>
        <div className="tools">
          <input
            type="search"
            placeholder="Filter checks…"
            aria-label="Filter checks"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button type="button" onClick={() => setOpen(new Set(results.filter(hasData).map((r) => r.name)))}>
            Expand all
          </button>
          <button type="button" onClick={() => setOpen(new Set())}>
            Collapse all
          </button>
        </div>
      </div>
      {visible.length === 0 ? (
        <p className="muted">No checks match the filter.</p>
      ) : (
        <div className="groups">
          {[...groups].map(([prefix, items]) => {
            const c = countByStatus(items);
            return (
              <section className="group" key={prefix} aria-label={GROUPS[prefix] ?? prefix}>
                <header>
                  <h3>{GROUPS[prefix] ?? prefix}</h3>
                  <div className="minis">
                    {STATUSES.filter((s) => c[s]).map((s) => (
                      <span key={s} className={`dot ${s}`} title={`${c[s]} ${s}`}>
                        {c[s]}
                      </span>
                    ))}
                  </div>
                </header>
                {items.map((r) => (
                  <CheckRow
                    key={r.name}
                    r={r}
                    open={open.has(r.name)}
                    focused={r.name === focus}
                    source={view.source?.get(r.name)}
                    onToggle={(o) => setOne(r.name, o)}
                  />
                ))}
              </section>
            );
          })}
        </div>
      )}
    </>
  );
}

function RawPanel({ view }: { view: View }) {
  const json = useMemo(() => JSON.stringify(view.report, null, 2), [view.report]);
  const download = () => {
    const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `hwb-${view.report.host}-${view.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="raw">
      <button type="button" onClick={download}>
        Download JSON
      </button>
      <pre>{json}</pre>
    </div>
  );
}

export function ReportView({
  view,
  tab,
  check,
  go,
  shared,
}: {
  view: View;
  tab: string;
  check?: string;
  go: Go;
  shared?: Shared;
}) {
  const { report } = view;
  const results = report.results ?? [];
  const sections = report.sections ?? {};
  const inv = sections.inventory as Row | undefined;
  const facts = sections.gpu_facts as Row | undefined;
  const plan = sections.install_plan as Row | undefined;
  const bench = sections.bench as Row | undefined;
  const invGpus = ((inv?.gpus as Row | undefined)?.gpus ?? []) as unknown[];
  const dryRun = results.some((r) => r.name.startsWith("install.") && r.status === "skip");
  const other = Object.entries(sections).filter(([k]) => !KNOWN_SECTIONS.has(k));

  const [active, setActive] = useState<Set<Status>>(ALL);
  const [query, setQuery] = useState("");

  // Following a link to one check: make sure it is visible, then bring it into view.
  useEffect(() => {
    if (!check) return;
    setActive(ALL());
    setQuery("");
  }, [check, view.id]);
  useEffect(() => {
    if (tab === "checks" && check) document.getElementById(checkAnchor(check))?.scrollIntoView?.({ block: "center" });
  }, [tab, check, view.id, active, query]);

  const tabs: TabItem[] = [
    { id: "summary", label: "Summary" },
    ...(results.length ? [{ id: "checks", label: <>Checks <span className="count">{results.length}</span></> }] : []),
    ...(bench ? [{ id: "bench", label: "Benchmarks" }] : []),
    ...(inv ? [{ id: "inventory", label: "Inventory" }] : []),
    ...(facts ? [{ id: "driver", label: "GPU driver" }] : []),
    ...(plan ? [{ id: "plan", label: "Install plan" }] : []),
    ...other.map(([k]) => ({ id: `section-${k}`, label: label(k) })),
    ...(view.kind === "run" ? [{ id: "raw", label: "Raw JSON" }] : []),
  ];
  const current = tabs.some((t) => t.id === tab) ? tab : "summary";
  const panel = (id: string, body: ReactNode) => (
    <TabPanel key={id} idPrefix="section" id={id} active={current === id}>
      {body}
    </TabPanel>
  );

  return (
    <main>
      <Header view={view} />
      <Tabs
        items={tabs}
        active={current}
        onSelect={(id) => go(view.id, id)}
        label="Report sections"
        idPrefix="section"
        className="section-tabs"
      />
      {panel(
        "summary",
        <SummaryPanel
          view={view}
          onShowStatus={(s) => {
            setActive(new Set([s]));
            setQuery("");
            go(view.id, "checks");
          }}
        />,
      )}
      {results.length > 0 &&
        panel(
          "checks",
          <ChecksPanel view={view} active={active} setActive={setActive} query={query} setQuery={setQuery} focus={check} />,
        )}
      {bench && panel("bench", <BenchPanel bench={bench} runId={view.id} />)}
      {inv && panel("inventory", <Inventory inv={inv} />)}
      {facts && panel("driver", <DriverFacts facts={facts} showSmi={invGpus.length === 0} />)}
      {plan && panel("plan", <InstallPlan plan={plan} dryRun={dryRun} />)}
      {other.map(([k, v]) =>
        panel(
          `section-${k}`,
          <div className="grid">
            <Card wide title={label(k)}>
              <Value value={v} />
            </Card>
          </div>,
        ),
      )}
      {view.kind === "run" && panel("raw", <RawPanel view={view} />)}
      <footer>
        Generated by hwb {report.tool?.version ?? "?"} (report schema {report.schema}).{" "}
        {shared && (
          <>
            Shared report <code>{shared.id}</code>
            {shared.label ? <> ({shared.label})</> : null}
            {shared.submitted_at ? <>, submitted {formatWhen(shared.submitted_at)}</> : null}. Host names,
            addresses, serials and paths were removed before upload.{" "}
          </>
        )}
        {view.dir ? (
          <>
            Files for this run: <a href={`${view.dir}/report.json`}>report.json</a> ·{" "}
            <a href={`${view.dir}/report.md`}>report.md</a> · <a href={`${view.dir}/hwb.log.jsonl`}>hwb.log.jsonl</a>
          </>
        ) : shared && view.kind === "run" ? null : (
          "The overview shows the most recent result of each check; open a run tab for its full detail."
        )}
        <p className="credit">
          hwb is made by{" "}
          <a href={COMPANY_URL} target="_blank" rel="noopener noreferrer">
            Autonomous
          </a>{" "}
          · open source on{" "}
          <a href={REPO_URL} target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </p>
      </footer>
    </main>
  );
}
