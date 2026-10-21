// Dedicated views for the sections hwb knows about: inventory, gpu_facts, install_plan.

import type { ReactNode } from "react";
import { num, tempClass } from "../format";
import type { Row } from "../types";
import { Dash, KV, Table, TableWrap, Value, type Columns } from "./Data";

function obj(v: unknown): Row {
  return typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Row) : {};
}

function rows(v: unknown): Row[] {
  return Array.isArray(v) ? (v.filter((r) => typeof r === "object" && r !== null) as Row[]) : [];
}

export function Card({ title, wide, children }: { title: ReactNode; wide?: boolean; children: ReactNode }) {
  return (
    <section className={wide ? "card wide" : "card"}>
      <h3>{title}</h3>
      {children}
    </section>
  );
}

export function Temp({ c }: { c: number }) {
  return <span className={`temp ${tempClass(c)}`}>{c.toFixed(0)} °C</span>;
}

export function GpuTable({ gpus }: { gpus: Row[] }) {
  return (
    <TableWrap>
      <table>
        <thead>
          <tr>
            {["#", "Model", "Memory", "SM", "Driver", "PCIe", "Temp", "Power limit", "Persistence", "Bus ID", "Serial"].map(
              (h) => (
                <th key={h}>{h}</th>
              ),
            )}
          </tr>
        </thead>
        <tbody>
          {gpus.map((g, i) => {
            const mem = num(g["memory.total"]);
            const temp = num(g["temperature.gpu"]);
            const power = num(g["power.limit"]);
            const cur = num(g["pcie.link.width.current"]);
            const max = num(g["pcie.link.width.max"]);
            const gen = g["pcie.link.gen.max"];
            const degraded = cur !== null && max !== null && cur < max;
            return (
              <tr key={i}>
                <td>{String(g.index ?? "?")}</td>
                <td>
                  <strong>{String(g.name ?? "?")}</strong>
                </td>
                <td>{mem !== null ? `${(mem / 1024).toFixed(0)} GiB` : <Dash />}</td>
                <td>{String(g.compute_cap ?? "—")}</td>
                <td>{String(g.driver_version ?? "—")}</td>
                <td>
                  <span className={degraded ? "bad" : undefined} title={degraded ? "link narrower than the slot supports" : undefined}>
                    {gen ? `Gen${String(gen)} ` : ""}
                    {cur !== null && max !== null ? `x${cur}/x${max}` : "—"}
                  </span>
                </td>
                <td>{temp !== null ? <Temp c={temp} /> : <Dash />}</td>
                <td>{power !== null ? `${power.toFixed(0)} W` : <Dash />}</td>
                <td>{String(g.persistence_mode ?? "—")}</td>
                <td>
                  <code>{String(g["pci.bus_id"] ?? "—")}</code>
                </td>
                <td>{g.gpu_serial ? <code>{String(g.gpu_serial)}</code> : <Dash />}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </TableWrap>
  );
}

function Sensors({ temps }: { temps: Row }) {
  const chips = Object.entries(temps).filter(([, v]) => typeof v === "object" && v !== null);
  if (!chips.length) return <p className="muted">no hwmon sensors</p>;
  return (
    <div className="sensors">
      {chips.map(([chip, labels]) => (
        <div className="sensor" key={chip}>
          <h4>{chip}</h4>
          <ul>
            {Object.entries(obj(labels)).map(([name, v]) => (
              <li key={name}>
                <span>{name}</span>
                {typeof v === "number" ? <Temp c={v} /> : <Value value={v} />}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

const SERVER_COLUMNS: Columns = [
  ["framework", "Framework"],
  ["model", "Model"],
  ["served_name", "Served as"],
  ["port", "Port"],
  ["gpus", "GPUs"],
  ["used_gib", "GPU memory (GiB)"],
  ["processes", "Processes"],
];
const PROCESS_COLUMNS: Columns = [
  ["gpu", "GPU"],
  ["pid", "PID"],
  ["process", "Process"],
  ["used_mib", "Memory (MiB)"],
];

function Workloads({ data }: { data: Row }) {
  const servers = rows(data.servers).map((s): Row => ({
    ...s,
    gpus: Array.isArray(s.gpus) ? (s.gpus as unknown[]).join(", ") : s.gpus,
    serving: s.serving,
  }));
  const procs = rows(data.processes);
  if (!servers.length && !procs.length) return <p className="muted">GPUs idle: no compute processes.</p>;
  return (
    <>
      {servers.length > 0 && <Table rows={servers} columns={SERVER_COLUMNS} />}
      {servers.some((s) => s.serving) && (
        <p className="muted small">
          Serving (from each server's /v1/models):{" "}
          {servers
            .filter((s) => s.serving)
            .map((s) => `${String(s.framework)} :${String(s.port)} → ${((s.serving as Row).ids as string[] | undefined)?.join(", ") ?? "?"}`)
            .join("; ")}
        </p>
      )}
      {procs.length > 0 && (
        <details className="data">
          <summary>{procs.length} GPU process(es)</summary>
          <div className="data-body">
            <Table rows={procs} columns={PROCESS_COLUMNS} />
          </div>
        </details>
      )}
    </>
  );
}

const WEIGHT_COLUMNS: Columns = [
  ["id", "Model"],
  ["source", "Where"],
  ["size_gb", "Size (GB)"],
  ["architecture", "Architecture"],
  ["format", "Format"],
  ["complete", "Complete"],
];

const DISK_COLUMNS: Columns = [
  ["name", "Device"],
  ["model", "Model"],
  ["size", "Size"],
  ["rota", "Rotational"],
  ["tran", "Transport"],
];

export function Inventory({ inv }: { inv: Row }) {
  const memory = obj(inv.memory);
  const gpus = obj(inv.gpus);
  const gpuRows = rows(gpus.gpus);
  const dimms = rows(memory.dimms);
  const disks = rows(obj(inv.disks).devices);
  const nics = rows(obj(inv.network).nics);
  const temps = obj(inv.sensors).temperatures_c;
  const basics: [string, string][] = [
    ["system", "System"],
    ["board", "Board"],
    ["cpu", "CPU"],
    ["memory", "Memory"],
    ["numa", "NUMA"],
  ];
  return (
    <div className="grid">
      {basics
        .filter(([k]) => k in inv)
        .map(([k, title]) => {
          const { dimms: _skip, ...rest } = obj(inv[k]);
          return (
            <Card key={k} title={title}>
              <KV data={rest} />
            </Card>
          );
        })}
      {gpuRows.length > 0 ? (
        <Card wide title={`GPUs (${gpuRows.length})`}>
          <GpuTable gpus={gpuRows} />
        </Card>
      ) : typeof gpus.error === "string" ? (
        <Card wide title="GPUs">
          <p className="muted">{gpus.error}</p>
        </Card>
      ) : null}
      {typeof inv.workloads === "object" && inv.workloads !== null && (
        <Card wide title="Models on the GPUs">
          <Workloads data={obj(inv.workloads)} />
        </Card>
      )}
      {rows(obj(inv.model_weights).models).length > 0 && (
        <Card wide title={`Model weights on disk (${rows(obj(inv.model_weights).models).length})`}>
          <Table rows={rows(obj(inv.model_weights).models)} columns={WEIGHT_COLUMNS} />
        </Card>
      )}
      {dimms.length > 0 && (
        <Card wide title={`Memory modules (${dimms.length})`}>
          <Table rows={dimms} />
        </Card>
      )}
      {disks.length > 0 && (
        <Card wide title={`Disks (${disks.length})`}>
          <Table rows={disks} columns={DISK_COLUMNS} />
        </Card>
      )}
      {nics.length > 0 && (
        <Card wide title={`Network (${nics.length})`}>
          <Table rows={nics} />
        </Card>
      )}
      {temps !== undefined && (
        <Card wide title="Temperatures">
          <Sensors temps={obj(temps)} />
        </Card>
      )}
    </div>
  );
}

function gen(v: unknown): string {
  return num(v) !== null ? `Gen${String(v)}` : "—";
}

function aer(v: unknown): string {
  const cor = obj(obj(v).correctable);
  const n = num(cor.total);
  if (n === null) return "—";
  const kinds = Object.entries(cor)
    .filter(([k]) => k !== "total")
    .map(([k, c]) => `${k} ${String(c)}`);
  return kinds.length ? `${n} (${kinds.join(", ")})` : String(n);
}

// nvidia-smi bus IDs carry an 8-digit domain (00000000:70:00.0); sysfs uses 4 (0000:70:00.0)
function serialsByAddr(smi: Row[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const g of smi) if (g.gpu_serial && g["pci.bus_id"]) out[String(g["pci.bus_id"]).slice(-12).toLowerCase()] = String(g.gpu_serial);
  return out;
}

const LINK_HEADERS = ["GPU", "Card max", "Slot max", "Slot set to", "Now", "nvidia-smi host max", "Replays", "AER card", "AER port"];

// Both ends of each GPU's link: what the card, the slot and its setting allow, what it runs at,
// and the errors since boot. Register fields need root; "—" means hwb could not read them.
function PcieLinks({ links, smi, serials }: { links: Row[]; smi: Row; serials: Record<string, string> }) {
  return (
    <>
      <TableWrap>
        <table>
          <thead>
            <tr>
              {LINK_HEADERS.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {links.map((l, i) => {
              const g = obj(l.gpu);
              const p = obj(l.port);
              const regs = obj(p.regs);
              const s = obj(smi[String(g.addr)]);
              const target = num(regs.target_gen);
              const slot = Math.min(num(g.max_gen) ?? Infinity, num(p.max_gen) ?? Infinity);
              const capped = target !== null && target < slot;
              const errors = (num(obj(obj(g.aer).correctable).total) ?? 0) + (num(obj(obj(p.aer).correctable).total) ?? 0);
              return (
                <tr key={i}>
                  <td>
                    <code>{String(g.addr ?? "?")}</code>
                    {serials[String(g.addr)] && <div className="muted small">SN {serials[String(g.addr)]}</div>}
                  </td>
                  <td>
                    {gen(g.max_gen)} x{String(g.max_width ?? "?")}
                  </td>
                  <td>{gen(p.max_gen)}</td>
                  <td className={capped ? "bad" : undefined}>{gen(regs.target_gen)}</td>
                  <td>
                    {gen(g.gen)} x{String(g.width ?? "?")}
                  </td>
                  <td>{gen(s.gen_host_max)}</td>
                  <td className={`num${num(s.replays) ? " bad" : ""}`}>{num(s.replays) ?? "—"}</td>
                  <td className={errors ? "bad" : undefined}>{aer(g.aer)}</td>
                  <td className={errors ? "bad" : undefined}>{aer(p.aer)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </TableWrap>
      <p className="muted small">Idle GPUs drop their link to Gen1 to save power; "Now" matters only under load.</p>
    </>
  );
}

function KernelPcie({ k, cooling }: { k: Row; cooling: Row[] }) {
  const lines = Array.isArray(k.sample) ? (k.sample as [number, string][]) : [];
  const types = Object.entries(obj(k.error_types)).map(([t, n]) => `${t} ${String(n)}`);
  const devices = Object.entries(obj(k.devices)).map(([d, n]) => `${d} (${String(n)})`);
  // cur_state only mirrors the link's current speed; a cooling device acts only when a thermal zone drives it
  const active = cooling.filter((c) => Array.isArray(c.zones) && c.zones.length > 0);
  const first = num(k.first_s);
  const last = num(k.last_s);
  return (
    <>
      <p>
        {num(k.lines) ? (
          <>
            <strong>{String(k.lines)}</strong> PCIe/AER/Xid lines this boot: {String(k.corrected ?? 0)} corrected,{" "}
            {String(k.uncorrected ?? 0)} uncorrected
            {first !== null && last !== null && `, from ${Math.round(first)} s to ${Math.round(last)} s after boot`}
          </>
        ) : (
          "No PCIe errors in the kernel log this boot."
        )}
      </p>
      {k.aer_native === false && (
        <p className="muted small">
          The firmware handles PCIe errors (AER), so the OS never sees corrected ones: zero counts here and in the AER
          columns prove nothing.
        </p>
      )}
      {types.length > 0 && <p className="muted small">Error types: {types.join(", ")}</p>}
      {devices.length > 0 && <p className="muted small">Devices: {devices.join(", ")}</p>}
      {active.length > 0 && (
        <p className="bad">
          Thermal zones that can slow a PCIe link:{" "}
          {active
            .map((c) => `${(c.zones as string[]).join(", ")} → ${String(c.type).replace("PCIe_Port_Link_Speed_", "")}`)
            .join("; ")}
        </p>
      )}
      {lines.length > 0 && (
        <details className="data">
          <summary>
            Kernel lines ({lines.length < (num(k.lines) ?? 0) ? `first and last ${lines.length}` : lines.length})
          </summary>
          <pre className="log">{lines.map(([t, m]) => `[${t.toFixed(1).padStart(8)}] ${m}`).join("\n")}</pre>
        </details>
      )}
    </>
  );
}

export function DriverFacts({ facts, showSmi }: { facts: Row; showSmi: boolean }) {
  const { smi_rows, pcie_debug, smi_pci, kernel_pcie, pcie_cooling, ...rest } = facts;
  const smi = rows(smi_rows);
  const links = rows(pcie_debug);
  return (
    <div className="grid">
      {links.length > 0 && (
        <Card wide title="PCIe links">
          <PcieLinks links={links} smi={obj(smi_pci)} serials={serialsByAddr(smi)} />
        </Card>
      )}
      {kernel_pcie !== null && kernel_pcie !== undefined && (
        <Card wide title="Kernel PCIe log">
          <KernelPcie k={obj(kernel_pcie)} cooling={rows(pcie_cooling)} />
        </Card>
      )}
      <Card wide title="Driver facts">
        {showSmi && smi.length > 0 && <GpuTable gpus={smi} />}
        <KV data={rest} />
      </Card>
    </div>
  );
}

export function InstallPlan({ plan, dryRun }: { plan: Row; dryRun: boolean }) {
  const steps = rows(plan.steps);
  return (
    <div className="grid">
      <Card wide title={`Install plan: ${String(plan.target ?? "?")}`}>
        <ol className="steps">
          {steps.map((s, i) => (
            <li key={i}>
              <span className="step-title">{String(s.title ?? "")}</span>
              <pre className="cmd">$ {(Array.isArray(s.argv) ? s.argv : []).map(String).map(shellQuote).join(" ")}</pre>
            </li>
          ))}
        </ol>
        {dryRun && <p className="muted small">Dry run. Re-run with --apply as root to execute these steps.</p>}
      </Card>
    </div>
  );
}

/** Same quoting as Python's shlex.quote, so the commands can be copied as-is. */
export function shellQuote(s: string): string {
  if (s === "") return "''";
  return /^[\w@%+=:,./-]+$/.test(s) ? s : `'${s.replace(/'/g, `'"'"'`)}'`;
}
