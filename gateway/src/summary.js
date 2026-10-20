// What D1 keeps about a report for analysis: one row per report plus its check statuses.
// Mirrors the viewer's overview: the latest result of each check across the submitted runs.

const SEVERITY = ["fail", "error", "warn", "pass", "skip"];

const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});
const str = (v, max = 200) => (typeof v === "string" && v ? v.slice(0, max) : null);
const num = (v) => {
  const n = typeof v === "number" ? v : Number.parseFloat(v);
  return Number.isFinite(n) ? n : null;
};

function mostCommon(values) {
  const counts = new Map();
  for (const v of values) if (v) counts.set(v, (counts.get(v) ?? 0) + 1);
  return [...counts].sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
}

export function summarize(bundle) {
  const runs = [...bundle.runs].sort((a, b) => Date.parse(a.report.started_at) - Date.parse(b.report.started_at));
  const latest = new Map();
  const sections = {};
  for (const { report } of runs) {
    for (const r of report.results) latest.set(r.name, r);
    for (const [k, v] of Object.entries(obj(report.sections))) {
      // Benchmarks run separately (bench cpu, then bench thermal): keep the latest of each.
      sections[k] = k === "bench" ? { ...obj(sections[k]), ...obj(v) } : v;
    }
  }
  const checks = [...latest.values()].map((r) => ({ name: r.name.slice(0, 120), status: r.status }));
  const counts = Object.fromEntries(SEVERITY.map((s) => [s, 0]));
  for (const c of checks) counts[c.status]++;
  const verdict = SEVERITY.find((s) => counts[s] > 0 && s !== "skip") ?? "skip";

  const last = runs[runs.length - 1].report;
  const inv = obj(sections.inventory);
  const facts = obj(sections.gpu_facts);
  const system = obj(inv.system);
  const cpu = obj(inv.cpu);
  const gpuRows = [obj(inv.gpus).gpus, facts.smi_rows].find((r) => Array.isArray(r) && r.length) ?? [];
  const smiCheck = latest.get("gpu.nvidia_smi");
  const bench = obj(sections.bench);
  const bcpu = obj(bench.cpu);
  const thermal = obj(bench.thermal);
  const peaks = [num(bcpu.cpu_temp_peak_c), num(thermal.cpu_temp_peak_c)].filter((v) => v !== null);

  return {
    checks,
    row: {
      hwb_version: str(obj(last.tool).version, 32),
      report_schema: num(last.schema),
      rig_alias: str(last.host, 64),
      label: str(bundle.label, 64),
      run_count: runs.length,
      commands: [...new Set(runs.map((r) => r.report.command))].join(","),
      verdict,
      n_fail: counts.fail,
      n_error: counts.error,
      n_warn: counts.warn,
      n_pass: counts.pass,
      n_skip: counts.skip,
      gpu_model: str(mostCommon(gpuRows.map((g) => obj(g).name)), 120),
      gpu_count: gpuRows.length || (Array.isArray(facts.pci) ? facts.pci.length : null) || null,
      driver_version: str(mostCommon(gpuRows.map((g) => obj(g).driver_version)) ?? obj(facts.kernel_driver).version, 32),
      cuda_version: str(facts.cuda_driver_api ?? obj(smiCheck?.data).cuda_driver_api, 16),
      cpu_model: str(Array.isArray(cpu.models) ? cpu.models[0] : null, 160),
      cpu_sockets: num(cpu.sockets),
      mem_gib: num(obj(inv.memory).total_gib),
      os: str(system.os, 120),
      kernel: str(system.kernel, 120),
      distro_family: str(system.distro_family, 16),
      cpu_score: num(bcpu.multi_ops_per_s),
      cpu_single_score: num(bcpu.single_ops_per_s),
      cpu_score_method: str(bcpu.method, 16),
      cpu_temp_peak_c: peaks.length ? Math.max(...peaks) : null,
      cpu_temp_margin_c: num(thermal.cpu_temp_margin_c),
      cpu_clock_sag_pct: num(thermal.cpu_clock_sag_pct),
      thermal_verdict: latest.get("bench.thermal")?.status ?? null,
    },
  };
}
