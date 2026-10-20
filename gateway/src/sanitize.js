// Server-side scrubbing: the second line of defence after the client (src/hwb/share/sanitize.py).
// Keep the rules in sync with the Python side. The client knows the real hostname, user and
// paths; the server only sees the upload, so here we apply the generic rules and make sure
// the host is never stored as anything but an opaque alias.

import { sha256Hex } from "./util.js";

// Keys dropped wherever they appear: they identify a unit, a network endpoint or a person.
export const DROP_KEYS = new Set([
  "uuid",
  "serial",
  "serial_number",
  "mac",
  "mac_address",
  "ip",
  "ip_address",
  "ipv4",
  "ipv6",
  "hostname",
  "fqdn",
  "machine_id",
  "username",
  "user",
  "email",
]);

// [rule, pattern, replacement]. Order matters: GPU UUIDs before anything that could eat parts.
export const PATTERNS = [
  ["gpu_uuid", /\b(?:GPU|MIG)-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b/gi, "[gpu-uuid]"],
  ["mac", /\b[0-9a-f]{2}(?::[0-9a-f]{2}){5}\b/gi, "[mac]"],
  ["email", /\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b/g, "[email]"],
  ["ipv4", /(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\w|\.\d)/g, "[ip]"],
  // Needs "::" or eight groups, so PCI addresses like 0000:41:00.0 are left alone.
  [
    "ipv6",
    /(?<![\w:])(?:(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}|(?:[0-9a-f]{1,4}:){1,7}:(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4})*)?|::(?:[0-9a-f]{1,4}:)*[0-9a-f]{1,4})(?![\w:])/gi,
    "[ip]",
  ],
  ["home_path", /(?:\/home|\/Users)\/[^/\s"':]+/g, "~"],
  // Xid lines name the process that hit the error, which can reveal a customer's workload.
  ["xid_process", /(\bpid=)'?[^,\s]+'?(, name=)[^,)]+/g, (_m, pid, name) => `${pid}[pid]${name}[process]`],
];

// eslint-disable-next-line no-control-regex
const CONTROL = /[\u0000-\u0008\u000b-\u001f\u007f]/g;
export const ALIAS = /^rig-[0-9a-f]{8}$/;
const RUN_TS = /^(\d{8}T\d{6}Z)-/;
const MAX_DEPTH = 12;

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Where a host name counts as an occurrence. Distinctive names (6+ chars, or with digits or
 * hyphens, like "gpu-rig-01") match anywhere between non-alphanumerics, including inside
 * paths and run directory names. Short plain names ("gpu") must stand alone, so they never
 * rewrite check names such as "gpu.pci".
 */
export function hostPattern(host) {
  const distinctive = host.length >= 6 || /[\d-]/.test(host);
  const [before, after] = distinctive ? ["(?<![a-z0-9])", "(?![a-z0-9])"] : ["(?<![\\w.-])", "(?![\\w.-])"];
  return new RegExp(`${before}${escapeRe(host)}${after}`, "gi");
}

export function scrubString(s, ctx) {
  let out = s.replace(CONTROL, "");
  for (const [rule, re, repl] of PATTERNS) {
    out = out.replace(re, (...m) => {
      ctx.counts[rule] = (ctx.counts[rule] ?? 0) + 1;
      return typeof repl === "function" ? repl(...m) : repl;
    });
  }
  for (const [re, alias] of ctx.hosts) {
    out = out.replace(re, () => {
      ctx.counts.hostname = (ctx.counts.hostname ?? 0) + 1;
      return alias;
    });
  }
  return out;
}

export function scrub(value, ctx, depth = 0) {
  if (typeof value === "string") return scrubString(value, ctx);
  if (value === null || typeof value !== "object") return value;
  if (depth > MAX_DEPTH) throw new ValidationError("report nests too deeply");
  if (Array.isArray(value)) return value.map((v) => scrub(v, ctx, depth + 1));
  const out = {};
  for (const [k, v] of Object.entries(value)) {
    if (DROP_KEYS.has(k.toLowerCase())) {
      ctx.counts[`key:${k}`] = (ctx.counts[`key:${k}`] ?? 0) + 1;
      continue;
    }
    out[scrubString(k, ctx)] = scrub(v, ctx, depth + 1);
  }
  return out;
}

export class ValidationError extends Error {}

function compactTs(iso) {
  return new Date(iso).toISOString().replace(/[-:]/g, "").replace(/\.\d+Z$/, "Z");
}

const STATUSES = new Set(["pass", "warn", "fail", "skip", "error"]);

function validateReport(r, i) {
  const where = `runs[${i}].report`;
  if (!r || typeof r !== "object" || Array.isArray(r)) throw new ValidationError(`${where} must be an object`);
  if (typeof r.host !== "string" || !r.host) throw new ValidationError(`${where}.host missing`);
  if (typeof r.command !== "string" || !/^[a-z]{1,32}$/.test(r.command))
    throw new ValidationError(`${where}.command invalid`);
  if (typeof r.started_at !== "string" || Number.isNaN(Date.parse(r.started_at)))
    throw new ValidationError(`${where}.started_at invalid`);
  if (!Array.isArray(r.results)) throw new ValidationError(`${where}.results must be a list`);
  r.results.forEach((c, j) => {
    if (!c || typeof c.name !== "string" || !STATUSES.has(c.status) || typeof c.summary !== "string")
      throw new ValidationError(`${where}.results[${j}] invalid`);
  });
}

/**
 * Validate and scrub an upload ({runs: [{id, report}], label?}).
 * Returns {bundle, redactions}; throws ValidationError for malformed input.
 */
export async function sanitizeBundle(input, { maxRuns = 50 } = {}) {
  if (!input || typeof input !== "object" || !Array.isArray(input.runs))
    throw new ValidationError("body must be {runs: [...]}");
  if (input.runs.length === 0 || input.runs.length > maxRuns)
    throw new ValidationError(`runs must hold 1-${maxRuns} entries`);
  input.runs.forEach((run, i) => validateReport(run?.report, i));

  // Every distinct host becomes rig-<8 hex>, unless the client already aliased it.
  const hosts = [];
  for (const { report } of input.runs) {
    if (!ALIAS.test(report.host) && !hosts.some(([, , h]) => h === report.host)) {
      const alias = `rig-${(await sha256Hex(report.host)).slice(0, 8)}`;
      hosts.push([hostPattern(report.host), alias, report.host]);
    }
  }
  const ctx = { counts: {}, hosts: hosts.map(([re, alias]) => [re, alias]) };

  // Run ids are rebuilt as <timestamp>-<alias>-<command> instead of trusting the client's
  // directory names, which embed the real host name.
  const seen = new Set();
  const runs = input.runs.map((run) => {
    const report = scrub(run.report, ctx);
    const host = hosts.find(([, , h]) => h === run.report.host)?.[1] ?? run.report.host;
    const ts = (typeof run.id === "string" && run.id.match(RUN_TS)?.[1]) || compactTs(run.report.started_at);
    let id = `${ts}-${host}-${report.command}`;
    for (let n = 2; seen.has(id); n++) id = `${ts}-${host}-${report.command}-${n}`;
    seen.add(id);
    return { id, report: { ...report, host } };
  });
  let label = null;
  if (typeof input.label === "string" && input.label.trim()) label = scrubString(input.label.trim(), ctx).slice(0, 64);
  return { bundle: { runs, ...(label ? { label } : {}) }, redactions: ctx.counts };
}
