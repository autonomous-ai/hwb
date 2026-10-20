import { SELF, createExecutionContext, env, waitOnExecutionContext } from "cloudflare:test";
import { describe, expect, it } from "vitest";
import broken from "../../ui/src/fixtures/broken.json";
import healthy from "../../ui/src/fixtures/healthy.json";
import bench from "../../ui/src/fixtures/bench.json";
import worker, { purgeExpired } from "../src/index.js";
import { sanitizeBundle } from "../src/sanitize.js";

const BASE = "https://hwb.example.com";
const UPLOAD = { Authorization: "Bearer test-upload-token", "Content-Type": "application/json" };
const ADMIN = { Authorization: "Bearer test-admin-token" };
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

const clone = (v) => structuredClone(v);
const bundleOf = (...reports) => ({ runs: reports.map((r, i) => ({ id: `20261020T06100${i}Z-gpu-rig-01-${r.command}`, report: clone(r) })) });

async function submit(body, headers = UPLOAD) {
  const res = await SELF.fetch(`${BASE}/api/reports`, {
    method: "POST",
    headers,
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
  return { res, body: res.headers.get("Content-Type")?.includes("json") ? await res.json() : await res.text() };
}

// Things a real rig could leak into free text, and things that must survive scrubbing.
const SECRET = [
  "admin@corp.example.com",
  "10.20.30.40",
  "fe80::1ff:fe23:4567:890a",
  "00:1a:2b:3c:4d:5e",
  "GPU-12345678-1234-1234-1234-123456789abc",
  "/home/alice",
  "gpu-rig-01",
];
function leaky() {
  const r = clone(healthy);
  r.results[0].summary = `owner admin@corp.example.com on 10.20.30.40 / fe80::1ff:fe23:4567:890a nic 00:1a:2b:3c:4d:5e`;
  r.results[1].hint = "see /home/alice/hwb-runs/x/hwb.log.jsonl on gpu-rig-01";
  r.results[2].data = { gpu: "GPU-12345678-1234-1234-1234-123456789abc", serial: "PHX123", nested: { mac: "aa:bb" } };
  return r;
}

describe("upload", () => {
  it("is public: no token needed, a wrong token is still refused", async () => {
    const PUBLIC = { "Content-Type": "application/json" };
    expect((await submit(bundleOf(healthy), PUBLIC)).res.status).toBe(201);
    expect((await submit(bundleOf(healthy), { ...PUBLIC, Authorization: "Bearer nope" })).res.status).toBe(401);
  });

  it("can be made private with REQUIRE_UPLOAD_TOKEN", async () => {
    const call = (headers) =>
      worker.fetch(
        new Request(`${BASE}/api/reports`, { method: "POST", headers, body: JSON.stringify(bundleOf(healthy)) }),
        { ...env, REQUIRE_UPLOAD_TOKEN: "true" },
        createExecutionContext(),
      );
    expect((await call({ "Content-Type": "application/json" })).status).toBe(401);
    expect((await call(UPLOAD)).status).toBe(201);
  });

  it("rejects malformed and oversized uploads", async () => {
    expect((await submit("{nope")).res.status).toBe(400);
    expect((await submit({ runs: [] })).res.status).toBe(422);
    expect((await submit({ runs: [{ id: "x", report: { host: "h" } }] })).res.status).toBe(422);
    const big = bundleOf(healthy);
    big.runs[0].report.sections.padding = "x".repeat(600_000);
    const { res } = await submit(big);
    expect(res.status).toBe(413);
    expect(res.headers.get("X-Max-Bytes")).toBe("512000");
    expect((await submit(bundleOf(healthy), { ...UPLOAD, "Content-Type": "text/plain" })).res.status).toBe(415);
  });

  it("stores the report and returns a UUID link, QR matrix and delete token", async () => {
    const { res, body } = await submit(bundleOf(healthy));
    expect(res.status).toBe(201);
    expect(body.id).toMatch(UUID);
    expect(body.url).toBe(`${BASE}/r/${body.id}`);
    expect(res.headers.get("Location")).toBe(body.url);
    expect(body.delete_token).toMatch(/^[0-9a-f]{64}$/);
    expect(body.qr.length).toBeGreaterThan(20);
    expect(body.qr.every((row) => row.length === body.qr.length && /^[01]+$/.test(row))).toBe(true);
    expect(body.retention_days).toBe(90);

    const stored = await (await env.REPORTS.get(`reports/${body.id}.json`)).json();
    expect(stored.runs[0].report.host).toMatch(/^rig-[0-9a-f]{8}$/);
    expect(JSON.stringify(stored)).not.toContain("gpu-rig-01");
    expect(JSON.stringify(stored)).not.toMatch(/"uuid"/);

    const row = await env.DB.prepare("SELECT * FROM reports WHERE id = ?").bind(body.id).first();
    expect(row).toMatchObject({
      verdict: "warn",
      n_warn: 4,
      n_fail: 0,
      gpu_model: "NVIDIA GeForce RTX 5090",
      gpu_count: 2,
      driver_version: "580.65.06",
      cuda_version: "13.0",
      cpu_model: "Intel(R) Xeon(R) Gold 6430",
      cpu_sockets: 2,
      mem_gib: 503.5,
      run_count: 1,
      commands: "all",
      rig_alias: stored.runs[0].report.host,
    });
    const { results } = await env.DB.prepare("SELECT status, COUNT(*) AS n FROM report_checks WHERE report_id = ? GROUP BY status")
      .bind(body.id)
      .all();
    expect(Object.fromEntries(results.map((r) => [r.status, r.n]))).toEqual({ pass: 18, warn: 4, skip: 5 });
  });

  it("scrubs addresses, identifiers and paths but keeps PCI ids and versions", async () => {
    const { res, body } = await submit(bundleOf(leaky()));
    expect(res.status).toBe(201);
    const text = JSON.stringify(await (await env.REPORTS.get(`reports/${body.id}.json`)).json());
    for (const s of SECRET) expect(text).not.toContain(s);
    expect(text).not.toContain("python3");
    expect(text).toContain("NVRM: Xid (PCI:0000:41:00): 13, pid=[pid], name=[process], Graphics Exception");
    expect(text).not.toContain("PHX123");
    expect(text).toContain("~/hwb-runs/x/hwb.log.jsonl");
    for (const keep of ["00000000:41:00.0", "0000:41:00.0", "580.65.06", "12.0"]) expect(text).toContain(keep);
    expect(body.redactions).toMatchObject({ email: 1, ipv4: 1, ipv6: 1, mac: 1, gpu_uuid: 1, home_path: 1, "key:serial": 1 });
  });

  it("indexes several runs as their merged overview", async () => {
    const { body } = await submit({ ...bundleOf(healthy, broken), label: "rack 7 / slot 3" });
    const row = await env.DB.prepare("SELECT verdict, n_fail, run_count, commands, label FROM reports WHERE id = ?").bind(body.id).first();
    expect(row).toEqual({ verdict: "fail", n_fail: 5, run_count: 2, commands: "all,detect", label: "rack 7 / slot 3" });
  });
});

describe("viewing", () => {
  it("serves the hwb viewer with the report embedded and counts views", async () => {
    const summary = "</script><script>alert(1)</script>";
    const r = clone(healthy);
    r.results[0].summary = summary;
    const { body } = await submit(bundleOf(r));

    const ctx = createExecutionContext();
    const res = await worker.fetch(new Request(`${BASE}/r/${body.id}`), env, ctx);
    await waitOnExecutionContext(ctx);
    expect(res.status).toBe(200);
    expect(res.headers.get("Content-Security-Policy")).toContain("connect-src 'none'");
    expect(res.headers.get("X-Robots-Tag")).toContain("noindex");
    const page = await res.text();
    expect(page).not.toContain(summary);
    const data = JSON.parse(page.match(/<script id="hwb-report" type="application\/json">(.*?)<\/script>/s)[1]);
    expect(data.shared.id).toBe(body.id);
    expect(data.runs[0].report.results[0].summary).toBe(summary);
    expect((await env.DB.prepare("SELECT views FROM reports WHERE id = ?").bind(body.id).first()).views).toBe(1);
  });

  it("serves the QR code and the raw JSON", async () => {
    const { body } = await submit(bundleOf(healthy));
    const svg = await SELF.fetch(`${BASE}/r/${body.id}/qr.svg`);
    expect(svg.headers.get("Content-Type")).toBe("image/svg+xml");
    expect(await svg.text()).toMatch(/^<svg /);
    const raw = await SELF.fetch(`${BASE}/api/reports/${body.id}`);
    expect((await raw.json()).runs).toHaveLength(1);
  });

  it("404s unknown or malformed ids without listing anything", async () => {
    for (const path of ["/r/not-a-uuid", `/r/${crypto.randomUUID()}`, "/r/../../etc/passwd"])
      expect((await SELF.fetch(`${BASE}${path}`)).status).toBe(404);
    expect((await SELF.fetch(`${BASE}/api/reports`)).status).toBe(405);
    expect((await SELF.fetch(`${BASE}/`)).status).toBe(200);
    expect((await SELF.fetch(`${BASE}/`, { method: "HEAD" })).status).toBe(200);
  });
});

describe("deleting and retention", () => {
  it("deletes with the uploader's token", async () => {
    const { body } = await submit(bundleOf(healthy));
    const del = (token) => SELF.fetch(`${BASE}/api/reports/${body.id}`, { method: "DELETE", headers: { Authorization: `Bearer ${token}` } });
    expect((await del("wrong")).status).toBe(403);
    expect((await del(body.delete_token)).status).toBe(204);
    expect((await SELF.fetch(`${BASE}/r/${body.id}`)).status).toBe(404);
    expect(await env.REPORTS.get(`reports/${body.id}.json`)).toBeNull();
    expect((await env.DB.prepare("SELECT COUNT(*) AS n FROM report_checks WHERE report_id = ?").bind(body.id).first()).n).toBe(0);
  });

  it("purges reports older than RETENTION_DAYS", async () => {
    const { body } = await submit(bundleOf(healthy));
    expect(await purgeExpired(env, Date.now() + 89 * 86_400_000)).toBe(0);
    expect(await purgeExpired(env, Date.now() + 91 * 86_400_000)).toBeGreaterThan(0);
    expect(await env.REPORTS.get(`reports/${body.id}.json`)).toBeNull();
  });
});

describe("stats", () => {
  it("needs the admin token and aggregates by GPU and check", async () => {
    expect((await SELF.fetch(`${BASE}/api/stats`)).status).toBe(401);
    await submit(bundleOf(healthy));
    await submit(bundleOf(broken));
    const s = await (await SELF.fetch(`${BASE}/api/stats?days=7`, { headers: ADMIN })).json();
    expect(s.totals.reports).toBeGreaterThanOrEqual(2);
    expect(s.by_gpu_model.map((g) => g.gpu_model)).toContain("NVIDIA GeForce RTX 5090");
    expect(s.problem_checks.find((c) => c.name === "gpu.nouveau").fail).toBeGreaterThanOrEqual(1);
  });
});

describe("sanitizeBundle", () => {
  it("never rewrites check names that contain a short host name", async () => {
    const r = clone(healthy);
    r.host = "gpu";
    const { bundle } = await sanitizeBundle({ runs: [{ id: "x-gpu-all", report: r }] });
    expect(bundle.runs[0].report.results.map((c) => c.name)).toContain("gpu.pci");
    expect(bundle.runs[0].report.host).toMatch(/^rig-/);
    expect(bundle.runs[0].id).toMatch(/^\d{8}T\d{6}Z-rig-[0-9a-f]{8}-all$/);
  });
});

describe("limits", () => {
  // Stand-in for Cloudflare's rate-limiting binding: allows `n` calls per key.
  const limiter = (n) => {
    const seen = new Map();
    return { limit: async ({ key }) => ({ success: (seen.set(key, (seen.get(key) ?? 0) + 1), seen.get(key) <= n) }) };
  };
  const post = (e, ip, headers = { "Content-Type": "application/json" }, body = JSON.stringify(bundleOf(healthy))) =>
    worker.fetch(
      new Request(`${BASE}/api/reports`, { method: "POST", headers: { ...headers, "CF-Connecting-IP": ip }, body }),
      e,
      createExecutionContext(),
    );

  it("rate-limits uploads per address, before reading the body", async () => {
    const e = { ...env, UPLOAD_LIMITER: limiter(2) };
    expect((await post(e, "198.51.100.1")).status).toBe(201);
    expect((await post(e, "198.51.100.1")).status).toBe(201);
    const third = await post(e, "198.51.100.1", undefined, "not even json");
    expect(third.status).toBe(429);
    expect(third.headers.get("Retry-After")).toBe("60");
    expect((await post(e, "198.51.100.2")).status).toBe(201); // other addresses unaffected
  });

  it("lets the upload token skip the upload limit", async () => {
    const e = { ...env, UPLOAD_LIMITER: limiter(0) };
    expect((await post(e, "198.51.100.3")).status).toBe(429);
    expect((await post(e, "198.51.100.3", UPLOAD)).status).toBe(201);
  });

  it("rate-limits page and API reads", async () => {
    const { body } = await submit(bundleOf(healthy));
    const e = { ...env, READ_LIMITER: limiter(1) };
    const get = (path) =>
      worker.fetch(new Request(`${BASE}${path}`, { headers: { "CF-Connecting-IP": "198.51.100.4" } }), e, createExecutionContext());
    expect((await get(`/r/${body.id}`)).status).toBe(200);
    expect((await get(`/api/reports/${body.id}`)).status).toBe(429);
  });

  it("stops reading an oversized body that has no Content-Length", async () => {
    let pulled = 0;
    const stream = new ReadableStream({
      pull(c) {
        pulled += 64_000;
        c.enqueue(new Uint8Array(64_000).fill(32));
        if (pulled > 5_000_000) c.close();
      },
    });
    const res = await worker.fetch(
      new Request(`${BASE}/api/reports`, { method: "POST", headers: { "Content-Type": "application/json" }, body: stream, duplex: "half" }),
      env,
      createExecutionContext(),
    );
    expect(res.status).toBe(413);
    expect(pulled).toBeLessThan(1_000_000);
  });
});

describe("hard upload quota", () => {
  const post = (e, ip, headers = { "Content-Type": "application/json" }) =>
    worker.fetch(
      new Request(`${BASE}/api/reports`, {
        method: "POST",
        headers: { ...headers, "CF-Connecting-IP": ip },
        body: JSON.stringify(bundleOf(healthy)),
      }),
      e,
      createExecutionContext(),
    );

  it("caps uploads per client per hour, counting rejected ones, without storing the address", async () => {
    const e = { ...env, UPLOADS_PER_HOUR: "2" };
    expect((await post(e, "203.0.113.7")).status).toBe(201);
    expect((await post(e, "203.0.113.7")).status).toBe(201);
    const third = await post(e, "203.0.113.7");
    expect(third.status).toBe(429);
    expect(Number(third.headers.get("Retry-After"))).toBeGreaterThan(0);
    expect((await post(e, "203.0.113.8")).status).toBe(201);
    expect((await post(e, "203.0.113.7", UPLOAD)).status).toBe(201); // token skips the cap
    const { results } = await env.DB.prepare("SELECT * FROM upload_counters").all();
    expect(JSON.stringify(results)).not.toContain("203.0.113");
    expect(results.map((r) => r.n).sort()).toEqual([1, 3]);
  });

  it("caps the whole gateway per day", async () => {
    const e = { ...env, UPLOADS_PER_DAY: "1" };
    expect((await post(e, "203.0.113.9")).status).toBe(201);
    expect((await post(e, "203.0.113.10")).status).toBe(503);
  });

  it("forgets counters after a day", async () => {
    await post(env, "203.0.113.11");
    await purgeExpired(env, Date.now() + 2 * 86_400_000);
    expect((await env.DB.prepare("SELECT COUNT(*) AS n FROM upload_counters").first()).n).toBe(0);
  });
});

describe("benchmarks", () => {
  it("indexes CPU scores and thermal results, merging benches from separate runs", async () => {
    const cpuRun = structuredClone(bench);
    delete cpuRun.sections.bench.thermal;
    cpuRun.results = cpuRun.results.filter((r) => r.name !== "bench.thermal");
    const thermalRun = structuredClone(bench);
    delete thermalRun.sections.bench.cpu;
    thermalRun.started_at = "2026-10-20T09:00:00+00:00";
    thermalRun.results = [{ name: "bench.thermal", status: "fail", summary: "at limit", hint: "cool it", data: {} }];
    const { res, body } = await submit(bundleOf(cpuRun, thermalRun));
    expect(res.status).toBe(201);
    const row = await env.DB.prepare(
      "SELECT cpu_score, cpu_single_score, cpu_score_method, cpu_temp_peak_c, cpu_temp_margin_c, thermal_verdict FROM reports WHERE id = ?",
    )
      .bind(body.id)
      .first();
    expect(row).toMatchObject({ cpu_score: 26411.3, cpu_single_score: 268.9, cpu_score_method: "stress-ng", thermal_verdict: "fail" });
    expect(row.cpu_temp_peak_c).toBeGreaterThan(80);
    const stats = await (await SELF.fetch(`${BASE}/api/stats`, { headers: ADMIN })).json();
    expect(stats.cpu_bench.some((r) => r.method === "stress-ng" && r.runs >= 1)).toBe(true);
  });
});

describe("favicon", () => {
  const PNG = [0x89, 0x50, 0x4e, 0x47];
  it("serves the autonomous.ai icons and links them from pages", async () => {
    for (const path of ["/favicon.ico", "/favicon.png", "/apple-touch-icon.png"]) {
      const res = await SELF.fetch(`${BASE}${path}`);
      expect(res.status).toBe(200);
      expect(res.headers.get("Content-Type")).toBe("image/png");
      expect(res.headers.get("Cache-Control")).toContain("max-age");
      expect([...new Uint8Array(await res.arrayBuffer()).slice(0, 4)]).toEqual(PNG);
    }
    expect((await SELF.fetch(`${BASE}/favicon.ico`, { method: "HEAD" })).status).toBe(200);
    const landing = await (await SELF.fetch(`${BASE}/`)).text();
    expect(landing).toContain('<link rel="icon" href="/favicon.png"');
    const { body } = await submit(bundleOf(healthy));
    const page = await SELF.fetch(`${BASE}/r/${body.id}`);
    expect(page.headers.get("Content-Security-Policy")).toContain("img-src 'self'");
  });
});
