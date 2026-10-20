// hwb gateway: receives sanitized hwb reports, keeps them in R2, indexes them in D1 and serves
// each one as a web page at /r/<uuid>.
//
//   POST   /api/reports          public upload -> {id, url, qr, delete_token}
//   GET    /r/<id>               the report page (the hwb viewer with the data embedded)
//   GET    /r/<id>/qr.svg        QR code of the page URL
//   GET    /api/reports/<id>     the stored (sanitized) JSON
//   DELETE /api/reports/<id>     Bearer delete_token (or ADMIN_TOKEN)
//   GET    /api/stats            fleet analysis from D1 (Bearer ADMIN_TOKEN)
//
// Public by design, so abuse is bounded instead of gated:
//   - Cloudflare rate-limiting bindings per IP: a cheap, best-effort first filter (they are
//     eventually consistent and let short bursts through);
//   - a hard per-client hourly upload cap in D1 (upload_counters, keyed by a salted hourly
//     hash of the IP) and a gateway-wide daily cap;
//   - a 512 kB upload cap enforced while streaming, and retention.
// An UPLOAD_TOKEN, if configured, skips the upload limits for trusted fleets;
// REQUIRE_UPLOAD_TOKEN="true" turns the gateway private again.
//
// Privacy: uploads are scrubbed again here (see sanitize.js); uploader IPs are never stored
// (only the salted hourly hash, deleted after a day); reports are unlisted and only reachable
// through their random UUID.

import { encode, renderSVG } from "uqr";
import { landingPage, notFoundPage, reportPage } from "./pages.js";
import { sanitizeBundle, ValidationError } from "./sanitize.js";
import { summarize } from "./summary.js";
import { UUID_RE, bearer, error, html, intVar, json, randomToken, safeEqual, sha256Hex } from "./util.js";

const DEFAULT_MAX_BYTES = 512_000;
const RETRY_AFTER_S = "60"; // the limiters count per 60 s window
const MAX_RUNS = 50;
const objectKey = (id) => `reports/${id}.json`;

function baseUrl(request, env) {
  return (env.PUBLIC_BASE_URL || new URL(request.url).origin).replace(/\/+$/, "");
}

/** The body as text, or null once it exceeds maxBytes: stops reading there, never buffers more. */
async function readBody(request, maxBytes) {
  if (intVar(request.headers.get("Content-Length"), 0) > maxBytes) return null;
  if (!request.body) return "";
  const reader = request.body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > maxBytes) {
      await reader.cancel();
      return null;
    }
    chunks.push(value);
  }
  const buf = new Uint8Array(size);
  let offset = 0;
  for (const c of chunks) {
    buf.set(c, offset);
    offset += c.byteLength;
  }
  return new TextDecoder().decode(buf);
}

/** True when this client is over the limit. No binding (local dev) means no limit. */
async function overLimit(limiter, request) {
  if (!limiter) return false;
  const key = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const { success } = await limiter.limit({ key });
  return !success;
}

const tooMany = (what, retryAfter = RETRY_AFTER_S) =>
  error(429, `too many ${what} from your address; try again later`, { "Retry-After": retryAfter });

const hourOf = (ms) => new Date(ms).toISOString().slice(0, 13); // 2026-10-20T07

/**
 * Hard limits, counted before the body is read so junk uploads count too. Returns an error
 * Response, or null to proceed.
 */
async function uploadQuota(request, env, now = Date.now()) {
  const perHour = intVar(env.UPLOADS_PER_HOUR, 30);
  const perDay = intVar(env.UPLOADS_PER_DAY, 2000);
  const hour = hourOf(now);
  const ip = request.headers.get("CF-Connecting-IP") ?? "unknown";
  const keyHash = await sha256Hex(`${env.RATE_LIMIT_SALT || env.ADMIN_TOKEN || ""}|${hour}|${ip}`);
  const row = await env.DB.prepare(
    `INSERT INTO upload_counters (key_hash, window_start, n) VALUES (?, ?, 1)
     ON CONFLICT (key_hash, window_start) DO UPDATE SET n = n + 1 RETURNING n`,
  )
    .bind(keyHash, hour)
    .first();
  if (perHour > 0 && row.n > perHour) {
    const nextHour = Math.ceil((now + 1) / 3_600_000) * 3_600_000;
    return tooMany("uploads this hour", String(Math.max(1, Math.ceil((nextHour - now) / 1000))));
  }
  if (perDay > 0) {
    const today = new Date(now).toISOString().slice(0, 10);
    const { n } = await env.DB.prepare("SELECT COUNT(*) AS n FROM reports WHERE created_at >= ?").bind(today).first();
    if (n >= perDay) return error(503, "the gateway has reached its daily upload capacity; try again tomorrow", { "Retry-After": "3600" });
  }
  return null;
}

async function submit(request, env) {
  const token = bearer(request);
  const trusted = Boolean(env.UPLOAD_TOKEN) && (await safeEqual(token, env.UPLOAD_TOKEN));
  if (token && !trusted) return error(401, "wrong upload token", { "WWW-Authenticate": "Bearer" });
  if (env.REQUIRE_UPLOAD_TOKEN === "true" && !trusted) {
    if (!env.UPLOAD_TOKEN) return error(503, "uploads are disabled: the gateway has no UPLOAD_TOKEN configured");
    return error(401, "this gateway needs an upload token", { "WWW-Authenticate": "Bearer" });
  }
  // Before touching the body, so a flood costs a counter lookup and nothing else.
  if (!trusted) {
    if (await overLimit(env.UPLOAD_LIMITER, request)) return tooMany("uploads");
    const refused = await uploadQuota(request, env);
    if (refused) return refused;
  }
  if (!(request.headers.get("Content-Type") ?? "").startsWith("application/json"))
    return error(415, "send application/json");
  const maxBytes = intVar(env.MAX_REPORT_BYTES, DEFAULT_MAX_BYTES);
  const text = await readBody(request, maxBytes);
  if (text === null) return error(413, `report larger than ${maxBytes} bytes`, { "X-Max-Bytes": String(maxBytes) });

  let input;
  try {
    input = JSON.parse(text);
  } catch {
    return error(400, "body is not valid JSON");
  }
  let bundle, redactions;
  try {
    ({ bundle, redactions } = await sanitizeBundle(input, { maxRuns: MAX_RUNS }));
  } catch (e) {
    if (e instanceof ValidationError) return error(422, e.message);
    throw e;
  }

  const id = crypto.randomUUID();
  const createdAt = new Date().toISOString();
  const deleteToken = randomToken();
  const body = JSON.stringify(bundle);
  const { row, checks } = summarize(bundle);

  await env.REPORTS.put(objectKey(id), body, {
    httpMetadata: { contentType: "application/json" },
    customMetadata: { created_at: createdAt },
  });
  const cols = { id, created_at: createdAt, delete_token_sha256: await sha256Hex(deleteToken), size_bytes: body.length, ...row };
  const names = Object.keys(cols);
  try {
    await env.DB.batch([
      env.DB.prepare(`INSERT INTO reports (${names.join(", ")}) VALUES (${names.map(() => "?").join(", ")})`).bind(
        ...Object.values(cols),
      ),
      ...checks.map((c) =>
        env.DB.prepare("INSERT INTO report_checks (report_id, name, status) VALUES (?, ?, ?)").bind(id, c.name, c.status),
      ),
    ]);
  } catch (e) {
    await env.REPORTS.delete(objectKey(id)); // no orphaned report without its index row
    throw e;
  }

  const url = `${baseUrl(request, env)}/r/${id}`;
  const qr = encode(url, { ecc: "M", border: 0 }).data.map((row) => row.map((dark) => (dark ? "1" : "0")).join(""));
  return json(
    {
      id,
      url,
      qr,
      delete_token: deleteToken,
      created_at: createdAt,
      redactions,
      retention_days: intVar(env.RETENTION_DAYS, 0) || null,
      max_bytes: maxBytes,
    },
    201,
    { Location: url },
  );
}

async function load(env, id) {
  const obj = await env.REPORTS.get(objectKey(id));
  return obj ? { bundle: await obj.json(), createdAt: obj.customMetadata?.created_at ?? null } : null;
}

async function viewPage(env, ctx, id) {
  const found = await load(env, id);
  if (!found) return html(notFoundPage(), 404);
  ctx.waitUntil(
    env.DB.prepare("UPDATE reports SET views = views + 1, last_viewed_at = ? WHERE id = ?")
      .bind(new Date().toISOString(), id)
      .run(),
  );
  return html(reportPage(found.bundle, { id, submitted_at: found.createdAt, label: found.bundle.label ?? null }));
}

async function remove(request, env, id) {
  const row = await env.DB.prepare("SELECT delete_token_sha256 FROM reports WHERE id = ?").bind(id).first();
  if (!row) return error(404, "no such report");
  const token = bearer(request);
  const ok =
    (env.ADMIN_TOKEN && (await safeEqual(token, env.ADMIN_TOKEN))) ||
    (token && (await safeEqual(await sha256Hex(token), row.delete_token_sha256)));
  if (!ok) return error(403, "wrong delete token");
  await deleteReports(env, [id]);
  return new Response(null, { status: 204 });
}

async function deleteReports(env, ids) {
  if (!ids.length) return;
  await env.REPORTS.delete(ids.map(objectKey));
  const marks = ids.map(() => "?").join(", ");
  await env.DB.batch([
    env.DB.prepare(`DELETE FROM report_checks WHERE report_id IN (${marks})`).bind(...ids),
    env.DB.prepare(`DELETE FROM reports WHERE id IN (${marks})`).bind(...ids),
  ]);
}

async function stats(request, env) {
  if (!env.ADMIN_TOKEN || !(await safeEqual(bearer(request), env.ADMIN_TOKEN)))
    return error(401, "admin token required", { "WWW-Authenticate": "Bearer" });
  const days = Math.min(Math.max(intVar(new URL(request.url).searchParams.get("days"), 30), 1), 365);
  const since = new Date(Date.now() - days * 86_400_000).toISOString();
  const [totals, byVerdict, byGpu, topChecks, perDay, cpuBench] = await env.DB.batch([
    env.DB.prepare("SELECT COUNT(*) AS reports, COALESCE(SUM(views), 0) AS views FROM reports WHERE created_at >= ?").bind(since),
    env.DB.prepare("SELECT verdict, COUNT(*) AS reports FROM reports WHERE created_at >= ? GROUP BY verdict ORDER BY reports DESC").bind(since),
    env.DB.prepare(
      `SELECT COALESCE(gpu_model, 'unknown') AS gpu_model, COUNT(*) AS reports,
              SUM(verdict = 'fail') AS failed, SUM(verdict = 'warn') AS warned, ROUND(AVG(gpu_count), 1) AS avg_gpus
       FROM reports WHERE created_at >= ? GROUP BY 1 ORDER BY reports DESC LIMIT 50`,
    ).bind(since),
    env.DB.prepare(
      `SELECT c.name, SUM(c.status = 'fail') AS fail, SUM(c.status = 'warn') AS warn,
              SUM(c.status = 'error') AS error, COUNT(*) AS seen
       FROM report_checks c JOIN reports r ON r.id = c.report_id
       WHERE r.created_at >= ? GROUP BY c.name
       HAVING SUM(c.status IN ('fail', 'warn', 'error')) > 0
       ORDER BY fail DESC, warn DESC LIMIT 50`,
    ).bind(since),
    env.DB.prepare(
      `SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS reports, SUM(verdict = 'fail') AS failed
       FROM reports WHERE created_at >= ? GROUP BY day ORDER BY day`,
    ).bind(since),
    // Scores are only comparable within one load generator and CPU model.
    env.DB.prepare(
      `SELECT cpu_model, cpu_score_method AS method, COUNT(*) AS runs,
              ROUND(AVG(cpu_score), 1) AS avg_score, ROUND(MIN(cpu_score), 1) AS min_score,
              ROUND(MAX(cpu_temp_peak_c), 1) AS max_temp_c, SUM(thermal_verdict = 'fail') AS thermal_failed
       FROM reports WHERE created_at >= ? AND cpu_score IS NOT NULL
       GROUP BY cpu_model, cpu_score_method ORDER BY runs DESC LIMIT 50`,
    ).bind(since),
  ]);
  return json({
    since,
    days,
    totals: totals.results[0],
    by_verdict: byVerdict.results,
    by_gpu_model: byGpu.results,
    problem_checks: topChecks.results,
    per_day: perDay.results,
    cpu_bench: cpuBench.results,
  });
}

export async function purgeExpired(env, now = Date.now()) {
  // Rate-limit counters are only needed for the current hour.
  await env.DB.prepare("DELETE FROM upload_counters WHERE window_start < ?").bind(hourOf(now - 86_400_000)).run();
  const days = intVar(env.RETENTION_DAYS, 0);
  if (days <= 0) return 0;
  const cutoff = new Date(now - days * 86_400_000).toISOString();
  let total = 0;
  for (;;) {
    const { results } = await env.DB.prepare("SELECT id FROM reports WHERE created_at < ? LIMIT 500").bind(cutoff).all();
    if (!results.length) return total;
    await deleteReports(env, results.map((r) => r.id));
    total += results.length;
  }
}

async function route(request, env, ctx) {
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  const method = request.method === "HEAD" ? "GET" : request.method; // the runtime drops HEAD bodies

  if (path === "/" && method === "GET") return html(landingPage());
  if (path === "/api/reports") return method === "POST" ? submit(request, env) : error(405, "use POST", { Allow: "POST" });
  if (path === "/api/stats" && method === "GET") return stats(request, env);

  const m = path.match(/^\/(r|api\/reports)\/([^/]+)(\/qr\.svg)?$/);
  if (m) {
    if (method === "GET" && (await overLimit(env.READ_LIMITER, request))) return tooMany("requests");
    const [, kind, id, qr] = m;
    if (!UUID_RE.test(id)) return kind === "r" ? html(notFoundPage(), 404) : error(404, "no such report");
    if (kind === "r" && qr && method === "GET") {
      if (!(await env.REPORTS.head(objectKey(id)))) return error(404, "no such report");
      const svg = renderSVG(`${baseUrl(request, env)}/r/${id}`, { ecc: "M", border: 2 });
      return new Response(svg, { headers: { "Content-Type": "image/svg+xml", "Cache-Control": "public, max-age=86400" } });
    }
    if (kind === "r" && !qr && method === "GET") return viewPage(env, ctx, id);
    if (kind === "api/reports" && !qr && method === "GET") {
      const found = await load(env, id);
      return found ? json(found.bundle) : error(404, "no such report");
    }
    if (kind === "api/reports" && !qr && method === "DELETE") return remove(request, env, id);
    return error(405, "method not allowed");
  }
  return error(404, "not found");
}

export default {
  async fetch(request, env, ctx) {
    try {
      return await route(request, env, ctx);
    } catch (e) {
      // Last-resort boundary: log for `wrangler tail`, never leak internals to the client.
      console.error("unhandled", e?.stack ?? e);
      return error(500, "internal error");
    }
  },
  async scheduled(_event, env, ctx) {
    ctx.waitUntil(purgeExpired(env).then((n) => n && console.log(`purged ${n} expired report(s)`)));
  },
};
