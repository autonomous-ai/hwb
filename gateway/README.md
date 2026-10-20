# hwb gateway

A Cloudflare Worker that lets operators share hwb acceptance reports as web pages. The public instance runs at **https://hwb-gateway.autonomous-ai.workers.dev**, and hwb releases submit there by default.

It is **public**: anyone with hwb can submit without a token. Abuse is bounded rather than gated:
- **Native rate limits:** Cloudflare rate-limiting bindings, per IP. Uploads get 5 per minute and page or API reads 120 per minute. They're a cheap first filter, but best-effort: counts sync asynchronously, so short bursts get through.
- **Hard limits in D1:** **30 uploads per hour per client**, counted before the body is read, so rejected and junk uploads count too. There is also **2000 uploads per day for the whole gateway**. Over-limit requests get 429 or 503 with `Retry-After`.
- **Size limit:** 512 kB per upload, enforced while streaming the body. hwb also checks it before sending.
- **Retention:** 90 days.

**Client addresses are never stored.** The hourly counter is keyed by `sha256(RATE_LIMIT_SALT | hour | IP)`. That can't be reversed without the secret salt, changes every hour, and is deleted by the daily cron after a day. An optional `UPLOAD_TOKEN` lets trusted fleets, such as many rigs behind one NAT, skip the upload limits. `REQUIRE_UPLOAD_TOKEN = "true"` makes the gateway private.

```
rig: hwb submit ──(sanitize, preview, confirm)──► POST /api/reports ─► R2   reports/<uuid>.json
                                                                    └─► D1   reports + report_checks
     ◄── URL https://<gateway>/r/<uuid> + QR code + delete token
anyone with the link ──► GET /r/<uuid> ─► the same viewer as hwb-runs/report.html
```

- **Storage:** R2 holds each sanitized report. D1 holds one row per report with the facts used for fleet analysis: verdict, counts, GPU model and count, driver, CUDA, CPU, memory, OS. It also holds the latest status of every check.
- **Links:** reports are **unlisted**. A page is reachable only through its random UUIDv4. There is no index of reports and nothing is crawled (`X-Robots-Tag: noindex`).
- **Viewer:** pages use the hwb viewer (`src/hwb/assets/index.html`, built from `ui/`) with the report embedded. A strict CSP (`connect-src 'none'`) stops a page from sending anything anywhere.

## Privacy

Nothing identifying is stored. Sanitizing happens twice:

1. **On the rig** (`src/hwb/share/sanitize.py`), before anything is sent. The operator sees a summary of what was removed and confirms.
2. **In the Worker** (`src/sanitize.js`), again, in case the client is old or modified.

Removed or replaced:
- host names, replaced by `rig-<8 hex>`, a salted one-way alias that stays stable per rig;
- GPU and MIG UUIDs;
- MAC, IPv4 and IPv6 addresses;
- email addresses;
- home directory paths, replaced by `~`;
- process names in Xid events;
- the keys `uuid`, `serial*`, `mac*`, `ip*`, `hostname`, `fqdn`, `machine_id`, `user*` and `email` wherever they appear.

Run ids are rebuilt from timestamp, alias and command. The uploader's IP address is never stored; see the rate limits above for the hourly salted hash. Keep the two sanitizers in sync, and test both (`tests/test_share.py`, `test/gateway.test.js`).

## API

| method | path | auth | |
|---|---|---|---|
| `POST` | `/api/reports` | none (`Bearer UPLOAD_TOKEN` skips the rate limit) | body `{runs: [{id, report}], label?}`. Returns `201 {id, url, qr, delete_token, created_at, redactions, retention_days}`. `qr` is the module matrix as rows of `0`/`1`, which `hwb` draws in the terminal |
| `GET` | `/r/<id>` | none | the report page |
| `GET` | `/r/<id>/qr.svg` | none | QR code of the page URL |
| `GET` | `/api/reports/<id>` | none | the stored (sanitized) JSON |
| `DELETE` | `/api/reports/<id>` | `Bearer <delete_token>` or `ADMIN_TOKEN` | removes the R2 object and D1 rows |
| `GET` | `/api/stats?days=30` | `Bearer ADMIN_TOKEN` | reports by verdict, by GPU model, the most failing checks, per day |

Limits: `MAX_REPORT_BYTES` (512 kB; larger bodies get 413 and are not read further), 50 runs per report, and the rate limits (429 with `Retry-After: 60`). Malformed input gets a 4xx with `{error}`. Unexpected failures are logged (visible with `wrangler tail`) and return a bare 500.

## Configuration (`wrangler.toml`)

| name | kind | default | |
|---|---|---|---|
| `REPORTS` | R2 binding | bucket `hwb-reports` | report bodies |
| `DB` | D1 binding | database `hwb-reports` | analysis tables (`migrations/`) |
| `UPLOAD_TOKEN` | secret | (none) | optional: clients sending it as `HWB_GATEWAY_TOKEN` skip the upload rate limit |
| `ADMIN_TOKEN` | secret | (none) | `/api/stats`, deleting any report |
| `REQUIRE_UPLOAD_TOKEN` | var | `"false"` | `"true"` makes uploads need `UPLOAD_TOKEN` (private gateway) |
| `UPLOAD_LIMITER`, `READ_LIMITER` | rate-limit bindings | 5 and 120 per 60 s | per client IP, best-effort; `[[unsafe.bindings]]` in `wrangler.toml` |
| `UPLOADS_PER_HOUR` / `UPLOADS_PER_DAY` | var | `"30"` / `"2000"` | hard limits in D1 (per client / whole gateway); `"0"` disables |
| `RATE_LIMIT_SALT` | secret | (none) | salts the hashed client keys; set once to a random value |
| `PUBLIC_BASE_URL` | var | request origin | e.g. a custom domain for the returned links |
| `MAX_REPORT_BYTES` | var | `"512000"` | keep in sync with `MAX_UPLOAD_BYTES` in `src/hwb/share/client.py` |
| `RETENTION_DAYS` | var | `"90"` | a daily cron deletes older reports; `"0"` keeps them forever |

## Deploy

```bash
cd gateway && npm ci
npx wrangler login
npx wrangler r2 bucket create hwb-reports
npx wrangler d1 create hwb-reports          # put the printed database_id into wrangler.toml
npx wrangler secret put ADMIN_TOKEN         # e.g. `openssl rand -hex 32`
npx wrangler secret put RATE_LIMIT_SALT     # random, e.g. `openssl rand -hex 32`
npx wrangler secret put UPLOAD_TOKEN        # optional, see above
npm run deploy                              # embeds the viewer, applies D1 migrations, deploys
```

**Every hwb release redeploys the gateway** (release.yml), since the gateway embeds the report viewer. **Actions → Deploy gateway** redeploys by hand. Both need one repository secret: `CLOUDFLARE_API_TOKEN`, with Workers Scripts, D1 and R2 edit on the account. The account id and D1 database id are committed in `wrangler.toml`. Without the secret, the release job only leaves a notice.

Rigs need no setup: `hwb submit` (or the prompt after any interactive run) uses the built-in `DEFAULT_GATEWAY`. A self-hosted gateway is selected with `HWB_GATEWAY=https://…`. If you change the public URL, update `DEFAULT_GATEWAY` in `src/hwb/share/client.py` before the next release.

## Develop

```bash
make gateway-dev      # local R2 + D1 (miniflare) on http://127.0.0.1:8787, tokens from .dev.vars
make gateway-test     # vitest inside workerd: upload, scrubbing, page, QR, delete, retention, stats
```

The viewer is embedded at build time (`npm run embed` → `src/viewer.gen.js`, gitignored). Rebuild it with `make ui` when `ui/` changes.

## Analysis with D1

```bash
npx wrangler d1 execute hwb-reports --remote --command "
  SELECT gpu_model, COUNT(*) AS rigs, SUM(verdict='fail') AS failed
  FROM reports GROUP BY gpu_model ORDER BY rigs DESC"

npx wrangler d1 execute hwb-reports --remote --command "
  SELECT c.name, SUM(c.status='fail') AS fail, SUM(c.status='warn') AS warn
  FROM report_checks c JOIN reports r ON r.id = c.report_id
  WHERE r.gpu_model LIKE '%5090%' GROUP BY c.name ORDER BY fail DESC, warn DESC LIMIT 10"
```
