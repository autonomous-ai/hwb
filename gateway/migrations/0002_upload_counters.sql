-- Hard upload limits (the native rate-limit binding is best-effort and lets bursts through).
-- key_hash = sha256(RATE_LIMIT_SALT | hour | client IP): not reversible without the secret salt,
-- different every hour, and rows are deleted by the daily cron after a day. No IPs are stored.
CREATE TABLE upload_counters (
  key_hash TEXT NOT NULL,
  window_start TEXT NOT NULL,  -- UTC hour, e.g. 2026-10-20T07
  n INTEGER NOT NULL,
  PRIMARY KEY (key_hash, window_start)
);
CREATE INDEX upload_counters_window ON upload_counters (window_start);
