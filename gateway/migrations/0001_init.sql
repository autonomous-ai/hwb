-- One row per submitted report. The report itself lives in R2 at reports/<id>.json;
-- these columns are the sanitized facts needed for fleet analysis (no hostnames, serials,
-- addresses or uploader IPs are stored).
CREATE TABLE reports (
  id TEXT PRIMARY KEY,                 -- random UUIDv4, also the public URL
  created_at TEXT NOT NULL,            -- ISO 8601 UTC
  delete_token_sha256 TEXT NOT NULL,   -- the uploader's delete token, hashed
  size_bytes INTEGER NOT NULL,
  hwb_version TEXT,
  report_schema INTEGER,
  rig_alias TEXT,                      -- rig-<8 hex>, a one-way alias of the host name
  label TEXT,                          -- optional operator-chosen label
  run_count INTEGER NOT NULL,
  commands TEXT,                       -- e.g. "info,detect,bench"
  verdict TEXT NOT NULL CHECK (verdict IN ('pass', 'warn', 'fail', 'error', 'skip')),
  n_fail INTEGER NOT NULL DEFAULT 0,
  n_error INTEGER NOT NULL DEFAULT 0,
  n_warn INTEGER NOT NULL DEFAULT 0,
  n_pass INTEGER NOT NULL DEFAULT 0,
  n_skip INTEGER NOT NULL DEFAULT 0,
  gpu_model TEXT,
  gpu_count INTEGER,
  driver_version TEXT,
  cuda_version TEXT,
  cpu_model TEXT,
  cpu_sockets INTEGER,
  mem_gib REAL,
  os TEXT,
  kernel TEXT,
  distro_family TEXT,
  views INTEGER NOT NULL DEFAULT 0,
  last_viewed_at TEXT
);
CREATE INDEX reports_created_at ON reports (created_at);
CREATE INDEX reports_gpu_model ON reports (gpu_model);
CREATE INDEX reports_verdict ON reports (verdict);

-- Latest status of every check in a report: "which checks fail most, on which GPUs".
CREATE TABLE report_checks (
  report_id TEXT NOT NULL REFERENCES reports (id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pass', 'warn', 'fail', 'error', 'skip')),
  PRIMARY KEY (report_id, name)
);
CREATE INDEX report_checks_name_status ON report_checks (name, status);
