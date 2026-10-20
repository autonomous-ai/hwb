-- Benchmark results for fleet analysis (hwb >= 0.3: bench cpu / bench thermal).
ALTER TABLE reports ADD COLUMN cpu_score REAL;              -- all-core ops/s
ALTER TABLE reports ADD COLUMN cpu_single_score REAL;
ALTER TABLE reports ADD COLUMN cpu_score_method TEXT;       -- "stress-ng" or "python" (not comparable)
ALTER TABLE reports ADD COLUMN cpu_temp_peak_c REAL;        -- hottest CPU sensor in any benchmark
ALTER TABLE reports ADD COLUMN cpu_temp_margin_c REAL;      -- thermal bench: limit - peak
ALTER TABLE reports ADD COLUMN cpu_clock_sag_pct REAL;
ALTER TABLE reports ADD COLUMN thermal_verdict TEXT;
CREATE INDEX reports_cpu_method ON reports (cpu_score_method, cpu_model);
