// Time-series line chart (SVG, no chart library: the viewer must stay one small file).
// Follows the dataviz rules: one unit per chart (no dual axes), fixed categorical slot per
// entity, 2px lines broken at missing samples, recessive grid, legend for >= 2 series plus
// direct end labels for <= 4, crosshair + tooltip on hover/arrow keys, text in ink colours.

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";

// Reference categorical palette (light), validated against the white surface. Slots below
// 3:1 contrast are relieved by the legend/end labels and the phase table.
export const SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

export interface Line {
  key: string;
  label: string;
  values: (number | null)[];
  color: string;
}

export interface Phase {
  name: string;
  start: number;
  end: number;
}

const HEIGHT = 210;
// Same right margin on every chart so side-by-side time axes line up (room for end labels).
const MARGIN = { top: 22, bottom: 26, left: 46, right: 96 };
// Narrow charts (phones) drop the end labels (the legend names the lines) and their margin.
const NARROW = 480;
const M_NARROW = { ...MARGIN, left: 40, right: 12 };
// Smallest y span per unit, so sensor noise (95.2 vs 95.4 °C) never fills the whole height.
const MIN_SPAN: Record<string, number> = { "°C": 20, MHz: 500, W: 50, "": 4, errors: 5 };
// Discrete states (link generation, cumulative counters) are drawn as steps, not slopes.
const STEPPED = new Set(["gen", "errors"]);
const FALLBACK_WIDTH = 720;

export function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s % 60)).padStart(2, "0")}`;
}

function niceStep(span: number, target: number): number {
  const raw = span / Math.max(target, 1);
  const pow = 10 ** Math.floor(Math.log10(raw || 1));
  const n = raw / pow;
  return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * pow;
}

/** Nice y domain and ticks; percentages are pinned to 0-100. */
export function yScale(values: number[], unit: string, extra: number[] = []): { lo: number; hi: number; ticks: number[] } {
  if (unit === "%") return { lo: 0, hi: 100, ticks: [0, 25, 50, 75, 100] };
  if (unit === "gen") {
    const hi = Math.max(4, ...values.map(Math.ceil));
    return { lo: 0, hi, ticks: Array.from({ length: hi + 1 }, (_, i) => i) };
  }
  const all = [...values, ...extra];
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (!Number.isFinite(lo)) return { lo: 0, hi: 1, ticks: [0, 1] };
  const minSpan = MIN_SPAN[unit] ?? 0;
  if (hi - lo < minSpan) {
    const pad = (minSpan - (hi - lo)) / 2;
    [lo, hi] = [lo - pad, hi + pad];
    if (lo < 0 && Math.min(...all) >= 0) [lo, hi] = [0, hi - lo];
  }
  if (lo === hi) [lo, hi] = [lo - 1, hi + 1];
  if (lo >= 0 && lo < (hi - lo) * 0.5) lo = 0; // start at zero when it costs little
  const step = niceStep(hi - lo, 4);
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = lo; v <= hi + step / 2; v += step) ticks.push(Number(v.toFixed(6)));
  return { lo, hi, ticks };
}

function fmtValue(v: number, unit: string): string {
  if (unit === "gen") return `Gen ${v}`;
  if (unit === "errors") return `${v} errors`;
  const digits = Math.abs(v) >= 100 || unit === "MHz" ? 0 : 1;
  return `${v.toFixed(digits)}${unit ? (unit === "%" ? "%" : ` ${unit}`) : ""}`;
}

/** Path segments, broken wherever a sample is missing; ``step`` holds each value until the next. */
function path(values: (number | null)[], x: (i: number) => number, y: (v: number) => number, step = false): string {
  let d = "";
  let pen = false;
  values.forEach((v, i) => {
    if (v === null) {
      pen = false;
      return;
    }
    if (pen && step) d += `H${x(i).toFixed(1)}`;
    d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  });
  return d;
}

function useWidth() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(FALLBACK_WIDTH);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => e && setWidth(Math.max(280, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, width] as const;
}

export function LineChart({
  title,
  unit,
  t,
  lines,
  phases,
  limit,
}: {
  title: string;
  unit: string;
  t: number[];
  lines: Line[];
  phases: Phase[];
  limit?: { value: number; label: string } | null;
}) {
  const [ref, width] = useWidth();
  const [hover, setHover] = useState<number | null>(null);
  const narrow = width < NARROW;
  const M = narrow ? M_NARROW : MARGIN;
  const endLabels = !narrow && lines.length > 1 && lines.length <= 4;
  const plotW = Math.max(width - M.left - M.right, 40);
  const plotH = HEIGHT - M.top - M.bottom;
  const t0 = t[0] ?? 0;
  const t1 = t[t.length - 1] ?? 1;
  const x = (i: number) => M.left + ((t[i]! - t0) / (t1 - t0 || 1)) * plotW;
  const xt = (s: number) => M.left + ((s - t0) / (t1 - t0 || 1)) * plotW;
  const values = useMemo(() => lines.flatMap((l) => l.values.filter((v): v is number => v !== null)), [lines]);
  const showLimit = limit && values.length && limit.value <= Math.max(...values) * 1.6 ? limit : null;
  const { lo, hi, ticks } = yScale(values, unit, showLimit ? [showLimit.value] : []);
  const y = (v: number) => M.top + plotH - ((v - lo) / (hi - lo || 1)) * plotH;
  const xStep = niceStep(t1 - t0, Math.max(2, Math.floor(plotW / 90)));
  const xTicks: number[] = [];
  for (let s = 0; t0 + s <= t1; s += xStep) xTicks.push(t0 + s); // elapsed time: 0:00, 0:20, ...

  const nearest = (clientX: number, svg: SVGSVGElement) => {
    const px = clientX - svg.getBoundingClientRect().left;
    let best = 0;
    for (let i = 1; i < t.length; i++) if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i;
    return best;
  };
  const onMove = (e: PointerEvent<SVGSVGElement>) => setHover(nearest(e.clientX, e.currentTarget));
  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    setHover((h) => Math.min(Math.max((h ?? (step > 0 ? -1 : t.length)) + step, 0), t.length - 1));
  };

  // End labels: at each line's last value, nudged apart so they never overlap.
  const ends = endLabels
    ? lines
        .map((l) => {
          const i = l.values.map((v, j) => (v === null ? -1 : j)).filter((j) => j >= 0).pop();
          return i === undefined ? null : { l, y: y(l.values[i]!) };
        })
        .filter((e): e is { l: Line; y: number } => e !== null)
        .sort((a, b) => a.y - b.y)
        .reduce<{ l: Line; y: number }[]>((acc, e) => [...acc, { ...e, y: Math.max(e.y, (acc.at(-1)?.y ?? -99) + 13) }], [])
    : [];

  const phaseAt = (s: number) => phases.find((p) => s >= p.start && s <= p.end)?.name;
  const hx = hover !== null ? x(hover) : 0;

  return (
    <figure className="chart" ref={ref}>
      <figcaption>
        <span className="chart-title">{title}</span>
        {unit && <span className="chart-unit">{unit === "%" ? "percent" : unit}</span>}
      </figcaption>
      {lines.length > 1 && (
        <ul className="legend">
          {lines.map((l) => (
            <li key={l.key}>
              <span className="swatch" style={{ background: l.color }} aria-hidden="true" />
              {l.label}
            </li>
          ))}
        </ul>
      )}
      <div className="chart-box">
        <svg
          width={width}
          height={HEIGHT}
          role="img"
          aria-label={`${title}: ${lines.map((l) => l.label).join(", ")} over ${fmtTime(t1 - t0)}`}
          tabIndex={0}
          onPointerMove={onMove}
          onPointerDown={onMove}
          onPointerLeave={(e) => e.pointerType !== "touch" && setHover(null)}
          onKeyDown={onKey}
          onBlur={() => setHover(null)}
        >
          {phases.map((p, i) => (
            <g key={p.name + i} className="phase">
              <rect
                x={xt(p.start)}
                y={M.top}
                width={Math.max(0, xt(p.end) - xt(p.start))}
                height={plotH}
                className={i % 2 ? "band-b" : "band-a"}
              />
              {xt(p.end) - xt(p.start) > p.name.length * 6.2 + 8 && (
                <text x={xt(p.start) + 4} y={M.top - 7} className="phase-label">
                  {p.name}
                </text>
              )}
            </g>
          ))}
          {ticks.map((v) => (
            <g key={v}>
              <line x1={M.left} x2={M.left + plotW} y1={y(v)} y2={y(v)} className="grid" />
              <text x={M.left - 6} y={y(v) + 4} className="tick" textAnchor="end">
                {unit === "gen" ? (v ? `Gen${v}` : "") : v}
              </text>
            </g>
          ))}
          {xTicks.map((s) => (
            <text key={s} x={xt(s)} y={HEIGHT - 8} className="tick" textAnchor="middle">
              {fmtTime(s - t0)}
            </text>
          ))}
          {showLimit && (
            <g className="limit">
              <line x1={M.left} x2={M.left + plotW} y1={y(showLimit.value)} y2={y(showLimit.value)} />
              <text x={M.left + plotW - 4} y={y(showLimit.value) - 5} textAnchor="end">
                {showLimit.label}
              </text>
            </g>
          )}
          {lines.map((l) => (
            <path key={l.key} d={path(l.values, x, y, STEPPED.has(unit))} stroke={l.color} className="line" />
          ))}
          {ends.map((e) => (
            <text key={e.l.key} x={M.left + plotW + 8} y={e.y + 4} className="end-label">
              <tspan fill={e.l.color}>■ </tspan>
              {e.l.label}
            </text>
          ))}
          {hover !== null && (
            <g className="crosshair">
              <line x1={hx} x2={hx} y1={M.top} y2={M.top + plotH} />
              {lines.map((l) =>
                l.values[hover] === null || l.values[hover] === undefined ? null : (
                  <circle key={l.key} cx={hx} cy={y(l.values[hover]!)} r={4} fill={l.color} />
                ),
              )}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div
            className="tooltip"
            aria-live="polite"
            style={{ left: Math.max(0, Math.min(hx + 12, width - 180)), top: M.top }}
          >
            <div className="tt-time">
              {fmtTime(t[hover]! - t0)}
              {phaseAt(t[hover]!) ? ` · ${phaseAt(t[hover]!)}` : ""}
            </div>
            {lines.map((l) => (
              <div key={l.key} className="tt-row">
                <span className="swatch" style={{ background: l.color }} aria-hidden="true" />
                <span>{l.label}</span>
                <b>{l.values[hover] === null || l.values[hover] === undefined ? "—" : fmtValue(l.values[hover]!, unit)}</b>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
