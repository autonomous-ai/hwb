"""Judging benchmark time series. Pure functions of a Timeseries, so they are unit-tested
with synthetic data; thresholds are named constants here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..metrics.sampler import DEFAULT_CPU_CRIT_C, Timeseries

MIN_LOAD_UTIL_PCT = 90.0  # all-core load should keep the CPUs this busy
CRIT_MARGIN_FAIL_C = 3.0  # within this of the sensor's critical temp: at the thermal limit
CRIT_MARGIN_WARN_C = 10.0
CLOCK_SAG_WARN_PCT = 10.0  # clocks lost between the start and end of sustained load
STILL_RISING_C_PER_MIN = 1.0  # temperature still climbing at the end of the load phase
EDGE_FRACTION = 0.2  # "start"/"end" of a phase = its first/last 20 %
GPU_BUSY_PCT = 50.0  # a GPU this busy keeps its link at full speed; idle ones drop it on purpose


def clean(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def mean(values: list[float | None]) -> float | None:
    v = clean(values)
    return sum(v) / len(v) if v else None


def peak(values: list[float | None]) -> float | None:
    v = clean(values)
    return max(v) if v else None


def cpu_temp_keys(ts: Timeseries) -> list[str]:
    return [k for k in ts.series if k.startswith("cpu.temp.")]


def cpu_temp(ts: Timeseries) -> list[float | None]:
    """Hottest CPU sensor at each sample."""
    cols = [ts.values(k) for k in cpu_temp_keys(ts)]
    return [peak([c[i] for c in cols]) for i in range(len(ts.t))]


def edges(values: list[float | None]) -> tuple[list[float | None], list[float | None]]:
    k = max(1, int(len(values) * EDGE_FRACTION))
    return values[:k], values[-k:]


def slope_per_min(t: list[float], values: list[float | None]) -> float | None:
    """Least-squares slope of ``values`` over ``t`` (seconds), per minute."""
    pts = [(x, y) for x, y in zip(t, values, strict=True) if y is not None]
    if len(pts) < 3:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    var = sum((x - mx) ** 2 for x, _ in pts)
    if var == 0:
        return None
    return 60 * sum((x - mx) * (y - my) for x, y in pts) / var


@dataclass
class PhaseStats:
    util: float | None
    mhz: float | None
    power: float | None
    temp_peak: float | None
    temp_mean: float | None


def phase_stats(ts: Timeseries, phase: str) -> PhaseStats:
    s = ts.phase_slice(phase)
    temp = cpu_temp(ts)[s]
    return PhaseStats(
        util=mean(ts.values("cpu.util")[s]),
        mhz=mean(ts.values("cpu.mhz")[s]),
        power=mean(ts.values("cpu.power")[s]),
        temp_peak=peak(temp),
        temp_mean=mean(temp),
    )


@dataclass
class Finding:
    level: str  # "fail" | "warn" | "note"
    text: str
    hint: str | None = None


def thermal_findings(
    ts: Timeseries, load_phase: str, *, sustained_load: bool = True
) -> tuple[list[Finding], dict[str, Any]]:
    """What the phase did to temperatures, clocks, GPUs and PCIe links. Clock sag and
    "still heating" only mean something under our own steady load (``sustained_load``);
    under someone else's workload clocks and temperatures follow that workload."""
    findings: list[Finding] = []
    s = ts.phase_slice(load_phase)
    temp = cpu_temp(ts)
    load_temp, load_t = temp[s], ts.t[s]
    crit = ts.meta.get("cpu_crit_c") or DEFAULT_CPU_CRIT_C
    temp_peak = peak(temp)
    facts: dict[str, Any] = {"cpu_temp_peak_c": temp_peak, "cpu_crit_c": crit}

    if temp_peak is not None:
        margin = crit - temp_peak
        facts["cpu_temp_margin_c"] = round(margin, 1)
        if margin < CRIT_MARGIN_FAIL_C:
            findings.append(
                Finding(
                    "fail",
                    f"CPU reached {temp_peak:.0f} °C, at its {crit:.0f} °C limit",
                    "check heatsink mounting and paste, fan curves/BMC fan mode and chassis airflow",
                )
            )
        elif margin < CRIT_MARGIN_WARN_C:
            findings.append(
                Finding(
                    "warn",
                    f"CPU peaked at {temp_peak:.0f} °C, {margin:.0f} °C below its limit",
                    "little thermal headroom: check fan curves and airflow before production load",
                )
            )

    events = ts.meta.get("cpu_throttle_events")
    facts["cpu_throttle_events"] = events
    if events:
        findings.append(
            Finding(
                "warn",
                f"{events} CPU thermal-throttle events",
                "the CPU cut its clocks to cool down; check cooling and the BIOS power profile",
            )
        )

    first, last = edges(ts.values("cpu.mhz")[s])
    m_first, m_last = mean(first), mean(last)
    if sustained_load and m_first and m_last:
        sag = 100 * (m_first - m_last) / m_first
        facts["cpu_clock_sag_pct"] = round(sag, 1)
        if sag >= CLOCK_SAG_WARN_PCT:
            findings.append(
                Finding(
                    "warn",
                    f"clocks sagged {sag:.0f}% as it heated ({m_first:.0f} → {m_last:.0f} MHz)",
                    "sustained clocks are power/thermal limited; check cooling and power limits",
                )
            )

    _, tail = edges(load_temp)
    rise = slope_per_min(load_t[-len(tail) :], tail) if tail else None
    facts["cpu_temp_end_slope_c_per_min"] = None if rise is None else round(rise, 2)
    if sustained_load and rise is not None and rise > STILL_RISING_C_PER_MIN:
        findings.append(
            Finding(
                "note",
                f"still heating at the end (+{rise:.1f} °C/min); a longer --duration shows the plateau",
            )
        )

    for gpu, n in (ts.meta.get("gpu_slowdown_samples") or {}).items():
        if n:
            findings.append(
                Finding(
                    "warn",
                    f"GPU{gpu} slowed down (thermal/hardware) in {n} samples",
                    "check GPU airflow, fan speed and PCIe power cabling",
                )
            )
    findings += pcie_findings(ts)
    facts["pcie"] = ts.meta.get("pcie") or None
    return findings, facts


PCIE_HINT = (
    "the link retrained to a lower speed or is taking errors: reseat the card and clean the "
    "contacts, replace risers, check slot airflow (it often starts once the chassis is warm), "
    "update the BIOS; forcing the slot one generation lower in the BIOS is the stable fallback"
)


BIOS_GEN_HINT = (
    "the BIOS sets this slot to a lower PCIe generation (often done for a riser): set its link "
    "speed to Auto, then check the link holds the higher generation with no correctable errors "
    "under load (hwb serve, hwb bench thermal); keep the lower setting only if it does not"
)


WIDTH_HINT = (
    "x8 or narrower from the start usually means slot wiring, a riser or the BIOS bifurcation "
    "setting; a link that narrows during the run is marginal: reseat, replace risers, check airflow"
)


def pcie_findings(ts: Timeseries) -> list[Finding]:
    """Per GPU: link below its max generation while busy, narrower link, error counters."""
    findings: list[Finding] = []
    for gpu, m in (ts.meta.get("pcie") or {}).items():
        gens = ts.values(f"gpu{gpu}.pcie_gen")
        util = ts.values(f"gpu{gpu}.util")
        max_gen, port, target = m.get("max_gen"), m.get("port_max_gen"), m.get("port_target_gen")
        if max_gen and port and port < max_gen:
            max_gen = port  # an older slot: the card runs at the slot's best, nothing to fix
        busy = [i for i, u in enumerate(util) if u is not None and u >= GPU_BUSY_PCT]
        if max_gen and busy:
            low = [g for i in busy if i < len(gens) and (g := gens[i]) is not None and g < max_gen]
            if low:
                capped = bool(target and target < max_gen)
                findings.append(
                    Finding(
                        "warn",
                        f"GPU{gpu} PCIe link at Gen{min(low):.0f} (max Gen{max_gen}) while busy, "
                        f"in {len(low)} of {len(busy)} busy samples"
                        + (f"; the BIOS limits the slot to Gen{target}" if capped else ""),
                        BIOS_GEN_HINT if capped else PCIE_HINT,
                    )
                )
        if m.get("max_width") and m.get("min_width") and m["min_width"] < m["max_width"]:
            findings.append(
                Finding(
                    "warn",
                    f"GPU{gpu} PCIe link narrowed to x{m['min_width']} (max x{m['max_width']})",
                    WIDTH_HINT,
                )
            )
        if m.get("uncorrectable_errors"):
            findings.append(
                Finding(
                    "fail",
                    f"{m['uncorrectable_errors']} uncorrectable PCIe errors on GPU{gpu}'s link",
                    "the link is losing data: check dmesg for AER, reseat or move the card, replace risers",
                )
            )
        if m.get("correctable_errors"):
            findings.append(
                Finding(
                    "warn",
                    f"{m['correctable_errors']} correctable PCIe errors on GPU{gpu}'s link",
                    PCIE_HINT,
                )
            )
        if m.get("replays"):
            findings.append(
                Finding(
                    "warn",
                    f"GPU{gpu} retransmitted {m['replays']} PCIe packets (replays)",
                    PCIE_HINT,
                )
            )
    return findings


def verdict(findings: list[Finding]) -> str:
    levels = {f.level for f in findings}
    return "fail" if "fail" in levels else "warn" if "warn" in levels else "pass"
