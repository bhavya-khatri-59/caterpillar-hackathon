"""Anomaly engine: fixed rules (layer 1) + modified z-score against the operator's own
last 20 shifts (layer 2). Every flag carries a one-sentence reason built from the numbers.

Isolation forest (layer 3) is the first item on the cut list and is not built.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from analytics.settings import SETTINGS, Settings
from shared import rule_ids as R
from shared.contracts import AlertEvent, AlertType, AnomalyFlag


def modified_z(x: float, history: Iterable[float]) -> tuple[Optional[float], Optional[float]]:
    """Iglewicz-Hoaglin modified z-score M = 0.6745 (x - median) / MAD. Returns (z, median)."""
    h = np.asarray([v for v in history if v is not None and np.isfinite(v)], dtype=float)
    if len(h) < SETTINGS.min_baseline_n:
        return None, (float(np.median(h)) if len(h) else None)
    med = float(np.median(h))
    mad = float(np.median(np.abs(h - med)))
    if mad == 0:
        mad = 1.253314 * float(np.mean(np.abs(h - med)))   # mean-absolute-deviation fallback
    if mad == 0:
        return None, med
    return 0.6745 * (x - med) / mad, med


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


# ---------------------------------------------------------------------------------------------
# after-hours: engine-meter hours that no logged window accounts for
# ---------------------------------------------------------------------------------------------

def _off_shift_hours(t0: pd.Timestamp, t1: pd.Timestamp, start_h: int, end_h: int) -> float:
    """Hours in (t0, t1] that fall outside the scheduled shift."""
    if t1 <= t0:
        return 0.0
    rng = pd.date_range(t0, t1, freq="15min", inclusive="right")
    off = [(ts - pd.Timedelta(minutes=1)).hour < start_h or (ts - pd.Timedelta(minutes=1)).hour >= end_h for ts in rng]
    return 0.25 * sum(off)


def meter_gaps(df: pd.DataFrame) -> pd.Series:
    """Unaccounted engine hours at each row where the gap since the previous row includes off-shift time.
    0 where the gap is entirely inside the scheduled shift (that's unlogged work, not after-hours)."""
    d = df.sort_values(["machine_id", "timestamp"], kind="stable")
    prev_ts = d.groupby("machine_id")["timestamp"].shift(1)
    unaccounted = d.groupby("machine_id")["engine_hours"].diff() - d["window_min"] / 60
    out = pd.Series(0.0, index=df.index)
    for idx in unaccounted.index[unaccounted > 1e-9]:
        row = d.loc[idx]
        window_start = row.timestamp - pd.Timedelta(minutes=float(row.window_min))
        if _off_shift_hours(prev_ts[idx], window_start, int(row.shift_start_h), int(row.shift_end_h)) > 0:
            out[idx] = round(float(unaccounted[idx]), 2)
    return out


def _after_hours_flag(op: str, date: str, fid: str, unaccounted: float, prev_ts: Optional[pd.Timestamp],
                      ts: pd.Timestamp) -> AnomalyFlag:
    since = f" since {prev_ts:%d %b %H:%M}" if prev_ts is not None else ""
    return AnomalyFlag(
        flag_id=fid, operator_id=op, shift_date=date, rule=R.AFTER_HOURS_RUNNING,
        metric="unaccounted_engine_h", value=round(unaccounted, 2), severity="review",
        reason=(f"Engine meter rose {unaccounted:.1f} h more than the logged work{since} to {ts:%d %b %H:%M}, "
                "across off-shift hours: either unlogged work or after-hours running. Please review."),
    )


# ---------------------------------------------------------------------------------------------
# shift level
# ---------------------------------------------------------------------------------------------

def shift_features(tel: pd.DataFrame) -> pd.DataFrame:
    t = tel.copy()
    t["work_min"] = t["window_min"] - t["idling_time_min"]
    t["stretch_unlogged"] = np.where(t["logged_wait"].astype(bool), 0.0, t["max_idle_stretch_min"].fillna(0.0))
    t["unaccounted_h"] = meter_gaps(t)
    g = t.groupby(["operator_id", "shift_date"], sort=True)
    f = g.agg(
        machine_id=("machine_id", "first"), window_min=("window_min", "sum"), idle_min=("idling_time_min", "sum"),
        work_min=("work_min", "sum"), fuel_l=("fuel_used_l", "sum"), cycles=("load_cycles", "sum"),
        belt_off_min=("belt_off_min", "sum"), belt_off_idle_min=("belt_off_idle_min", "sum"),
        max_stretch_min=("stretch_unlogged", "max"), alerts=("safety_alert_triggered", "sum"),
        unaccounted_h=("unaccounted_h", "sum"), first_ts=("timestamp", "min"),
    ).reset_index()
    f["idle_ratio"] = f.idle_min / f.window_min
    f["fuel_per_cycle"] = np.where(f.cycles > 0, f.fuel_l / f.cycles.clip(lower=1), np.nan)
    f["cycle_time_min"] = np.where(f.cycles > 0, f.work_min / f.cycles.clip(lower=1), np.nan)
    f["belt_off_work_min"] = f.belt_off_min - f.belt_off_idle_min
    f["belt_compliance_pct"] = 100 * (1 - f.belt_off_min / f.window_min)
    f["alerts_per_h"] = f.alerts / (f.window_min / 60)
    return f


def shift_flags(feats: pd.DataFrame, s: Settings = SETTINGS) -> list[AnomalyFlag]:
    """Walks each operator's shifts in order. The baseline is the last 20 *clean* shifts: a shift that
    raised a review or coach flag doesn't feed later baselines, so repeated behaviour can't normalise itself."""
    flags: list[AnomalyFlag] = []
    for op, g in feats.groupby("operator_id", sort=True):
        g = g.sort_values("shift_date").reset_index(drop=True)
        # Drift is sustained, not one slow shift: median of the last few shifts vs the clean shifts before them.
        g["cycle_time_recent"] = g.cycle_time_min.rolling(s.drift_window, min_periods=s.drift_window).median()
        clean: list[int] = []
        for i, r in g.iterrows():
            hist = g.loc[clean[-s.baseline_shifts:]]
            before_window = [j for j in clean if j <= i - s.drift_window][-s.baseline_shifts:]
            new = _flags_for_shift(op, r, hist, g.loc[before_window], s)
            flags.extend(new)
            if not any(f.severity != "info" for f in new):
                clean.append(i)
    return flags


def _flags_for_shift(op: str, r: pd.Series, hist: pd.DataFrame, drift_hist: pd.DataFrame, s: Settings) -> list[AnomalyFlag]:
    date, out = r.shift_date, []

    def fid(rule: str) -> str:
        return f"{op}-{date}-{rule}"

    if r.belt_off_idle_min > s.belt_off_min:
        fuel = r.belt_off_idle_min / 60 * s.idle_burn_l_h
        out.append(AnomalyFlag(
            flag_id=fid(R.BELT_OFF_IDLE), operator_id=op, shift_date=date, rule=R.BELT_OFF_IDLE,
            metric="belt_off_idle_min", value=float(r.belt_off_idle_min), severity="coach",
            reason=(f"Engine idling with the belt off for {r.belt_off_idle_min:.0f} min this shift, "
                    f"which suggests the seat was left with the engine running; about {fuel:.1f} L of fuel."),
        ))
    if r.belt_off_work_min > s.belt_off_min:
        out.append(AnomalyFlag(
            flag_id=fid(R.BELT_OFF_ENGINE_ON), operator_id=op, shift_date=date, rule=R.BELT_OFF_ENGINE_ON,
            metric="belt_off_work_min", value=float(r.belt_off_work_min), severity="coach",
            reason=f"Operated for about {r.belt_off_work_min:.0f} min with the belt off; belt compliance {r.belt_compliance_pct:.0f}%.",
        ))

    z, med = modified_z(r.idle_ratio, hist.idle_ratio)
    if r.idle_ratio > s.idle_ratio_shift_max or (z is not None and z > s.z_cutoff):
        usual = f" vs your usual {_pct(med)}" if med is not None else ""
        out.append(AnomalyFlag(
            flag_id=fid(R.IDLE_RATIO_HIGH), operator_id=op, shift_date=date, rule=R.IDLE_RATIO_HIGH,
            metric="idle_ratio", value=round(float(r.idle_ratio), 3), baseline=_r(med), z=_r(z), severity="coach",
            reason=(f"Idle {r.idle_min:.0f} of {r.window_min:.0f} min ({_pct(r.idle_ratio)}){usual}; "
                    f"about {r.idle_min / 60 * s.idle_burn_l_h:.1f} L burnt at idle."),
        ))

    if r.max_stretch_min > s.idle_continuous_min:
        out.append(AnomalyFlag(
            flag_id=fid(R.IDLE_CONTINUOUS), operator_id=op, shift_date=date, rule=R.IDLE_CONTINUOUS,
            metric="max_idle_stretch_min", value=float(r.max_stretch_min), severity="info",
            reason=(f"Idled {r.max_stretch_min:.0f} min in one stretch with no logged truck wait "
                    f"(auto idle shutdown can be set from 5 to 60 min)."),
        ))

    if r.unaccounted_h > s.after_hours_unaccounted_h:
        out.append(_after_hours_flag(op, date, fid(R.AFTER_HOURS_RUNNING), float(r.unaccounted_h), None, r.first_ts))

    recent = r.get("cycle_time_recent", np.nan)
    if np.isfinite(recent):
        z, med = modified_z(recent, drift_hist.cycle_time_min)
        if z is not None and z > s.z_cutoff:
            out.append(AnomalyFlag(
                flag_id=fid(R.CYCLE_TIME_DRIFT), operator_id=op, shift_date=date, rule=R.CYCLE_TIME_DRIFT,
                metric="cycle_time_min_3shift", value=round(float(recent), 2), baseline=_r(med), z=_r(z),
                severity="review",
                reason=(f"Over your last {s.drift_window} shifts, load cycles took {recent:.1f} min on average "
                        f"vs your usual {med:.1f} min ({100 * (recent / med - 1):+.0f}%)."),
            ))

    if np.isfinite(r.fuel_per_cycle):
        z, med = modified_z(r.fuel_per_cycle, hist.fuel_per_cycle)
        if z is not None and z > s.z_cutoff:
            out.append(AnomalyFlag(
                flag_id=fid(R.FUEL_PER_CYCLE_HIGH), operator_id=op, shift_date=date, rule=R.FUEL_PER_CYCLE_HIGH,
                metric="fuel_per_cycle_l", value=round(float(r.fuel_per_cycle), 3), baseline=_r(med), z=_r(z),
                severity="coach",
                reason=(f"Fuel per load cycle {r.fuel_per_cycle:.2f} L vs your usual {med:.2f} L; "
                        "if the work didn't change, it may be a machine issue worth checking."),
            ))
    return out


def _r(x: Optional[float], nd: int = 3) -> Optional[float]:
    return None if x is None else round(float(x), nd)


# ---------------------------------------------------------------------------------------------
# window level (the brief's rows: each row is a one-hour window)
# ---------------------------------------------------------------------------------------------

def window_flags(rows: pd.DataFrame, history: pd.DataFrame, s: Settings = SETTINGS) -> list[AnomalyFlag]:
    """Flags per telemetry row, with the operator's hourly history (last 20 shifts) as the baseline.
    `history` rows before `rows` on the same machine are used for the meter-gap check."""
    flags: list[AnomalyFlag] = []
    both = pd.concat([history.assign(_target=False), rows.assign(_target=True)], ignore_index=True)
    both = both.sort_values("timestamp", kind="stable")
    gaps = meter_gaps(both)
    prev_ts = both.groupby("machine_id")["timestamp"].shift(1)
    for idx, r in both[both._target].iterrows():
        op, date = r.operator_id, r.shift_date
        tag = f"brief-r{int(r.row)}" if r.source == "brief" else f"{op}-{r.timestamp:%Y%m%d%H}"
        hist = history[history.operator_id == op]
        if len(hist):
            recent = sorted(hist.shift_date.unique())[-s.baseline_shifts:]
            hist = hist[hist.shift_date.isin(recent)]
        idle_ratio = r.idling_time_min / r.window_min
        work = r.window_min - r.idling_time_min
        idle_fuel = r.idling_time_min / 60 * s.idle_burn_l_h

        if r.belt_off_idle_min > s.belt_off_min:
            flags.append(AnomalyFlag(
                flag_id=f"{tag}-{R.BELT_OFF_IDLE}", operator_id=op, shift_date=date, rule=R.BELT_OFF_IDLE,
                metric="belt_off_idle_min", value=float(r.belt_off_idle_min), severity="coach",
                reason=(f"Idle {r.idling_time_min:.0f} of {r.window_min:.0f} min with the belt off and "
                        f"{int(r.load_cycles)} load cycle{'s' if r.load_cycles != 1 else ''}: looks like the seat was "
                        f"left with the engine running; about {idle_fuel:.1f} L at idle."),
            ))
        belt_off_work = r.belt_off_min - r.belt_off_idle_min
        if belt_off_work > s.belt_off_min:
            flags.append(AnomalyFlag(
                flag_id=f"{tag}-{R.BELT_OFF_ENGINE_ON}", operator_id=op, shift_date=date, rule=R.BELT_OFF_ENGINE_ON,
                metric="belt_off_work_min", value=float(belt_off_work), severity="coach",
                reason=f"Belt off during about {belt_off_work:.0f} min of operating in this hour.",
            ))

        hist_ratio = (hist.idling_time_min / hist.window_min) if len(hist) else pd.Series(dtype=float)
        z, med = modified_z(idle_ratio, hist_ratio)
        if idle_ratio > s.idle_ratio_window_max or (z is not None and z > s.z_cutoff and idle_ratio > s.idle_ratio_shift_max):
            usual = f" vs your usual {_pct(med)}" if med is not None else ""
            flags.append(AnomalyFlag(
                flag_id=f"{tag}-{R.IDLE_RATIO_HIGH}", operator_id=op, shift_date=date, rule=R.IDLE_RATIO_HIGH,
                metric="idle_ratio", value=round(idle_ratio, 3), baseline=_r(med), z=_r(z), severity="coach",
                reason=f"Idle {r.idling_time_min:.0f} of {r.window_min:.0f} min ({_pct(idle_ratio)}){usual}.",
            ))

        if pd.notna(r.max_idle_stretch_min) and not r.logged_wait and r.max_idle_stretch_min > s.idle_continuous_min:
            flags.append(AnomalyFlag(
                flag_id=f"{tag}-{R.IDLE_CONTINUOUS}", operator_id=op, shift_date=date, rule=R.IDLE_CONTINUOUS,
                metric="max_idle_stretch_min", value=float(r.max_idle_stretch_min), severity="info",
                reason=f"Idled {r.max_idle_stretch_min:.0f} min in one stretch with no logged truck wait.",
            ))

        if r.load_cycles > 0:
            fpc = r.fuel_used_l / r.load_cycles
            hist_fpc = (hist.fuel_used_l / hist.load_cycles.where(hist.load_cycles > 0)) if len(hist) else pd.Series(dtype=float)
            z, med = modified_z(fpc, hist_fpc)
            if z is not None and z > s.z_cutoff:
                flags.append(AnomalyFlag(
                    flag_id=f"{tag}-{R.FUEL_PER_CYCLE_HIGH}", operator_id=op, shift_date=date, rule=R.FUEL_PER_CYCLE_HIGH,
                    metric="fuel_per_cycle_l", value=round(fpc, 3), baseline=_r(med), z=_r(z), severity="coach",
                    reason=(f"{fpc:.2f} L per load cycle vs your usual {med:.2f} L "
                            f"(about {fpc / med:.0f}x), with {work:.0f} min of work in the hour."),
                ))

        if gaps.get(idx, 0.0) > s.after_hours_unaccounted_h:
            p = prev_ts.get(idx)
            flags.append(_after_hours_flag(op, date, f"{tag}-{R.AFTER_HOURS_RUNNING}", float(gaps[idx]),
                                           None if pd.isna(p) else p, r.timestamp))
    return flags


# ---------------------------------------------------------------------------------------------
# flags from the incident log (the safety engine's unattended-idle events)
# ---------------------------------------------------------------------------------------------

def flags_from_events(events: Iterable[AlertEvent], shift_date: str) -> list[AnomalyFlag]:
    flags = []
    for e in events:
        if e.type == AlertType.unattended_idle:
            flags.append(AnomalyFlag(
                flag_id=f"evt-{e.alert_id}-{R.UNATTENDED_IDLE}", operator_id=e.operator_id, shift_date=shift_date,
                rule=R.UNATTENDED_IDLE, metric="event_t_s", value=float(e.t), severity="review",
                reason="Machine idled with the belt off and hydraulics locked for over 5 min: possibly unattended. "
                       + (e.reasons[0] if e.reasons else ""),
            ))
    return flags
