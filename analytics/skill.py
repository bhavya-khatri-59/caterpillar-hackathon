"""Skill level per task type, computed from data (never a self-set toggle).

Inputs: shifts on the task type, cycle time vs the fleet median for that type, hours on this machine
type, total hours, soft alerts per hour and anomaly flags in the last 10 shifts. Moving up is offered
after 5 consecutive shifts meeting the next level's criteria. Safety thresholds never depend on level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.settings import SETTINGS, Settings
from shared.contracts import AnomalyFlag, SkillProfile

TYPES = ("trenching", "truck_loading", "backfill")
LEVELS = ("guided", "standard", "expert")

CRITERIA = {
    # next level: (min shifts on type, min machine-type hours, max cycle ratio vs fleet, max flags per shift)
    "standard": (10, 500, 1.10, 0.30),
    "expert": (30, 3000, 0.95, 0.10),
}
MAX_ALERTS_PER_H_EXPERT = 0.05


def _cycle_min(df: pd.DataFrame) -> pd.Series:
    work = df.window_min - df.idling_time_min
    return work / df.load_cycles.where(df.load_cycles > 0)


def fleet_cycle_medians(tel: pd.DataFrame) -> dict[str, float]:
    t = tel[tel.task_type.isin(TYPES)]
    return {tt: float(np.nanmedian(_cycle_min(t[t.task_type == tt]))) for tt in TYPES}


def skill_profile(operator_id: str, tel: pd.DataFrame, flags: list[AnomalyFlag], op_row: dict,
                  fleet: dict[str, float], s: Settings = SETTINGS) -> SkillProfile:
    t = tel[(tel.operator_id == operator_id) & tel.task_type.isin(TYPES)]
    dates = sorted(t.shift_date.unique())
    last10 = set(dates[-10:])
    serious = [f for f in flags if f.operator_id == operator_id and f.severity != "info"]
    flags_by_date: dict[str, int] = {}
    for f in serious:
        flags_by_date[f.shift_date] = flags_by_date.get(f.shift_date, 0) + 1
    anomaly_rate = sum(flags_by_date.get(d, 0) for d in last10) / max(1, len(last10))
    recent = t[t.shift_date.isin(last10)]
    alerts_per_h = float(recent.safety_alert_triggered.sum()) / max(1.0, recent.window_min.sum() / 60)
    n_shifts_total = len(dates)
    total_h = float(op_row.get("total_hours", 0)) + 10 * n_shifts_total
    mt_h = float(op_row.get("machine_type_hours", 0)) + 10 * n_shifts_total

    inputs: dict[str, float] = {
        "total_hours": round(total_h), "machine_type_hours": round(mt_h),
        "anomaly_rate": round(anomaly_rate, 3), "soft_alerts_per_h": round(alerts_per_h, 3),
    }
    levels, ready = {}, False
    for tt in TYPES:
        tt_rows = t[t.task_type == tt]
        tt_dates = sorted(tt_rows.shift_date.unique())
        ratio = float(np.nanmedian(_cycle_min(tt_rows))) / fleet[tt] if len(tt_rows) else float("nan")
        inputs[f"shifts_{tt}"] = float(len(tt_dates))
        inputs[f"cycle_ratio_{tt}"] = round(ratio, 3) if np.isfinite(ratio) else -1.0

        def meets(level: str, n: int, r: float, rate: float) -> bool:
            min_n, min_h, max_r, max_rate = CRITERIA[level]
            ok = n >= min_n and mt_h >= min_h and np.isfinite(r) and r <= max_r and rate <= max_rate
            return ok and (level != "expert" or alerts_per_h <= MAX_ALERTS_PER_H_EXPERT)

        level = "expert" if meets("expert", len(tt_dates), ratio, anomaly_rate) else \
            "standard" if meets("standard", len(tt_dates), ratio, anomaly_rate) else "guided"
        levels[tt] = level

        # promotion: last 5 shifts on this type each meet the next level's per-shift criteria
        if level != "expert" and len(tt_dates) >= s.promotion_streak:
            nxt = LEVELS[LEVELS.index(level) + 1]
            streak = tt_dates[-s.promotion_streak:]
            per_shift_ok = all(
                meets(nxt, len(tt_dates), float(np.nanmedian(_cycle_min(tt_rows[tt_rows.shift_date == d]))) / fleet[tt],
                      flags_by_date.get(d, 0))
                for d in streak
            )
            ready = ready or per_shift_ok
    return SkillProfile(operator_id=operator_id, levels=levels, inputs=inputs, promotion_ready=ready)
