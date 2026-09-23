"""Shift summary, built from the telemetry, the incident log and today's tasks."""
from __future__ import annotations

from statistics import median
from typing import Optional

import pandas as pd

from shared import rule_ids as R
from shared.contracts import AnomalyFlag, IncidentRecord, ShiftSummary, Task

TIPS = {
    R.BELT_OFF_IDLE: ("Lock the hydraulics and shut the engine down if you leave the seat; auto idle shutdown at 5 min does it for you.",
                      "Leaving the seat: engine off. Auto idle shutdown 5 min."),
    R.BELT_OFF_ENGINE_ON: ("Buckle up before you release the lockout; an unbelted operator can be thrown from the cab in a rollover.",
                           "Belt before lockout release."),
    R.IDLE_RATIO_HIGH: ("When waiting on trucks, drop to low idle and lock the hydraulics; it saves about 1 L/h.",
                        "Waiting: low idle, about 1 L/h saved."),
    R.IDLE_CONTINUOUS: ("Long idle stretches add engine hours without work; set auto idle shutdown to 5-10 min.",
                        "Idle shutdown 5-10 min."),
    R.FUEL_PER_CYCLE_HIGH: ("Fuel per cycle was above your usual; shorter swings and a full bucket each pass bring it down. If nothing changed, report a possible machine issue.",
                            "Fuel per cycle high: check swing and fill, or report the machine."),
    R.CYCLE_TIME_DRIFT: ("Your cycles have been slower lately; position the truck within one swing and dig from the same bench height.",
                         "Cycles slower than your baseline: truck position, bench height."),
    R.AFTER_HOURS_RUNNING: ("The engine meter ran outside the shift; please note any extra work in the log so it isn't flagged.",
                            "Log any off-shift running."),
}
TIPS[R.UNATTENDED_IDLE] = TIPS[R.BELT_OFF_IDLE]
# Safety first, then cost, then efficiency.
RULE_PRIORITY = [R.BELT_OFF_ENGINE_ON, R.BELT_OFF_IDLE, R.UNATTENDED_IDLE, R.AFTER_HOURS_RUNNING,
                 R.FUEL_PER_CYCLE_HIGH, R.IDLE_RATIO_HIGH, R.CYCLE_TIME_DRIFT, R.IDLE_CONTINUOUS]
DEFAULT_TIP = ("Keep doing a walk-around at every restart; it's the easiest hazard check there is.", "Walk-around at every restart.")


def build_summary(operator_id: str, date: str, tel: pd.DataFrame, incidents: list[IncidentRecord],
                  flags: list[AnomalyFlag], tasks: list[Task], level: str = "standard",
                  usual_idle_pct: Optional[float] = None) -> ShiftSummary:
    rows = tel[(tel.operator_id == operator_id) & (tel.shift_date == date)]
    window = float(rows.window_min.sum())
    idle = float(rows.idling_time_min.sum())
    belt_off = float(rows.belt_off_min.sum())
    compliance = 100.0 * (1 - belt_off / window) if window else 100.0
    idle_pct = 100.0 * idle / window if window else 0.0
    fuel = float(rows.fuel_used_l.sum())

    seen, serious, warnings = set(), 0, []
    for rec in incidents:
        e = rec.event
        if e.operator_id != operator_id:
            continue
        if e.warning_time_s is not None:
            warnings.append(e.warning_time_s)
        if e.alert_id in seen:
            continue
        if e.tier.value == "hard" or rec.outcome in ("near_miss", "incident"):
            seen.add(e.alert_id)
            serious += 1

    if window and compliance >= 99.5:
        went_well = "Belt fastened for the whole shift."
    elif warnings and min(warnings) >= 3:
        went_well = f"You had at least {min(warnings):.0f} s of warning on every alert."
    elif window and usual_idle_pct is not None and idle_pct < usual_idle_pct:
        went_well = f"Idle was {idle_pct:.0f}%, below your usual {usual_idle_pct:.0f}%."
    elif tasks and all(t.status == "done" for t in tasks):
        went_well = "Every task finished."
    else:
        went_well = "You completed the pre-shift check and kept the log up to date."

    day_flags = sorted((f for f in flags if f.operator_id == operator_id and f.shift_date == date and f.severity != "info"),
                       key=lambda f: RULE_PRIORITY.index(f.rule) if f.rule in RULE_PRIORITY else 99)
    full, short = TIPS.get(day_flags[0].rule, DEFAULT_TIP) if day_flags else DEFAULT_TIP
    tip = short if level == "expert" else full

    return ShiftSummary(
        operator_id=operator_id, date=date, near_misses=serious,
        median_warning_s=round(median(warnings), 1) if warnings else None,
        belt_compliance_pct=round(compliance, 1), idle_pct=round(idle_pct, 1), fuel_l=round(fuel, 1),
        tasks=tasks, went_well=went_well, tip=tip, handover_note=None,
    )
