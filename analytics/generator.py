"""Seeded Layer A generator: 30 operators x 60 shifts of hourly telemetry in the brief's columns,
plus a task history for the task-time model. Behaviour patterns are planted on purpose so the
anomaly engine can be scored against a known truth.

Run `python -m analytics.generator` to write CSVs to data/generated/ for inspection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from analytics.conditions import heat_index_c
from analytics.settings import DATA_DIR, SETTINGS, Settings
from analytics.telemetry import COLUMNS
from shared import rule_ids as R

TASK_TYPES = ("trenching", "truck_loading", "backfill")
QTY_RANGE = {"trenching": (60, 200), "truck_loading": (100, 320), "backfill": (80, 260)}

# Planted behaviour profiles: operator -> pattern. OP1001 (the brief's operator) is normal.
PLANTED = {
    "OP1004": "belt_off_idle", "OP1011": "belt_off_idle", "OP1019": "belt_off_idle",
    "OP1007": "after_hours", "OP1023": "after_hours",
    "OP1014": "cycle_drift", "OP1027": "cycle_drift",
    "OP1009": "fuel_heavy",
}
# Which rule IDs count as detecting each planted pattern.
EXPECTED_RULES = {
    "belt_off_idle": {R.BELT_OFF_IDLE, R.BELT_OFF_ENGINE_ON, R.IDLE_RATIO_HIGH, R.IDLE_CONTINUOUS},
    "after_hours": {R.AFTER_HOURS_RUNNING},
    "cycle_drift": {R.CYCLE_TIME_DRIFT},
    "fuel_heavy": {R.FUEL_PER_CYCLE_HIGH},
}


@dataclass
class Operator:
    operator_id: str
    machine_id: str
    total_hours: float
    machine_type_hours: float
    speed: float          # cycle-time multiplier, lower = faster
    task_factor: float    # hidden operator effect on task time
    shift_start: int
    pattern: str | None


def _operators(rng: np.random.Generator, n: int) -> list[Operator]:
    ops = []
    for i in range(n):
        oid = f"OP{1001 + i}"
        total = 6000.0 if oid == "OP1001" else float(rng.uniform(300, 12000))
        exp = min(1.0, total / 8000)
        ops.append(Operator(
            operator_id=oid, machine_id=f"EXC{1 + i:03d}", total_hours=round(total),
            machine_type_hours=round(total * float(rng.uniform(0.3, 1.0))),
            speed=float(1.15 - 0.25 * exp + rng.normal(0, 0.03)),
            task_factor=float((1.25 - 0.3 * exp) * np.exp(rng.normal(0, 0.08))),
            shift_start=7 if oid == "OP1001" or rng.random() < 0.8 else 8,
            pattern=PLANTED.get(oid),
        ))
    return ops


def _shift_dates(n: int, end: str = "2025-04-30") -> list[str]:
    days = pd.bdate_range(end=end, periods=n, freq="C", weekmask="Mon Tue Wed Thu Fri Sat")
    return [d.strftime("%Y-%m-%d") for d in days]


def _weather(rng: np.random.Generator) -> dict:
    if rng.random() < 0.25:
        rain = float(rng.uniform(0.5, 9.0))
        ground = "mud" if rain > 5 else "wet"
        vis = float(rng.uniform(100, 600))
    else:
        rain, ground, vis = 0.0, "dry", float(rng.uniform(1000, 3000))
    temp = float(rng.uniform(20, 37) - (4 if rain else 0))
    hum = float(rng.uniform(40, 70) + (25 if rain else 0))
    return {"rain_mm_h": rain, "ground": ground, "visibility_m": vis, "temp_c": temp, "humidity_pct": min(hum, 98.0)}


def production_rate_m3_h(task_type: str, s: Settings = SETTINGS) -> float:
    """Cat Performance Handbook production formula: Q = V x FF x E x 3600 / t_cycle."""
    return s.bucket_m3 * s.fill_factor[task_type] * s.job_efficiency * 3600.0 / s.cycle_s[task_type]


def prior_minutes(task_type: str, quantity_m3: float, s: Settings = SETTINGS) -> float:
    return 60.0 * quantity_m3 / production_rate_m3_h(task_type, s)


GROUND_MULT = {
    "trenching": {"dry": 1.0, "wet": 1.12, "mud": 1.35},
    "truck_loading": {"dry": 1.0, "wet": 1.08, "mud": 1.20},
    "backfill": {"dry": 1.0, "wet": 1.10, "mud": 1.25},
}


def true_condition_multiplier(task_type: str, ground: str, rain: float, vis: float, slope: float,
                              trucks: float, hour: int, heat_index: float) -> float:
    """The planted truth the model has to recover. Never imported by the model code."""
    m = GROUND_MULT[task_type][ground] * (1 + 0.01 * rain)
    m *= 1.12 if vis < 200 else 1.05 if vis < 500 else 1.0
    m *= 1 + 0.012 * slope
    if task_type == "truck_loading":
        m *= 1 + 0.6 * (1 - trucks)
    m *= 1.04 if hour >= 15 else 1.0
    m *= 1.05 if heat_index > 38 else 1.0
    return m


def generate(s: Settings = SETTINGS) -> dict[str, pd.DataFrame]:
    """Returns {'telemetry', 'tasks', 'operators', 'planted'} DataFrames. Deterministic in s.seed."""
    rng = np.random.default_rng(s.seed)
    ops = _operators(rng, s.n_operators)
    dates = _shift_dates(s.n_shifts)
    tel_rows, task_rows, planted = [], [], []

    for op in ops:
        # OP1001's meter ends at 1244.0 so the brief's first row (1245.0) follows on.
        meter = 1244.0 - s.n_shifts * 10 if op.operator_id == "OP1001" else float(rng.uniform(500, 5000))
        pending_after_hours = 0.0
        for si, date in enumerate(dates):
            start, end = op.shift_start, op.shift_start + 10
            w = _weather(rng)
            hi = heat_index_c(w["temp_c"], w["humidity_pct"])
            meter += pending_after_hours
            if pending_after_hours:
                planted.append((op.operator_id, date, "after_hours"))
            pending_after_hours = 0.0

            # --- tasks for the shift (task history) ---
            hour_task: dict[int, str] = {}
            t_cursor = start + 0.25
            while t_cursor < end - 1.0:
                ttype = TASK_TYPES[int(rng.integers(0, 3))]
                qty = float(np.round(rng.uniform(*QTY_RANGE[ttype]), 0))
                slope = float(np.round(rng.uniform(0, 14), 1))
                trucks = float(np.round(rng.uniform(0.4, 1.0), 2)) if ttype == "truck_loading" else 1.0
                hour = int(t_cursor)
                prior = prior_minutes(ttype, qty, s)
                mult = true_condition_multiplier(ttype, w["ground"], w["rain_mm_h"], w["visibility_m"], slope, trucks, hour, hi)
                sigma = 0.10 + 0.02 * w["rain_mm_h"] + (0.05 if w["ground"] == "mud" else 0.0)
                actual = prior * mult * op.task_factor * float(np.exp(rng.normal(0, sigma)))
                task_rows.append({
                    "operator_id": op.operator_id, "shift_date": date, "shift_index": si, "task_type": ttype,
                    "quantity_m3": qty, "start_hour": hour, "ground": w["ground"], "rain_mm_h": w["rain_mm_h"],
                    "visibility_m": w["visibility_m"], "temp_c": w["temp_c"], "humidity_pct": w["humidity_pct"],
                    "heat_index_c": hi, "slope_deg": slope, "trucks_available": trucks,
                    "experience_h": op.total_hours + si * 10, "prior_min": prior, "actual_min": actual,
                })
                for h in range(hour, min(end, int(t_cursor + actual / 60) + 1)):
                    hour_task.setdefault(h, ttype)
                t_cursor += actual / 60 + 0.1

            # --- planted behaviour for this shift ---
            belt_hours: set[int] = set()
            if op.pattern == "belt_off_idle" and rng.random() < 0.35:
                belt_hours = set(rng.choice(np.arange(start + 1, end), size=int(rng.integers(2, 4)), replace=False).tolist())
                planted.append((op.operator_id, date, "belt_off_idle"))
            fuel_heavy = op.pattern == "fuel_heavy" and rng.random() < 0.12
            if fuel_heavy:
                planted.append((op.operator_id, date, "fuel_heavy"))
            drift = 1.0
            if op.pattern == "cycle_drift" and si >= s.n_shifts - 12:
                drift = 1.0 + 0.06 * (si - (s.n_shifts - 13))   # ramps to about +70%
                if drift - 0.06 >= 1.35:                        # label: 3-shift median drift over +35%
                    planted.append((op.operator_id, date, "cycle_drift"))
            if op.pattern == "after_hours" and rng.random() < 0.3 and si < s.n_shifts - 1:
                pending_after_hours = float(np.round(rng.uniform(0.8, 2.5), 1))

            burn = float(rng.normal(8.0, 0.4)) * (1.8 if fuel_heavy else 1.0) * (1 + 0.01 * w["rain_mm_h"])
            for h in range(start, end):
                ttype = hour_task.get(h, "backfill")
                if h in belt_hours:
                    idle = float(rng.uniform(40, 60))
                    stretch, lockout, belt_off, belt_off_idle = idle, idle, 60.0, idle
                    cycles = int(rng.integers(0, 4))
                    alert = rng.random() < 0.6
                    logged_wait = False
                else:
                    idle = float(np.clip(rng.normal(17 if ttype != "truck_loading" else 20, 5), 3, 38))
                    logged_wait = ttype == "truck_loading" and rng.random() < 0.08
                    stretch = min(idle, float(rng.uniform(11, 20))) if logged_wait else min(idle, float(rng.uniform(2, 9)))
                    lockout = float(np.round(stretch if logged_wait else rng.uniform(0, 0.4) * idle, 1))
                    belt_off = 1.0 if rng.random() < 0.02 else 0.0
                    belt_off_idle = belt_off
                    work = 60 - idle
                    cycles = int(rng.poisson(work * 0.3 / (op.speed * drift)))
                    alert = rng.random() < 0.02
                work = 60 - idle
                fuel = idle / 60 * float(rng.uniform(1.8, 2.0)) + work / 60 * burn
                meter += 1.0
                tel_rows.append({
                    "timestamp": pd.Timestamp(f"{date} {h + 1:02d}:00"), "shift_date": date,
                    "machine_id": op.machine_id, "operator_id": op.operator_id,
                    "engine_hours": round(meter, 1), "fuel_used_l": round(fuel, 1), "load_cycles": cycles,
                    "idling_time_min": round(idle, 0),
                    "seatbelt_status": "Unfastened" if belt_off >= 30 else "Fastened",
                    "safety_alert_triggered": bool(alert), "window_min": 60.0,
                    "belt_off_min": belt_off, "belt_off_idle_min": round(belt_off_idle, 0),
                    "max_idle_stretch_min": round(stretch, 1), "logged_wait": bool(logged_wait),
                    "lockout_idle_min": lockout, "task_type": ttype,
                    "shift_start_h": start, "shift_end_h": end, "source": "generated", "row": 0,
                })

    tel = pd.DataFrame(tel_rows)[COLUMNS]
    tasks = pd.DataFrame(task_rows)
    operators = pd.DataFrame([op.__dict__ for op in ops])
    planted_df = pd.DataFrame(planted, columns=["operator_id", "shift_date", "pattern"]).drop_duplicates()
    return {"telemetry": tel, "tasks": tasks, "operators": operators, "planted": planted_df}


if __name__ == "__main__":
    out = DATA_DIR / "generated"
    out.mkdir(parents=True, exist_ok=True)
    for name, df in generate().items():
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"wrote {out / name}.csv ({len(df)} rows)")
