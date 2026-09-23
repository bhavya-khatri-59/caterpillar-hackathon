"""Loader for the brief's telemetry table (data/brief_telemetry.csv), loaded verbatim."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analytics.settings import DATA_DIR
from analytics.telemetry import COLUMNS

BRIEF_PATH = DATA_DIR / "brief_telemetry.csv"
WINDOW_MIN = 60.0          # assumption: each row covers the hour ending at its timestamp
SHIFT = (7, 17)            # EXC001's scheduled shift

_ALIASES = {
    "timestamp": "timestamp", "time": "timestamp",
    "machine id": "machine_id", "machine_id": "machine_id",
    "operator id": "operator_id", "operator_id": "operator_id",
    "engine hours": "engine_hours", "engine_hours": "engine_hours",
    "fuel used (l)": "fuel_used_l", "fuel_used_l": "fuel_used_l", "fuel used": "fuel_used_l",
    "load cycles": "load_cycles", "load_cycles": "load_cycles",
    "idling time (min)": "idling_time_min", "idling_time_min": "idling_time_min", "idle (min)": "idling_time_min",
    "seatbelt status": "seatbelt_status", "seatbelt_status": "seatbelt_status", "seatbelt": "seatbelt_status",
    "safety alert triggered": "safety_alert_triggered", "safety_alert_triggered": "safety_alert_triggered",
    "alert": "safety_alert_triggered",
}


def load_brief(path: Path = BRIEF_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.rename(columns={c: _ALIASES.get(c.strip().lower(), c) for c in raw.columns})
    df = pd.DataFrame()
    df["timestamp"] = pd.to_datetime(raw["timestamp"])
    df["shift_date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    for col in ("machine_id", "operator_id"):
        df[col] = raw[col].astype(str)
    df["engine_hours"] = raw["engine_hours"].astype(float)
    df["fuel_used_l"] = raw["fuel_used_l"].astype(float)
    df["load_cycles"] = raw["load_cycles"].astype(int)
    df["idling_time_min"] = raw["idling_time_min"].astype(float)
    df["seatbelt_status"] = raw["seatbelt_status"].str.strip().str.capitalize()
    df["safety_alert_triggered"] = raw["safety_alert_triggered"].astype(str).str.strip().str.lower().isin(["yes", "true", "1"])
    df["window_min"] = WINDOW_MIN
    off = df["seatbelt_status"].eq("Unfastened")
    # Assumption: the belt status is the window's dominant state.
    df["belt_off_min"] = np.where(off, WINDOW_MIN, 0.0)
    df["belt_off_idle_min"] = np.where(off, df["idling_time_min"], 0.0)
    df["max_idle_stretch_min"] = np.nan     # not in the brief
    df["logged_wait"] = False
    df["lockout_idle_min"] = np.nan         # hydraulic lockout is an assumed signal, not in the brief
    df["task_type"] = None
    df["shift_start_h"], df["shift_end_h"] = SHIFT
    df["source"] = "brief"
    df["row"] = np.arange(1, len(df) + 1)
    return df[COLUMNS]
