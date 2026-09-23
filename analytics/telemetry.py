"""Hourly telemetry schema shared by the brief loader and the generator.

One row = one engine-on window ending at `timestamp`. Columns beyond the brief's are
NaN when the source doesn't have them (the brief has no idle-stretch or lockout data).
"""
from __future__ import annotations

COLUMNS = [
    "timestamp", "shift_date", "machine_id", "operator_id",
    # the brief's columns (ISO 15143-3 style fields)
    "engine_hours", "fuel_used_l", "load_cycles", "idling_time_min", "seatbelt_status", "safety_alert_triggered",
    # derived / extended columns
    "window_min", "belt_off_min", "belt_off_idle_min", "max_idle_stretch_min", "logged_wait",
    "lockout_idle_min", "task_type", "shift_start_h", "shift_end_h", "source", "row",
]
