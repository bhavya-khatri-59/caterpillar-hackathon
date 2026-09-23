"""Wires data, models and the incident store together. One lazily built instance per process."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analytics import anomaly, skill
from analytics.brief import load_brief
from analytics.eta import EtaModel
from analytics.generator import EXPECTED_RULES, generate
from analytics.planner import DayPlanner, apply_overrides, forecast_at
from analytics.conditions import make_conditions
from analytics.settings import DATA_DIR, ROOT, SETTINGS, Settings
from analytics.store import IncidentStore
from analytics.summary import build_summary
from shared.contracts import AlertEvent, AnomalyFlag, Conditions, ShiftSummary, SkillProfile, Task, TaskPlan

TODAY_PATH = DATA_DIR / "today.json"
ALERTS_FIXTURE = ROOT / "shared" / "fixtures" / "alerts.json"


class AnalyticsService:
    def __init__(self, s: Settings = SETTINGS, db_path: Optional[str] = None, today_path: Path = TODAY_PATH,
                 seed_fixtures: Optional[bool] = None):
        self.s = s
        gen = generate(s)
        self.generated = gen["telemetry"]
        self.task_history = gen["tasks"]
        self.operators = gen["operators"].set_index("operator_id")
        self.planted = gen["planted"]
        self.brief = load_brief()
        self.telemetry = pd.concat([self.generated, self.brief], ignore_index=True)

        self.shift_feats = anomaly.shift_features(self.generated)
        self.shift_flags = anomaly.shift_flags(self.shift_feats, s)
        self.brief_flags = anomaly.window_flags(self.brief, self.generated[self.generated.machine_id.isin(self.brief.machine_id)], s)

        self.eta = EtaModel(s)
        self.eta_metrics = self.eta.fit_evaluate(self.task_history)
        self.fleet_cycle = skill.fleet_cycle_medians(self.generated)
        self.burn = self.generated.groupby("task_type").fuel_used_l.mean().round(2).to_dict()

        self.today = json.loads(Path(today_path).read_text())
        self.store = IncidentStore(db_path or s.db_path)
        if seed_fixtures is None:
            seed_fixtures = os.environ.get("ANALYTICS_SEED_FIXTURES", "1") != "0"
        if seed_fixtures and len(self.store) == 0 and ALERTS_FIXTURE.exists():
            for e in json.loads(ALERTS_FIXTURE.read_text()):
                self.store.emit(AlertEvent(**e))
        self._plans: dict[str, tuple[list[Task], TaskPlan]] = {}
        self._lock = threading.Lock()

    # -- helpers -------------------------------------------------------------------------------
    def experience_h(self, operator_id: str) -> float:
        if operator_id in self.operators.index:
            return float(self.operators.loc[operator_id, "total_hours"]) + 10 * self.s.n_shifts
        return 0.0

    def current_conditions(self, hour: Optional[float] = None) -> Conditions:
        f = forecast_at(self.today["forecast"], hour if hour is not None else float(self.today["plan_start"][:2]))
        return make_conditions(f["ground"], f["rain_mm_h"], f["visibility_m"], f["temp_c"], f["humidity_pct"])

    def all_flags(self, operator_id: str) -> list[AnomalyFlag]:
        flags = [f for f in self.shift_flags + self.brief_flags if f.operator_id == operator_id]
        events = [r.event for r in self.store.records(operator_id)]
        flags += anomaly.flags_from_events(events, self.today["date"])
        # dedupe (an event can be logged more than once as it updates)
        seen, out = set(), []
        for f in flags:
            if f.flag_id not in seen:
                seen.add(f.flag_id)
                out.append(f)
        return sorted(out, key=lambda f: (f.shift_date, f.flag_id), reverse=True)

    # -- tasks -----------------------------------------------------------------------------
    def planner(self, operator_id: Optional[str] = None, rain_mm_h: Optional[float] = None,
                visibility_m: Optional[float] = None, ground: Optional[str] = None,
                trucks_available: Optional[float] = None, fuel_pct: Optional[float] = None) -> DayPlanner:
        op = operator_id or self.today["operator_id"]
        today = dict(self.today)
        today["forecast"] = apply_overrides(self.today["forecast"], rain_mm_h, visibility_m, ground)
        if fuel_pct is not None:
            today["fuel_pct"] = fuel_pct
        trucks = self.today.get("trucks_available", 1.0) if trucks_available is None else trucks_available
        zones, exp = today.get("zones", {}), self.experience_h(op)

        def estimate(task: dict, cond: dict, hour: int) -> dict:
            slope = float(zones.get(task["zone"], {}).get("slope_deg", 0.0))
            return self.eta.predict_one(task["type"], float(task["quantity_m3"]), cond, hour, op, exp, slope, trucks)

        return DayPlanner(today, estimate, self.burn, self.s)

    def plan(self, operator_id: Optional[str] = None) -> tuple[list[Task], TaskPlan]:
        op = operator_id or self.today["operator_id"]
        with self._lock:
            if op not in self._plans:
                self._plans[op] = self.planner(op).plan()
            return self._plans[op]

    # -- operators -----------------------------------------------------------------------------
    def skill(self, operator_id: str) -> SkillProfile:
        row = self.operators.loc[operator_id].to_dict() if operator_id in self.operators.index else {}
        return skill.skill_profile(operator_id, self.generated, self.shift_flags, row, self.fleet_cycle, self.s)

    def summary(self, operator_id: Optional[str] = None, date: Optional[str] = None) -> ShiftSummary:
        op = operator_id or self.today["operator_id"]
        date = date or self.today["date"]
        tel = self.telemetry
        dates = sorted(tel[tel.operator_id == op].shift_date.unique())
        if date not in dates and dates:
            date = max([d for d in dates if d <= date] or dates)   # latest shift with telemetry
        tasks = self.plan(op)[0] if date == self.today["date"] and op == self.today["operator_id"] else []
        hist = self.shift_feats[self.shift_feats.operator_id == op]
        usual = float(hist.idle_ratio.tail(self.s.baseline_shifts).median() * 100) if len(hist) else None
        level_counts = list(self.skill(op).levels.values())
        level = max(set(level_counts), key=level_counts.count) if level_counts else "standard"
        return build_summary(op, date, tel, self.store.records(op), self.all_flags(op), tasks, level, usual)

    # -- evidence ------------------------------------------------------------------------------
    def evidence(self) -> dict:
        if not hasattr(self, "_anomaly_metrics"):
            self._anomaly_metrics = anomaly_metrics(self.shift_feats, self.planted, self.s)
        return {
            "task_time": self.eta_metrics,
            "anomaly": self._anomaly_metrics,
            "brief_rows": brief_check(self.brief_flags),
            "incident_log": self.store.verify(),
            "task_order": self._plan_evidence(),
            "data": {
                "operators": int(len(self.operators)), "shifts_per_operator": self.s.n_shifts,
                "telemetry_rows": int(len(self.generated)), "task_history_rows": int(len(self.task_history)),
                "planted_shifts": int(len(self.planted)), "seed": self.s.seed,
                "note": "Seeded synthetic data in the brief's columns; the models recover effects we planted. "
                        "The point is the pipeline, which retrains on real history.",
            },
        }

    def _plan_evidence(self) -> dict:
        tasks, plan = self.plan()
        return {"order": plan.order, "saving_min": plan.saving_min, "reason": plan.reason,
                "total_p50_min": plan.total_p50_min, "fuel_needed_l": plan.fuel_needed_l,
                "refuel_before_task": plan.refuel_before_task}


def anomaly_metrics(feats: pd.DataFrame, planted: pd.DataFrame, s: Settings = SETTINGS) -> dict:
    """Recall on planted shifts and precision of review/coach flags, for rules alone vs rules + z-score."""
    truth = {(r.operator_id, r.shift_date): r.pattern for r in planted.itertuples()}

    def score(flags: list[AnomalyFlag]) -> dict:
        by_shift: dict[tuple, set] = {}
        serious: set = set()
        for f in flags:
            by_shift.setdefault((f.operator_id, f.shift_date), set()).add(f.rule)
            if f.severity != "info":
                serious.add((f.operator_id, f.shift_date))
        per = {}
        for pat, rules in EXPECTED_RULES.items():
            keys = [k for k, p in truth.items() if p == pat]
            per[pat] = {"planted": len(keys), "detected": sum(bool(by_shift.get(k, set()) & rules) for k in keys)}
        hit = sum(v["detected"] for v in per.values())
        n = sum(v["planted"] for v in per.values())
        tp = sum(1 for k in serious if k in truth)
        return {"recall": round(hit / n, 3) if n else None, "precision": round(tp / len(serious), 3) if serious else None,
                "flagged_shifts": len(serious), "per_pattern": per}

    both = score(anomaly.shift_flags(feats, s))
    rules_only = score(anomaly.shift_flags(feats, replace(s, z_cutoff=float("inf"))))
    return {"rules_plus_z": both, "rules_only": rules_only, "z_cutoff": s.z_cutoff,
            "unit": "operator-shift; precision counts review/coach flags only"}


def brief_check(flags: list[AnomalyFlag]) -> dict:
    rows = {}
    for f in flags:
        r = int(f.flag_id.split("-")[1][1:])
        rows.setdefault(r, []).append(f.rule)
    flagged = sorted(rows)
    return {"flagged_rows": flagged, "expected_rows": [2, 4], "pass": flagged == [2, 4],
            "rules_by_row": {str(k): v for k, v in sorted(rows.items())}}


_service: Optional[AnalyticsService] = None
_service_lock = threading.Lock()


def get_service() -> AnalyticsService:
    global _service
    with _service_lock:
        if _service is None:
            _service = AnalyticsService()
        return _service


def set_service(svc: Optional[AnalyticsService]) -> None:
    """For tests and for api/main.py to inject a configured instance."""
    global _service
    with _service_lock:
        _service = svc
