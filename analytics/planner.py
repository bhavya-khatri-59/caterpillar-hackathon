"""Weather-aware task order and fuel plan.

With 6 or fewer pending tasks, every order is tried (at most 720) through the ETA model under the
hourly forecast: each task is estimated with the conditions at the hour it would start. The chosen
order has the lowest total P50 whose P90 still ends within the shift; ties go to the order closest
to the given priorities, and the priority order is kept unless another saves at least 5 min. The P90 of the day assumes independent task errors, so spreads add in
quadrature: P90_total = P50_total + sqrt(sum((P90_i - P50_i)^2)).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Optional

from analytics.conditions import heat_index_c
from analytics.risk import task_risk_factors
from analytics.settings import SETTINGS, Settings
from shared.contracts import Task, TaskPlan

MAX_PERMUTE = 6
RAIN_SENSITIVE = {"trenching", "backfill"}


def _hh(s: str) -> float:
    h, m = s.split(":")
    return int(h) + int(m) / 60


def _fmt(h: float) -> str:
    h = max(0.0, h)
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def forecast_at(forecast: list[dict], hour: float) -> dict:
    """Conditions for the hour containing `hour`; clamps to the first/last forecast entry."""
    by_hour = {int(f["hour"]): f for f in forecast}
    h = min(max(int(hour), min(by_hour)), max(by_hour))
    return by_hour[h]


def apply_overrides(forecast: list[dict], rain_mm_h: Optional[float] = None, visibility_m: Optional[float] = None,
                    ground: Optional[str] = None) -> list[dict]:
    out = []
    for f in forecast:
        f = dict(f)
        if rain_mm_h is not None:
            f["rain_mm_h"] = float(rain_mm_h)
            if ground is None:
                f["ground"] = "dry" if rain_mm_h < 0.5 else "wet" if rain_mm_h <= 5 else "mud"
        if ground is not None:
            f["ground"] = ground
        if visibility_m is not None:
            f["visibility_m"] = float(visibility_m)
        out.append(f)
    return out


@dataclass
class Slot:
    task_id: str
    start_h: float
    est: dict
    cond: dict


Estimator = Callable[[dict, dict, int], dict]   # (task, conditions, start_hour) -> predict_one() output


class DayPlanner:
    def __init__(self, today: dict, estimate: Estimator, burn_l_h: dict, s: Settings = SETTINGS):
        self.today, self.estimate, self.burn, self.s = today, estimate, burn_l_h, s
        self.tasks = {t["task_id"]: t for t in today["tasks"]}
        self.start = _hh(today.get("plan_start", today["shift_start"]))
        self.end = _hh(today["shift_end"])
        self._cache: dict[tuple[str, int], dict] = {}

    def _est(self, tid: str, hour: float) -> tuple[dict, dict]:
        cond = forecast_at(self.today["forecast"], hour)
        key = (tid, int(cond["hour"]))
        if key not in self._cache:
            self._cache[key] = self.estimate(self.tasks[tid], cond, int(cond["hour"]))
        return self._cache[key], cond

    def schedule(self, order: list[str]) -> tuple[list[Slot], float, float]:
        """Returns slots, total P50 minutes, and P90 end hour."""
        t, slots, total, var = self.start, [], 0.0, 0.0
        for tid in order:
            est, cond = self._est(tid, t)
            slots.append(Slot(tid, t, est, cond))
            total += est["p50"]
            var += (est["p90"] - est["p50"]) ** 2
            t += est["p50"] / 60
        return slots, total, self.start + (total + math.sqrt(var)) / 60

    def _priority_cost(self, order: list[str]) -> int:
        ranked = sorted(order, key=lambda i: self.tasks[i]["priority"])
        return sum(abs(order.index(i) - ranked.index(i)) for i in order)

    def optimise(self) -> tuple[list[str], list[str], bool]:
        """Returns (best order, priority order, whether the best order's P90 fits the shift)."""
        active = [i for i, t in self.tasks.items() if t.get("status") == "active"]
        pending = sorted((i for i, t in self.tasks.items() if t.get("status", "pending") == "pending"),
                         key=lambda i: self.tasks[i]["priority"])
        baseline = active + pending
        if len(pending) > MAX_PERMUTE:                     # too many to try: keep priority order
            _, _, p90_end = self.schedule(baseline)
            return baseline, baseline, p90_end <= self.end
        best, best_key, best_fits = baseline, None, False
        for perm in itertools.permutations(pending):
            order = active + list(perm)
            _, total, p90_end = self.schedule(order)
            fits = p90_end <= self.end + 1e-9
            key = (not fits, round(total, 1), self._priority_cost(order))
            if best_key is None or key < best_key:
                best, best_key, best_fits = order, key, fits
        _, base_total, base_p90 = self.schedule(baseline)
        base_fits = base_p90 <= self.end + 1e-9
        if base_fits == best_fits and base_total - best_key[1] < self.s.min_reorder_saving_min:
            return baseline, baseline, base_fits                 # not worth overriding the priorities
        return best, baseline, best_fits

    # -- outputs -------------------------------------------------------------------------------
    def tasks_for(self, order: list[str]) -> list[Task]:
        slots, _, _ = self.schedule(order)
        by_id = {sl.task_id: sl for sl in slots}
        out = []
        for tid, t in self.tasks.items():
            sl = by_id.get(tid)
            if sl is None:                                  # done tasks: estimate at plan start
                est, cond = self._est(tid, self.start)
            else:
                est, cond = sl.est, sl.cond
            zone = self.today.get("zones", {}).get(t["zone"], {})
            out.append(Task(
                task_id=tid, type=t["type"], zone=t["zone"], quantity_m3=float(t["quantity_m3"]),
                priority=int(t["priority"]), status=t.get("status", "pending"),
                p10_min=est["p10"], p50_min=est["p50"], p90_min=est["p90"], eta_breakdown=est["breakdown"],
                risk_factors=task_risk_factors(zone, cond, heat_index_c(cond["temp_c"], cond["humidity_pct"])),
            ))
        return out

    def fuel(self, order: list[str]) -> tuple[float, Optional[str]]:
        slots, _, _ = self.schedule(order)
        available = self.today.get("fuel_pct", 100.0) / 100 * self.s.tank_l - self.s.fuel_reserve_pct / 100 * self.s.tank_l
        used, refuel = 0.0, None
        for sl in slots:
            used += sl.est["p50"] / 60 * self.burn.get(self.tasks[sl.task_id]["type"], 6.0)
            if refuel is None and used > available:
                refuel = sl.task_id
        return round(used, 1), refuel

    def plan(self) -> tuple[list[Task], TaskPlan]:
        best, baseline, fits = self.optimise()
        slots, total, p90_end = self.schedule(best)
        _, base_total, _ = self.schedule(baseline)
        saving = round(base_total - total, 1)
        fuel_l, refuel = self.fuel(best)
        plan = TaskPlan(order=best, total_p50_min=round(total, 1), saving_min=max(0.0, saving),
                        reason=self._reason(best, baseline, slots, saving, fits, p90_end),
                        fuel_needed_l=fuel_l, refuel_before_task=refuel)
        return self.tasks_for(best), plan

    def _reason(self, best, baseline, slots, saving, fits, p90_end) -> str:
        if not fits:
            return (f"No order finishes by {_fmt(self.end)} at P90 (this one ends around {_fmt(p90_end)}); "
                    "talk to the supervisor about moving a task.")
        if best == baseline:
            return "Keep the priority order: no other order saves more than a few minutes under today's forecast."
        rain = next((f["hour"] for f in sorted(self.today["forecast"], key=lambda f: f["hour"])
                     if f["hour"] >= self.start and f["rain_mm_h"] >= 1.0), None)
        moved = [i for i in best if best.index(i) < baseline.index(i)]
        sensitive = [i for i in moved if self.tasks[i]["type"] in RAIN_SENSITIVE]
        names = ", ".join(f"{i} ({self.tasks[i]['type'].replace('_', ' ')})" for i in (sensitive or moved)[:2])
        if rain is not None and sensitive:
            return f"Do {names} before the {rain:02d}:00 rain: saves about {saving:.0f} min at P50."
        return f"Moving {names} earlier saves about {saving:.0f} min at P50 under today's forecast."
