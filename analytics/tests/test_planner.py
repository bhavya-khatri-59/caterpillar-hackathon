import itertools
import math

from analytics.planner import DayPlanner, apply_overrides
from analytics.settings import SETTINGS
from shared.contracts import EtaStep

FORECAST = [{"hour": h, "ground": "dry" if h < 12 else "mud", "rain_mm_h": 0.0 if h < 12 else 7.0,
             "visibility_m": 2000 if h < 12 else 150, "temp_c": 25, "humidity_pct": 60} for h in range(7, 18)]


def _today(tasks, start="08:00", end="17:00", fuel_pct=80.0):
    return {"date": "2025-05-02", "operator_id": "OP1", "shift_start": "07:00", "plan_start": start, "shift_end": end,
            "fuel_pct": fuel_pct, "zones": {"A": {}}, "forecast": FORECAST, "tasks": tasks}


def _task(tid, ttype, prio, base):
    return {"task_id": tid, "type": ttype, "zone": "A", "quantity_m3": base, "priority": prio, "status": "pending"}


def toy_estimate(task, cond, hour):
    """Minutes = quantity; trenching takes 1.8x in mud. Spread +-10%."""
    p50 = task["quantity_m3"] * (1.8 if task["type"] == "trenching" and cond["ground"] == "mud" else 1.0)
    return {"p10": p50 * 0.9, "p50": p50, "p90": p50 * 1.1, "breakdown": [EtaStep(label="x", minutes=p50)]}


def brute_force(planner, ids):
    best = None
    for perm in itertools.permutations(ids):
        _, total, p90_end = planner.schedule(list(perm))
        key = (p90_end > planner.end, round(total, 1))
        best = key if best is None or key < best else best
    return best


def test_rain_sensitive_task_goes_before_the_rain():
    tasks = [_task("L1", "truck_loading", 1, 150), _task("L2", "truck_loading", 2, 90), _task("T1", "trenching", 3, 60)]
    p = DayPlanner(_today(tasks), toy_estimate, {"trenching": 6.0})
    order, baseline, fits = p.optimise()
    assert baseline == ["L1", "L2", "T1"]
    assert order.index("T1") < 2 and fits
    _, plan = p.plan()
    assert plan.saving_min == 48.0 and "before the 12:00 rain" in plan.reason and "T1" in plan.reason


def test_matches_brute_force_optimum():
    tasks = [_task(f"X{i}", t, i, q) for i, (t, q) in enumerate(
        [("trenching", 50), ("backfill", 70), ("truck_loading", 40), ("trenching", 30), ("truck_loading", 80)], 1)]
    p = DayPlanner(_today(tasks), toy_estimate, {})
    order, _, _ = p.optimise()
    _, total, p90_end = p.schedule(order)
    assert (p90_end > p.end, round(total, 1)) == brute_force(p, [t["task_id"] for t in tasks])
    assert sorted(order) == sorted(t["task_id"] for t in tasks)


def test_p90_must_fit_the_shift():
    tasks = [_task("A", "truck_loading", 1, 200), _task("B", "trenching", 2, 200)]
    p = DayPlanner(_today(tasks, end="15:30"), toy_estimate, {})
    order, _, fits = p.optimise()
    slots, total, p90_end = p.schedule(order)
    assert fits and p90_end <= p.end
    spread = math.sqrt(sum((s.est["p90"] - s.est["p50"]) ** 2 for s in slots))
    assert p90_end == p.start + (total + spread) / 60


def test_reports_when_nothing_fits():
    tasks = [_task("A", "truck_loading", 1, 400), _task("B", "trenching", 2, 300)]
    _, plan = DayPlanner(_today(tasks, end="12:00"), toy_estimate, {}).plan()
    assert plan.reason.startswith("No order finishes by 12:00")


def test_keeps_priority_order_when_saving_is_small():
    tasks = [_task("A", "truck_loading", 1, 60), _task("B", "backfill", 2, 60)]
    p = DayPlanner(_today(tasks), toy_estimate, {})
    order, baseline, _ = p.optimise()
    assert order == baseline == ["A", "B"]
    assert p.plan()[1].saving_min == 0.0


def test_active_task_stays_first():
    tasks = [_task("A", "truck_loading", 1, 120), _task("T", "trenching", 2, 60)]
    tasks[0]["status"] = "active"
    order, _, _ = DayPlanner(_today(tasks), toy_estimate, {}).optimise()
    assert order[0] == "A"


def test_more_than_six_tasks_keeps_priority_order():
    tasks = [_task(f"X{i}", "trenching", 8 - i, 20) for i in range(8)]
    order, baseline, _ = DayPlanner(_today(tasks), toy_estimate, {}).optimise()
    assert order == baseline == [f"X{i}" for i in reversed(range(8))]


def test_fuel_plan_flags_refuel():
    tasks = [_task("A", "truck_loading", 1, 120), _task("B", "backfill", 2, 120), _task("C", "backfill", 3, 120)]
    # available = 20% x 345 - 10% x 345 = 34.5 L; 10 L/h -> 20 L per task
    p = DayPlanner(_today(tasks, fuel_pct=20.0), toy_estimate, {"truck_loading": 10.0, "backfill": 10.0})
    used, refuel = p.fuel(["A", "B", "C"])
    assert used == 60.0 and refuel == "B"
    assert p.fuel(["A"])[1] is None
    assert SETTINGS.tank_l == 345.0


def test_what_if_overrides():
    f = apply_overrides(FORECAST, rain_mm_h=0.0)
    assert all(x["ground"] == "dry" and x["rain_mm_h"] == 0 for x in f)
    f = apply_overrides(FORECAST, rain_mm_h=3.0, visibility_m=100)
    assert all(x["ground"] == "wet" and x["visibility_m"] == 100 for x in f)


def test_real_day_plan(svc):
    tasks, plan = svc.plan()
    assert sorted(plan.order) == sorted(t.task_id for t in tasks)
    assert plan.saving_min > 5 and "rain" in plan.reason
    assert all(t.p10_min <= t.p50_min <= t.p90_min for t in tasks)
