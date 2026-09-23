"""Replaces M2's hand-made fixtures with real output: `python -m analytics.export_fixtures`.

Writes shared/fixtures/{incidents,tasks,task_plan,flags,skill,shift_summary}.json.
"""
from __future__ import annotations

import json

from analytics.service import AnalyticsService
from analytics.settings import ROOT

FIXTURES = ROOT / "shared" / "fixtures"


def _dump(name: str, obj) -> None:
    data = [o.model_dump(mode="json") for o in obj] if isinstance(obj, list) else obj.model_dump(mode="json")
    (FIXTURES / name).write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote shared/fixtures/{name}")


def main() -> None:
    svc = AnalyticsService(db_path=":memory:", seed_fixtures=True)
    op = svc.today["operator_id"]
    tasks, plan = svc.plan(op)
    _dump("incidents.json", svc.store.records())
    _dump("tasks.json", tasks)
    _dump("task_plan.json", plan)
    # The brief's rows 2 and 4 first, then one seeded example of every other rule.
    flags = list(svc.brief_flags)
    seen = {f.rule for f in flags}
    for f in svc.shift_flags:
        if f.rule not in seen:
            flags.append(f)
            seen.add(f.rule)
    flags += [f for f in svc.all_flags(op) if f.rule == "unattended_idle"]
    _dump("flags.json", flags)
    _dump("skill.json", [svc.skill(op), svc.skill("OP1004")])
    _dump("shift_summary.json", svc.summary(op))


if __name__ == "__main__":
    main()
