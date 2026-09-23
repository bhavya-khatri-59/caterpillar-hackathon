"""Every fixture validates against the contract. Must pass before any merge."""
import json
from pathlib import Path

import pytest

from shared import contracts as C
from shared.rule_ids import ALL_RULES

FIX = Path(__file__).parent / "fixtures"
MODELS = {
    "alerts.json": C.AlertEvent, "incidents.json": C.IncidentRecord, "tasks.json": C.Task,
    "task_plan.json": C.TaskPlan, "flags.json": C.AnomalyFlag, "skill.json": C.SkillProfile,
    "shift_summary.json": C.ShiftSummary, "lessons.json": C.Lesson, "tips.json": C.Tip,
    "agent_answers.json": C.AgentAnswer,
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fixture_matches_contract(name):
    path = FIX / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    data = json.loads(path.read_text())
    for item in data if isinstance(data, list) else [data]:
        MODELS[name](**item)


def test_live_frames():
    path = FIX / "live_scenario2.jsonl"
    if not path.exists():
        pytest.skip("live_scenario2.jsonl not written yet")
    for line in path.read_text().splitlines():
        C.LiveFrame(**json.loads(line))


def test_flags_use_known_rules():
    path = FIX / "flags.json"
    if path.exists():
        assert {f["rule"] for f in json.loads(path.read_text())} <= set(ALL_RULES)
