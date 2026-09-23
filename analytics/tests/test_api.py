import pytest
from fastapi.testclient import TestClient

from analytics.app import app
from analytics.service import set_service
from shared.contracts import AnomalyFlag, IncidentRecord, ShiftSummary, SkillProfile, Task, TaskPlan


@pytest.fixture(scope="module")
def client(svc):
    set_service(svc)
    with TestClient(app) as c:
        yield c
    set_service(None)


def test_tasks_and_plan(client):
    tasks = [Task(**t) for t in client.get("/api/tasks").json()]
    plan = TaskPlan(**client.get("/api/tasks/plan").json())
    assert len(tasks) == 5 and sorted(plan.order) == sorted(t.task_id for t in tasks)


def test_what_if_rain_widens_ranges(client):
    dry = client.post("/api/eta/whatif", json={"rain_mm_h": 0}).json()
    wet = client.post("/api/eta/whatif", json={"rain_mm_h": 8, "visibility_m": 100}).json()
    width = lambda r: sum(t["p90_min"] - t["p10_min"] for t in r["tasks"])
    assert wet["plan"]["total_p50_min"] > dry["plan"]["total_p50_min"]
    assert width(wet) > width(dry)
    assert client.post("/api/eta/whatif", json={"trucks_available": 5}).status_code == 422


def test_incidents_manual_report_and_verify(client):
    before = len(client.get("/api/incidents").json())
    r = client.post("/api/incidents", json={"operator_id": "OP1001", "category": "near_miss", "method": "voice",
                                           "note": "truck reversed close"})
    assert r.status_code == 201
    rec = IncidentRecord(**r.json())
    assert rec.outcome == "near_miss" and rec.event.source == "operator"
    assert len(client.get("/api/incidents").json()) == before + 1
    v = client.get("/api/incidents/verify").json()
    assert v["ok"] and v["count"] == before + 1
    hard = client.get("/api/incidents", params={"tier": "hard"}).json()
    assert hard and all(x["event"]["tier"] == "hard" for x in hard)


def test_event_append_over_http(client):
    ev = client.get("/api/incidents").json()[0]["event"]
    r = client.post("/api/incidents/events", json={"event": ev, "outcome": "cleared"})
    assert r.status_code == 201 and r.json()["outcome"] == "cleared"


def test_anomalies_skill_summary(client):
    flags = [AnomalyFlag(**f) for f in client.get("/api/operators/OP1001/anomalies").json()]
    assert {"brief-r2-belt_off_idle", "brief-r4-belt_off_idle"} <= {f.flag_id for f in flags}
    assert client.get("/api/operators/OP1001/anomalies", params={"severity": "review"}).json()
    SkillProfile(**client.get("/api/operators/OP1001/skill").json())
    s = ShiftSummary(**client.get("/api/shift/summary").json())
    assert s.operator_id == "OP1001" and s.tasks
    assert client.get("/api/operators/NOPE/skill").status_code == 404


def test_brief_and_evidence(client):
    b = client.get("/api/brief").json()
    assert len(b["rows"]) == 4
    ev = client.get("/api/evidence/analytics").json()
    assert ev["brief_rows"]["pass"] and ev["incident_log"]["ok"]
