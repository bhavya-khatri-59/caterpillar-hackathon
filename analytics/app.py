"""M2 REST API. Standalone: `uvicorn analytics.app:app --port 8002`.

Integration (api/main.py):

    from analytics.app import router as analytics_router
    from analytics.service import get_service
    app.include_router(analytics_router)
    sink = get_service().store          # EventSink for the safety engine
"""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analytics.service import get_service
from shared.contracts import (AlertEvent, AlertType, AnomalyFlag, Conditions, IncidentRecord, ShiftSummary,
                              SkillProfile, Task, TaskPlan, Tier)

router = APIRouter(tags=["analytics"])


# ---- request / response models owned by M2 (not in the shared contract) ------------------------

class ManualReport(BaseModel):
    """A report made by voice or with one button once the machine is stopped."""
    operator_id: str
    machine_id: str = "EXC001"
    category: Literal["near_miss", "incident", "hazard", "emergency"]
    method: Literal["voice", "button"] = "button"
    note: Optional[str] = None
    t: Optional[float] = None                  # sim time if known
    conditions: Optional[Conditions] = None    # defaults to the current forecast hour


class EventIn(BaseModel):
    """Append an AlertEvent over HTTP (the safety engine running as a separate process)."""
    event: AlertEvent
    outcome: Optional[Literal["near_miss", "incident", "cleared", "dismissed"]] = None
    note: Optional[str] = None


class VerifyResult(BaseModel):
    ok: bool
    count: int
    head_hash: str
    first_broken_seq: Optional[int] = None
    reason: Optional[str] = None


class WhatIfRequest(BaseModel):
    operator_id: Optional[str] = None
    rain_mm_h: Optional[float] = Field(None, ge=0, le=50)
    visibility_m: Optional[float] = Field(None, gt=0, le=10000)
    ground: Optional[Literal["dry", "wet", "mud"]] = None
    trucks_available: Optional[float] = Field(None, ge=0.1, le=1.0, description="share of planned trucks turning up")
    fuel_pct: Optional[float] = Field(None, ge=0, le=100)


class WhatIfResponse(BaseModel):
    tasks: list[Task]
    plan: TaskPlan


class BriefRow(BaseModel):
    row: int
    timestamp: str
    machine_id: str
    operator_id: str
    engine_hours: float
    fuel_used_l: float
    load_cycles: int
    idling_time_min: float
    seatbelt_status: str
    safety_alert_triggered: bool


class BriefResponse(BaseModel):
    rows: list[BriefRow]
    flags: list[AnomalyFlag]


MANUAL_ACTIONS = {
    "emergency": "Stop work, make the area safe and call the supervisor.",
    "incident": "Stop work; the supervisor reviews the report before restart.",
    "near_miss": "Supervisor reviews the report before the next shift.",
    "hazard": "Mark the hazard and tell the next operator in the handover.",
}


# ---- incidents -------------------------------------------------------------------------------

@router.get("/api/incidents", response_model=list[IncidentRecord])
def list_incidents(operator_id: Optional[str] = None, tier: Optional[Tier] = None, type: Optional[AlertType] = None,
                   limit: int = Query(500, ge=1, le=5000)):
    recs = get_service().store.records(operator_id)
    if tier is not None:
        recs = [r for r in recs if r.event.tier == tier]
    if type is not None:
        recs = [r for r in recs if r.event.type == type]
    return recs[-limit:]


@router.post("/api/incidents", response_model=IncidentRecord, status_code=201)
def report_incident(report: ManualReport):
    svc = get_service()
    kind = AlertType.emergency if report.category == "emergency" else AlertType.manual
    event = AlertEvent(
        alert_id=f"MAN-{uuid.uuid4().hex[:8]}", t=report.t if report.t is not None else 0.0,
        machine_id=report.machine_id, operator_id=report.operator_id,
        tier=Tier.hard if report.category == "emergency" else Tier.soft, type=kind, source="operator",
        reasons=[f"Reported by {report.method}: {report.category.replace('_', ' ')}"],
        action=MANUAL_ACTIONS[report.category], conditions=report.conditions or svc.current_conditions(),
    )
    outcome = report.category if report.category in ("near_miss", "incident") else None
    return svc.store.append(event, outcome=outcome, note=report.note)


@router.post("/api/incidents/events", response_model=IncidentRecord, status_code=201)
def append_event(body: EventIn):
    return get_service().store.append(body.event, outcome=body.outcome, note=body.note)


@router.get("/api/incidents/verify", response_model=VerifyResult)
def verify_incidents():
    return get_service().store.verify()


# ---- tasks -----------------------------------------------------------------------------------

@router.get("/api/tasks", response_model=list[Task])
def get_tasks(operator_id: Optional[str] = None):
    return get_service().plan(operator_id)[0]


@router.get("/api/tasks/plan", response_model=TaskPlan)
def get_plan(operator_id: Optional[str] = None):
    return get_service().plan(operator_id)[1]


@router.post("/api/eta/whatif", response_model=WhatIfResponse)
def what_if(req: WhatIfRequest):
    tasks, plan = get_service().planner(req.operator_id, req.rain_mm_h, req.visibility_m, req.ground,
                                        req.trucks_available, req.fuel_pct).plan()
    return WhatIfResponse(tasks=tasks, plan=plan)


# ---- operators -------------------------------------------------------------------------------

def _known(operator_id: str) -> None:
    svc = get_service()
    if operator_id not in svc.operators.index and operator_id not in set(svc.brief.operator_id):
        raise HTTPException(404, f"unknown operator {operator_id}")


@router.get("/api/operators/{operator_id}/anomalies", response_model=list[AnomalyFlag])
def get_anomalies(operator_id: str, shift_date: Optional[str] = None, since: Optional[str] = None,
                  severity: Optional[Literal["info", "review", "coach"]] = None, limit: int = Query(100, ge=1, le=2000)):
    _known(operator_id)
    flags = get_service().all_flags(operator_id)
    if shift_date:
        flags = [f for f in flags if f.shift_date == shift_date]
    if since:
        flags = [f for f in flags if f.shift_date >= since]
    if severity:
        flags = [f for f in flags if f.severity == severity]
    return flags[:limit]


@router.get("/api/operators/{operator_id}/skill", response_model=SkillProfile)
def get_skill(operator_id: str):
    _known(operator_id)
    return get_service().skill(operator_id)


# ---- shift, brief, evidence --------------------------------------------------------------------

@router.get("/api/shift/summary", response_model=ShiftSummary)
def shift_summary(operator_id: Optional[str] = None, date: Optional[str] = None):
    if operator_id:
        _known(operator_id)
    return get_service().summary(operator_id, date)


@router.get("/api/brief", response_model=BriefResponse)
def brief():
    svc = get_service()
    rows = [BriefRow(row=int(r.row), timestamp=f"{r.timestamp:%Y-%m-%d %H:%M}", machine_id=r.machine_id,
                     operator_id=r.operator_id, engine_hours=r.engine_hours, fuel_used_l=r.fuel_used_l,
                     load_cycles=int(r.load_cycles), idling_time_min=r.idling_time_min,
                     seatbelt_status=r.seatbelt_status, safety_alert_triggered=bool(r.safety_alert_triggered))
            for r in svc.brief.itertuples()]
    return BriefResponse(rows=rows, flags=svc.brief_flags)


@router.get("/api/evidence/analytics")
def evidence():
    return get_service().evidence()


@router.get("/api/analytics/health")
def health():
    svc = get_service()
    return {"status": "ok", "incidents": len(svc.store), "today": svc.today["date"], "operator": svc.today["operator_id"]}


# ---- standalone app ----------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(_: FastAPI):
    get_service()          # build data and models before the first request (about 7 s)
    yield


app = FastAPI(title="Foresight analytics (M2)", version="0.1.0", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
