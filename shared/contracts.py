# shared/contracts.py (frozen at Hour 1; add optional fields only, with all four agreeing)
from __future__ import annotations
from enum import Enum
from typing import Literal, Optional, Protocol
from pydantic import BaseModel

class Tier(str, Enum):
    log = "log"; soft = "soft"; hard = "hard"

class AlertType(str, Enum):
    proximity = "proximity"; stability = "stability"; seatbelt = "seatbelt"
    seatbelt_bypass = "seatbelt_bypass"; stale_sensor = "stale_sensor"
    unattended_idle = "unattended_idle"; manual = "manual"; emergency = "emergency"

class Conditions(BaseModel):
    ground: Literal["dry", "wet", "mud"]
    mu: float                      # ground friction
    visibility_m: float
    rain_mm_h: float
    temp_c: float
    humidity_pct: float
    heat_index_c: Optional[float] = None

class MachineState(BaseModel):
    machine_id: str; operator_id: str
    x: float; y: float; heading_deg: float; speed_mps: float
    swing_deg: float; swing_rate_dps: float; boom_reach_m: float
    load_t: float; slope_deg: float; stability_ratio: float
    engine_on: bool; belt_fastened: bool; seat_occupied: bool
    hydraulic_lockout: bool; motion_commanded: bool
    fuel_pct: float

class TrackedObject(BaseModel):
    id: str; cls: Literal["person", "vehicle", "object"]
    x: float; y: float; vx: float; vy: float
    t_cpa_s: Optional[float] = None; d_cpa_m: Optional[float] = None
    predicted_path: list[tuple[float, float]] = []

class Zones(BaseModel):
    inner_m: float; warn_m: float; outer_m: float

class AlertEvent(BaseModel):
    alert_id: str; t: float; machine_id: str; operator_id: str
    tier: Tier; type: AlertType; source: Literal["rule", "model", "operator"]
    p_nearmiss: Optional[float] = None
    t_cpa_s: Optional[float] = None; d_cpa_m: Optional[float] = None
    warning_time_s: Optional[float] = None   # filled when the hazard resolves
    reasons: list[str]                        # from rules / feature contributions, never an LLM
    action: str                               # from content/risk_actions.yaml
    conditions: Conditions
    model_version: Optional[str] = None

class AckRequest(BaseModel):
    alert_id: str; method: Literal["voice", "button"]
    action: Literal["ack", "dismiss"]
    reason_code: Optional[Literal["known_hazard", "spotter_present", "false_alarm"]] = None

class LiveFrame(BaseModel):                   # 10 Hz over /ws/live
    t: float; scenario_id: str; seed: int
    machine: MachineState; objects: list[TrackedObject]
    zones: Zones; conditions: Conditions
    active_alert: Optional[AlertEvent] = None
    sensor_ok: bool

class IncidentRecord(BaseModel):
    seq: int; event: AlertEvent
    outcome: Optional[Literal["near_miss", "incident", "cleared", "dismissed"]] = None
    note: Optional[str] = None
    prev_hash: str; hash: str

class RiskFactor(BaseModel):
    name: str; level: Literal["low", "med", "high"]; note: str; action: str

class EtaStep(BaseModel):
    label: str; minutes: float                # prior -> conditions -> operator -> range

class Task(BaseModel):
    task_id: str; type: Literal["trenching", "truck_loading", "backfill"]
    zone: str; quantity_m3: float; priority: int
    status: Literal["pending", "active", "done"]
    p10_min: float; p50_min: float; p90_min: float
    eta_breakdown: list[EtaStep]; risk_factors: list[RiskFactor]

class TaskPlan(BaseModel):
    order: list[str]; total_p50_min: float; saving_min: float; reason: str
    fuel_needed_l: float; refuel_before_task: Optional[str] = None

class AnomalyFlag(BaseModel):
    flag_id: str; operator_id: str; shift_date: str
    rule: str                                 # one of shared/rule_ids.py
    metric: str; value: float
    baseline: Optional[float] = None; z: Optional[float] = None
    severity: Literal["info", "review", "coach"]; reason: str

class SkillProfile(BaseModel):
    operator_id: str
    levels: dict[str, Literal["guided", "standard", "expert"]]   # per task type
    inputs: dict[str, float]                  # machine-type hours, total hours, anomaly rate ...
    promotion_ready: bool

class QuizItem(BaseModel):
    q: str; options: list[str]; answer_idx: int

class Lesson(BaseModel):
    lesson_id: str; title: str; trigger_rules: list[str]; minutes: int
    card_md: str; video_path: Optional[str] = None; quiz: list[QuizItem]

class Tip(BaseModel):
    tip_id: str; type: str; text: str; reason: str
    level: Literal["guided", "standard", "expert"]
    feedback: Optional[Literal["up", "down"]] = None

class ShiftSummary(BaseModel):
    operator_id: str; date: str
    near_misses: int; median_warning_s: Optional[float]
    belt_compliance_pct: float; idle_pct: float; fuel_l: float
    tasks: list[Task]; went_well: str; tip: str
    handover_note: Optional[str] = None

class AgentAnswer(BaseModel):
    text: str; intent: Optional[str] = None; tools_used: list[str]

class EventSink(Protocol):
    def emit(self, event: AlertEvent) -> None: ...
