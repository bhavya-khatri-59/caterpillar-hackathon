"""Every analytics threshold in one place, each with its source.

Values can be overridden from the repo-root `config.yaml` under an `analytics:` key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent / "out"
EVIDENCE_PATH = Path(__file__).resolve().parent / "evidence" / "analytics_metrics.json"


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    n_operators: int = 30
    n_shifts: int = 60

    # --- anomaly rules -------------------------------------------------------
    idle_continuous_min: float = 10.0        # inside Cat Engine Idle Shutdown's 5-60 min range
    idle_ratio_shift_max: float = 0.40       # per shift, configurable per site (design doc)
    idle_ratio_window_max: float = 0.75      # one hour window; 50% idle in an hour happens in truck waits
    belt_off_min: float = 2.0                # belt off with engine on for over 2 min (design doc)
    after_hours_unaccounted_h: float = 0.5   # engine-meter hours not covered by logged windows
    z_cutoff: float = 3.5                    # modified z-score cutoff, Iglewicz and Hoaglin (1993)
    baseline_shifts: int = 20                # personal baseline = last 20 shifts
    min_baseline_n: int = 10                 # fewer shifts than this -> no z-score flag (MAD too noisy)
    drift_window: int = 3                    # cycle-time drift = median of the last 3 shifts vs baseline

    # --- task time -----------------------------------------------------------
    bucket_m3: float = 1.0                   # 20 t class excavator, general-purpose bucket
    job_efficiency: float = 0.75             # 45-min hour, Cat Performance Handbook convention
    fill_factor: dict = field(default_factory=lambda: {"trenching": 0.90, "truck_loading": 1.00, "backfill": 1.10})
    cycle_s: dict = field(default_factory=lambda: {"trenching": 24.0, "truck_loading": 18.0, "backfill": 14.0})
    coverage_target: float = 0.80            # P10-P90 band
    operator_k: float = 5.0                  # shrinkage for the operator factor, f = (n r + k)/(n + k)

    min_reorder_saving_min: float = 5.0      # below this, keep the supervisor's priority order

    # --- fuel ----------------------------------------------------------------
    tank_l: float = 345.0                    # 20 t class excavator fuel tank
    fuel_reserve_pct: float = 10.0           # never plan below this
    idle_burn_l_h: float = 2.0               # upper bound from brief row 4: 2.0 L over 60 idle min

    # --- skill ---------------------------------------------------------------
    promotion_streak: int = 5                # consecutive shifts meeting the next level's criteria

    # --- storage -------------------------------------------------------------
    db_path: str = field(default_factory=lambda: os.environ.get("ANALYTICS_DB", str(OUT_DIR / "incidents.db")))


def _load_overrides() -> dict:
    cfg = ROOT / "config.yaml"
    if not cfg.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(cfg.read_text()) or {}
    except Exception:
        return {}
    section = data.get("analytics") or {}
    known = {f.name for f in fields(Settings)}
    return {k: v for k, v in section.items() if k in known}


SETTINGS = replace(Settings(), **_load_overrides())
