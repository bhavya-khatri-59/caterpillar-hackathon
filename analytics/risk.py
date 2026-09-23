"""Named risk factors per task. No single overall risk score: each factor says what it is and what to do.

Actions come from `content/risk_actions.yaml` (M4) when present, keyed by factor name and level:

    people_density: {med: "...", high: "..."}
    slope: {...}; visibility: {...}; ground: {...}; heat: {...}

Missing keys fall back to the defaults below.
"""
from __future__ import annotations

from functools import lru_cache

from analytics.settings import ROOT
from shared.contracts import RiskFactor

DEFAULT_ACTIONS = {
    "people_density": {"med": "Confirm the exclusion zone with the ground crew before starting.",
                       "high": "Agree a spotter and a fixed walkway before starting; keep people out of the swing radius."},
    "slope": {"med": "Keep the tracks square to the slope and the load low when swinging.",
              "high": "Bench the slope or reposition; check the lift chart for the reduced capacity."},
    "visibility": {"med": "Use lights and slow the swing near the spoil pile.",
                   "high": "Slow down, use a spotter, and stop if you lose sight of the ground crew."},
    "ground": {"med": "Allow longer stopping distance on wet ground; avoid sharp travel turns.",
               "high": "Mud: travel slowly, keep off soft edges, and allow extra stopping distance."},
    "heat": {"med": "Drink water every hour; take the scheduled break.",
             "high": "Heat stress risk: take a 10-minute cool-down break every hour."},
}


@lru_cache(maxsize=1)
def _actions() -> dict:
    path = ROOT / "content" / "risk_actions.yaml"
    merged = {k: dict(v) for k, v in DEFAULT_ACTIONS.items()}
    if path.exists():
        try:
            import yaml

            data = yaml.safe_load(path.read_text()) or {}
            for k, v in data.items():
                if isinstance(v, dict) and k in merged:
                    merged[k].update({lvl: str(t) for lvl, t in v.items() if lvl in ("low", "med", "high")})
        except Exception:
            pass
    return merged


def _factor(name: str, level: str, note: str) -> RiskFactor:
    return RiskFactor(name=name, level=level, note=note, action=_actions()[name].get(level, ""))


def task_risk_factors(zone: dict, cond: dict, heat_index: float | None) -> list[RiskFactor]:
    """Only factors at med or high are returned; an empty list means nothing notable."""
    out = []
    pd_level = zone.get("people_density", "low")
    if pd_level in ("med", "high"):
        out.append(_factor("people_density", pd_level, f"{pd_level.replace('med', 'moderate').capitalize()} foot traffic in this zone"))
    slope = float(zone.get("slope_deg", 0.0))
    if slope >= 5:
        out.append(_factor("slope", "high" if slope >= 10 else "med", f"Side slope about {slope:.0f} deg"))
    vis = float(cond["visibility_m"])
    if vis < 500:
        out.append(_factor("visibility", "high" if vis < 200 else "med", f"Visibility about {vis:.0f} m"))
    if cond["ground"] in ("wet", "mud"):
        out.append(_factor("ground", "high" if cond["ground"] == "mud" else "med",
                           f"{cond['ground'].capitalize()} ground, rain {cond['rain_mm_h']:.0f} mm/h"))
    if heat_index is not None and heat_index >= 32:
        out.append(_factor("heat", "high" if heat_index >= 41 else "med", f"Heat index {heat_index:.0f} C"))
    return out
