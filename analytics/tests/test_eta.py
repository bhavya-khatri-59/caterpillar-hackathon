import pytest
from hypothesis import given, settings, strategies as st

from analytics.generator import prior_minutes, production_rate_m3_h


def test_production_formula_by_hand():
    # Q = V x FF x E x 3600 / t = 1.0 x 0.9 x 0.75 x 3600 / 24 = 101.25 m3/h
    assert production_rate_m3_h("trenching") == pytest.approx(101.25)
    assert prior_minutes("trenching", 101.25) == pytest.approx(60.0)


def test_coverage_within_target(svc):
    m = svc.eta_metrics
    assert abs(m["coverage_p10_p90_calibrated"] - 0.80) <= 0.05


def test_model_beats_formula_prior(svc):
    m = svc.eta_metrics
    assert m["mae_p50_min_model"] < m["mae_p50_min_prior"]
    assert m["pinball_model"] < m["pinball_prior_pm20pct"]


@settings(max_examples=60, deadline=None)
@given(
    ttype=st.sampled_from(["trenching", "truck_loading", "backfill"]),
    qty=st.floats(10, 500), rain=st.floats(0, 20), vis=st.floats(30, 5000),
    ground=st.sampled_from(["dry", "wet", "mud"]), slope=st.floats(0, 20), trucks=st.floats(0.1, 1.0),
    hour=st.integers(6, 19), temp=st.floats(-5, 45), hum=st.floats(10, 100),
    op=st.sampled_from(["OP1001", "OP1009", "OP9999"]),
)
def test_quantiles_never_cross(svc, ttype, qty, rain, vis, ground, slope, trucks, hour, temp, hum, op):
    cond = {"rain_mm_h": rain, "visibility_m": vis, "ground": ground, "temp_c": temp, "humidity_pct": hum}
    r = svc.eta.predict_one(ttype, qty, cond, hour, op, 5000, slope, trucks)
    assert 0 < r["p10"] <= r["p50"] <= r["p90"]
    assert [s.label.split(" ")[0] for s in r["breakdown"]] == ["Formula", "Conditions", "Your", "Range"]


def test_breakdown_adds_up(svc):
    cond = {"rain_mm_h": 7.0, "visibility_m": 150, "ground": "mud", "temp_c": 25, "humidity_pct": 92}
    r = svc.eta.predict_one("trenching", 150, cond, 13, "OP1001", 6600, 4, 1.0)
    prior, conditions, operator, _ = (s.minutes for s in r["breakdown"])
    assert prior + conditions + operator == pytest.approx(r["p50"], abs=0.3)


def test_mud_is_slower_than_dry(svc):
    dry = {"rain_mm_h": 0, "visibility_m": 2000, "ground": "dry", "temp_c": 25, "humidity_pct": 50}
    mud = {"rain_mm_h": 7, "visibility_m": 150, "ground": "mud", "temp_c": 25, "humidity_pct": 92}
    a = svc.eta.predict_one("trenching", 150, dry, 9, "OP1001", 6600, 4, 1.0)
    b = svc.eta.predict_one("trenching", 150, mud, 9, "OP1001", 6600, 4, 1.0)
    assert b["p50"] > a["p50"] * 1.2 and (b["p90"] - b["p10"]) > (a["p90"] - a["p10"])


def test_unknown_operator_gets_neutral_factor(svc):
    assert svc.eta.op_factor("OP9999") == (1.0, 0)
