from analytics.anomaly import window_flags
from analytics.brief import load_brief


def test_brief_loads_verbatim():
    b = load_brief()
    assert list(b.load_cycles) == [12, 2, 10, 1]
    assert list(b.fuel_used_l) == [5.2, 3.8, 6.1, 2.0]
    assert list(b.idling_time_min) == [30, 55, 15, 60]
    assert list(b.seatbelt_status) == ["Fastened", "Unfastened", "Fastened", "Unfastened"]
    assert list(b.safety_alert_triggered) == [False, True, False, True]
    assert round(b.engine_hours.iloc[3] - b.engine_hours.iloc[2], 1) == 3.7


def test_rows_2_and_4_flagged_and_1_and_3_not(svc):
    rows = {int(f.flag_id.split("-")[1][1:]) for f in svc.brief_flags}
    assert rows == {2, 4}


def test_brief_flags_carry_reasons(svc):
    r2 = [f for f in svc.brief_flags if f.flag_id.startswith("brief-r2-")]
    assert {"belt_off_idle", "idle_ratio_high"} <= {f.rule for f in r2}
    idle = next(f for f in r2 if f.rule == "idle_ratio_high")
    assert "55 of 60 min (92%)" in idle.reason and "usual" in idle.reason


def test_overnight_meter_gap_is_review_not_a_claim(svc):
    f = next(f for f in svc.brief_flags if f.rule == "after_hours_running")
    assert f.flag_id.startswith("brief-r4-") and f.severity == "review"
    assert f.value == 2.7 and "either" in f.reason


def test_brief_without_history_still_flags_rules():
    rows = {int(f.flag_id.split("-")[1][1:]) for f in window_flags(load_brief(), load_brief().iloc[0:0])}
    assert rows == {2, 4}
