import pandas as pd
import pytest

from analytics.anomaly import meter_gaps, modified_z
from shared.rule_ids import ALL_RULES


def test_modified_z_by_hand():
    hist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    z, med = modified_z(15, hist)
    # median 5.5, MAD 2.5 -> 0.6745 x 9.5 / 2.5
    assert med == 5.5 and z == pytest.approx(0.6745 * 9.5 / 2.5)


def test_modified_z_needs_enough_history():
    assert modified_z(5, [1, 2, 3])[0] is None


def test_modified_z_zero_mad_falls_back():
    z, _ = modified_z(2.0, [1.0] * 9 + [1.5] * 3)
    assert z is not None and z > 0


def test_every_flag_uses_a_known_rule_and_has_a_reason(svc):
    for f in svc.shift_flags + svc.brief_flags:
        assert f.rule in ALL_RULES and len(f.reason) > 20


def test_planted_recall_above_90pct(svc):
    m = svc.evidence()["anomaly"]["rules_plus_z"]
    assert m["recall"] > 0.90
    assert all(v["detected"] / v["planted"] >= 0.7 for v in m["per_pattern"].values())


def test_normal_operator_is_quiet(svc):
    flags = [f for f in svc.shift_flags if f.operator_id == "OP1001" and f.severity != "info"]
    assert len(flags) <= 2


def test_in_shift_meter_gap_is_not_after_hours():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-05-01 10:00", "2025-05-01 14:00", "2025-05-02 09:00"]),
        "machine_id": "M", "engine_hours": [100.0, 104.0, 105.0], "window_min": 60.0,
        "shift_start_h": 7, "shift_end_h": 17,
    })
    assert list(meter_gaps(df)) == [0.0, 0.0, 0.0]
    df.loc[2, "engine_hours"] = 106.5
    assert list(meter_gaps(df)) == [0.0, 0.0, 1.5]
