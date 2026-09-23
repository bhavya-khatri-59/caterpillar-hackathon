"""Recomputes the analytics evidence and writes analytics/evidence/analytics_metrics.json.

    python -m analytics.evaluate
"""
from __future__ import annotations

import json

from analytics.service import AnalyticsService
from analytics.settings import EVIDENCE_PATH


def main() -> dict:
    svc = AnalyticsService(db_path=":memory:", seed_fixtures=True)
    ev = svc.evidence()
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(ev, indent=2) + "\n")
    tt, an = ev["task_time"], ev["anomaly"]["rules_plus_z"]
    ro = ev["anomaly"]["rules_only"]
    print(f"P10-P90 coverage (test): raw {tt['coverage_p10_p90_raw']:.1%}, calibrated "
          f"{tt['coverage_p10_p90_calibrated']:.1%} (target {tt['coverage_target']:.0%})")
    print(f"P50 MAE: model {tt['mae_p50_min_model']} min vs formula prior {tt['mae_p50_min_prior']} min")
    print(f"Anomaly recall {an['recall']:.1%}, precision {an['precision']:.1%} "
          f"(rules only: recall {ro['recall']:.1%}, precision {ro['precision']:.1%})")
    print(f"Brief rows flagged {ev['brief_rows']['flagged_rows']} -> {'PASS' if ev['brief_rows']['pass'] else 'FAIL'}")
    print(f"Task order: {ev['task_order']['reason']}")
    print(f"wrote {EVIDENCE_PATH}")
    return ev


if __name__ == "__main__":
    main()
