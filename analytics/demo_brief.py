"""Prints the flags for the brief's rows: `python -m analytics.demo_brief`."""
from __future__ import annotations

from analytics.anomaly import window_flags
from analytics.brief import load_brief
from analytics.generator import generate


def main() -> None:
    brief = load_brief()
    history = generate()["telemetry"]
    flags = window_flags(brief, history[history.machine_id.isin(brief.machine_id)])
    by_row: dict[int, list] = {}
    for f in flags:
        by_row.setdefault(int(f.flag_id.split("-")[1][1:]), []).append(f)
    for r in brief.itertuples():
        print(f"Row {r.row}  {r.timestamp:%Y-%m-%d %H:%M}  cycles={r.load_cycles:<3} fuel={r.fuel_used_l:<4} "
              f"idle={r.idling_time_min:.0f} min  belt={r.seatbelt_status:<10} alert={'Yes' if r.safety_alert_triggered else 'No'}")
        for f in by_row.get(int(r.row), []):
            print(f"    [{f.severity:6}] {f.rule:<20} {f.reason}")
        if not by_row.get(int(r.row)):
            print("    no flags")
    flagged = sorted(by_row)
    print(f"\nFlagged rows: {flagged} (expected [2, 4]) -> {'PASS' if flagged == [2, 4] else 'FAIL'}")


if __name__ == "__main__":
    main()
