# M2: data and analytics (`analytics/`, `data/`)

This module covers the shift-level work: the brief's telemetry, the seeded history, anomaly flags, the hash-chained incident log, P10/P50/P90 task times, the weather-aware task order, the fuel plan, skill levels and the shift summary. It imports only from `shared/`.

```bash
pip install -r analytics/requirements.txt
uvicorn analytics.app:app --port 8002        # standalone service (builds data + models on start, about 7 s)
python -m analytics.demo_brief               # prints the flags for the brief's 4 rows
python -m analytics.evaluate                 # writes analytics/evidence/analytics_metrics.json
python -m analytics.export_fixtures          # refreshes M2's fixtures in shared/fixtures/
python -m analytics.generator                # writes the seeded data to data/generated/ for inspection
python -m pytest shared analytics            # run from the repo root
```

Environment: `ANALYTICS_DB` (SQLite path, default `analytics/out/incidents.db`), `ANALYTICS_SEED_FIXTURES=0` to start with an empty log (by default the standalone service loads `shared/fixtures/alerts.json` into an empty log). Thresholds are in `analytics/settings.py`, each with its source. You can override them under `analytics:` in the root `config.yaml`.

## Measured results (`python -m analytics.evaluate`, seed 42)

| Metric | Result | Target |
| --- | --- | --- |
| Brief rows flagged | rows 2 and 4 only | rows 2 and 4, not 1 and 3 |
| P10–P90 coverage, held-out later shifts | 80.4% calibrated (81.3% raw) | 80% ± 5 |
| P50 mean absolute error | 9.8 min (model) vs 22.1 min (formula prior alone) | lower than prior |
| Pinball loss (mean of 0.1/0.5/0.9) | 3.15 vs 8.44 (prior ± 20%) | lower |
| Planted-pattern recall, per shift | 96.1% (rules + z-score) vs 78.7% (rules only) | > 90% |
| Precision of review/coach flags | 75.0% (rules + z-score) vs 95.2% (rules only) | reported, not targeted |
| Task order on today's seeded shift | "Do T2 (backfill), T4 (trenching) before the 12:00 rain: saves about 16 min at P50." | optimiser unit tests pass |

The data is seeded and synthetic, in the brief's columns, so the models recover effects we planted. The point is the pipeline, which retrains on real history. The z-score layer finds the cycle-drift and fuel patterns that the rules can't see. The cost is lower precision, because it also flags noisy shifts. Both numbers go on the evidence page.

## Endpoints

All responses use models from `shared/contracts.py` unless noted. Operator defaults to `today.json`'s operator (`OP1001`).

| Endpoint | Returns | Notes |
| --- | --- | --- |
| `GET /api/incidents?operator_id=&tier=&type=&limit=` | `IncidentRecord[]` | oldest first |
| `POST /api/incidents` | `IncidentRecord` (201) | manual report: `{"operator_id", "category": "near_miss"\|"incident"\|"hazard"\|"emergency", "method": "voice"\|"button", "note"?, "machine_id"?, "t"?, "conditions"?}` |
| `POST /api/incidents/events` | `IncidentRecord` (201) | `{"event": AlertEvent, "outcome"?, "note"?}`: for M1 if it runs as a separate process |
| `GET /api/incidents/verify` | `{"ok", "count", "head_hash", "first_broken_seq", "reason"}` | recomputes the hash chain |
| `GET /api/tasks?operator_id=` | `Task[]` | estimates at each task's slot in the suggested order |
| `GET /api/tasks/plan?operator_id=` | `TaskPlan` | |
| `POST /api/eta/whatif` | `{"tasks": Task[], "plan": TaskPlan}` | body, all optional: `operator_id`, `rain_mm_h`, `visibility_m`, `ground`, `trucks_available` (0.1–1), `fuel_pct` |
| `GET /api/operators/{id}/anomalies?shift_date=&since=&severity=&limit=` | `AnomalyFlag[]` | newest first; includes the brief's rows for OP1001 and `unattended_idle` events from the log |
| `GET /api/operators/{id}/skill` | `SkillProfile` | 404 for unknown operators |
| `GET /api/shift/summary?operator_id=&date=` | `ShiftSummary` | `handover_note` is left null for M4 |
| `GET /api/brief` | `{"rows": [...], "flags": AnomalyFlag[]}` | the brief's table plus its flags (demo scenario 4) |
| `GET /api/evidence/analytics` | JSON | same content as `evidence/analytics_metrics.json` |
| `GET /api/analytics/health` | JSON | |

Flag IDs: `brief-r{row}-{rule}` for the brief's rows, `{operator}-{date}-{rule}` for seeded shifts, `evt-{alert_id}-{rule}` for flags from the log. `rule` is always one of `shared/rule_ids.py`.

## Integration (M4, `api/main.py`)

```python
from analytics.app import router as analytics_router
from analytics.service import get_service

app.include_router(analytics_router)
store = get_service().store      # implements shared.contracts.EventSink: pass it to the safety engine
```

`get_service()` builds everything on first call (about 7 s), so call it at startup. Set `ANALYTICS_SEED_FIXTURES=0` so that the integrated log starts empty and only receives M1's events.

`content/risk_actions.yaml` (M4): if it exists, task risk-factor actions are read from it, keyed as `{people_density|slope|visibility|ground|heat: {med: "...", high: "..."}}`. Missing keys fall back to the defaults in `analytics/risk.py`.

## Methods in one line each

- **Anomalies**
  - Rules (belt off with engine on > 2 min, continuous idle > 10 min, idle ratio > 40% per shift or > 75% per hour, engine-meter hours not covered by logged windows across off-shift time).
  - Plus a modified z-score (Iglewicz–Hoaglin, cutoff 3.5) against the operator's last 20 clean shifts. Shifts that raised a flag don't feed later baselines.
  - Cycle-time drift compares the median of the last 3 shifts with that baseline.
  - The isolation forest was cut (first on the cut list).
- **Incident log:** SQLite with `h_i = SHA256(h_{i-1} || canonical_json(row))`. Triggers reject UPDATE and DELETE; direct file edits are caught by `verify`, which names the first broken row.
- **Task time**
  - Production-formula prior (Q = V × FF × E × 3600 / t) × XGBoost `reg:quantileerror` on log(actual / prior).
  - × an operator factor `(n r̄ + 5) / (n + 5)`.
  - Conformalised quantile regression on a time-based split. Quantiles are sorted so they never cross.
- **Task order:** every order of up to 6 pending tasks through the ETA model under the hourly forecast. The lowest total P50 wins if its P90 (spreads added in quadrature) ends within the shift. The priority order is kept unless another order saves at least 5 min.
- **Fuel plan:** P50 hours × burn rate per task type from history, against the tank minus a 10% reserve.
- **Skill:** per task type, from shifts on that type, cycle time vs the fleet median, machine-type hours, soft alerts per hour and flags in the last 10 shifts. Promotion is offered after 5 consecutive qualifying shifts.

## Known gaps

- `data/brief_telemetry.csv` engine-hour values are placeholders apart from the 3.7 h overnight rise. Replace them with the brief's verbatim values (see `data/README.md`).
- `shared/contracts.py` and `shared/rule_ids.py` are copied verbatim from the team split doc so this module runs. M4 owns them.
- `shared/fixtures/alerts.json` is a hand-made stand-in until M1 exports real events.
- No live ETA update during a task (blending the model with the observed rate); the contract has no progress field yet.
