# data/

| File | What it is |
| --- | --- |
| `brief_telemetry.csv` | The brief's 4-row telemetry table. Load cycles, fuel, idle, seatbelt and alert columns are the brief's values. **Check `engine_hours` against the brief before submission**: only the 3.7 h rise from 2025-05-01 14:00 to 2025-05-02 09:00 is taken from our analysis notes; the absolute meter values are placeholders. |
| `today.json` | Seeded shift for the demo: task queue, zones, hourly forecast (rain from 15:00), tank level. Edit freely; the analytics service reloads it on start. |
| `generated/` | Written by `python -m analytics.generator` (git-ignored). The service regenerates the same data from the seed in memory, so this is only for inspection. |

Assumption stated openly: each brief row covers the hour ending at its timestamp (60 idle minutes in row 4 only fits a 1-hour window), and the seatbelt status is the window's dominant state.
