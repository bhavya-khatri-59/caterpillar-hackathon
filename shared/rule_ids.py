"""Fixed list of anomaly rule IDs. M2 raises flags with these; M4 maps them to lessons."""

IDLE_CONTINUOUS = "idle_continuous"
IDLE_RATIO_HIGH = "idle_ratio_high"
BELT_OFF_ENGINE_ON = "belt_off_engine_on"
BELT_OFF_IDLE = "belt_off_idle"
AFTER_HOURS_RUNNING = "after_hours_running"
CYCLE_TIME_DRIFT = "cycle_time_drift"
FUEL_PER_CYCLE_HIGH = "fuel_per_cycle_high"
UNATTENDED_IDLE = "unattended_idle"

ALL_RULES = (
    IDLE_CONTINUOUS,
    IDLE_RATIO_HIGH,
    BELT_OFF_ENGINE_ON,
    BELT_OFF_IDLE,
    AFTER_HOURS_RUNNING,
    CYCLE_TIME_DRIFT,
    FUEL_PER_CYCLE_HIGH,
    UNATTENDED_IDLE,
)
