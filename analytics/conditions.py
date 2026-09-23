"""Site conditions helpers: friction by ground state, NWS heat index."""
from __future__ import annotations

from shared.contracts import Conditions

MU = {"dry": 0.60, "wet": 0.40, "mud": 0.25}


def heat_index_c(temp_c: float, humidity_pct: float) -> float:
    """NWS Rothfusz regression (with the simple formula below 80 F), returned in Celsius."""
    t = temp_c * 9 / 5 + 32
    rh = humidity_pct
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    if (simple + t) / 2 < 80:
        hi = simple
    else:
        hi = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
              - 0.00683783 * t * t - 0.05481717 * rh * rh + 0.00122874 * t * t * rh
              + 0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh)
        if rh < 13 and 80 <= t <= 112:
            hi -= ((13 - rh) / 4) * ((17 - abs(t - 95)) / 17) ** 0.5
        elif rh > 85 and 80 <= t <= 87:
            hi += ((rh - 85) / 10) * ((87 - t) / 5)
    return round((hi - 32) * 5 / 9, 1)


def make_conditions(ground: str = "dry", rain_mm_h: float = 0.0, visibility_m: float = 2000.0,
                    temp_c: float = 25.0, humidity_pct: float = 50.0) -> Conditions:
    return Conditions(
        ground=ground, mu=MU[ground], visibility_m=float(visibility_m), rain_mm_h=float(rain_mm_h),
        temp_c=float(temp_c), humidity_pct=float(humidity_pct),
        heat_index_c=heat_index_c(temp_c, humidity_pct),
    )
