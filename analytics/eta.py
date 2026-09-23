"""Task time estimation.

    minutes = prior x exp(q_model(conditions)) x f_op,   prior = quantity / Q (production formula)

- XGBoost with the native quantile objective (alphas 0.1 / 0.5 / 0.9) predicts log(actual / prior).
- Quantiles are sorted, so P10 <= P50 <= P90 always holds.
- Operator factor f_op = (n r_bar + k) / (n + k), k = 5: the operator's past actual-to-model ratio,
  pulled towards 1 when there is little history.
- Conformalized quantile regression (Romano et al., 2019) on a held-out split widens the band so
  P10-P90 covers about 80% of outcomes.
Splits are by time (earlier shifts train, later shifts test), never by row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from analytics.conditions import heat_index_c
from analytics.generator import prior_minutes
from analytics.settings import SETTINGS, Settings
from shared.contracts import EtaStep

ALPHAS = np.array([0.1, 0.5, 0.9])
TYPES = ("trenching", "truck_loading", "backfill")
GROUND_CODE = {"dry": 0, "wet": 1, "mud": 2}
EXP_BANDS = (1000, 3000, 8000)          # coarse, so the model can't identify individual operators
FEATURES = ["is_trenching", "is_loading", "is_backfill", "rain_mm_h", "log_visibility", "ground", "slope_deg",
            "trucks_available", "start_hour", "heat_index_c", "experience_band"]
REFERENCE = {"rain_mm_h": 0.0, "visibility_m": 2000.0, "ground": "dry", "slope_deg": 0.0,
             "trucks_available": 1.0, "temp_c": 25.0, "humidity_pct": 50.0}


def features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["is_trenching"] = (df.task_type == "trenching").astype(float)
    x["is_loading"] = (df.task_type == "truck_loading").astype(float)
    x["is_backfill"] = (df.task_type == "backfill").astype(float)
    x["rain_mm_h"] = df.rain_mm_h.astype(float)
    x["log_visibility"] = np.log(df.visibility_m.astype(float).clip(lower=10))
    x["ground"] = df.ground.map(GROUND_CODE).astype(float)
    x["slope_deg"] = df.slope_deg.astype(float)
    x["trucks_available"] = df.trucks_available.astype(float)
    x["start_hour"] = df.start_hour.astype(float)
    hi = df["heat_index_c"] if "heat_index_c" in df else pd.Series(
        [heat_index_c(t, h) for t, h in zip(df.temp_c, df.humidity_pct)], index=df.index)
    x["heat_index_c"] = hi.astype(float)
    x["experience_band"] = np.searchsorted(EXP_BANDS, df.experience_h.astype(float)).astype(float)
    return x[FEATURES]


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


@dataclass
class EtaModel:
    s: Settings = SETTINGS
    model: Optional[xgb.XGBRegressor] = None
    qhat: float = 0.0                              # conformal widening, log space
    op_stats: dict = field(default_factory=dict)   # operator -> (n, mean ratio)
    metrics: dict = field(default_factory=dict)

    # -- fitting -----------------------------------------------------------------------------
    def _fit_model(self, df: pd.DataFrame) -> None:
        y = np.log(df.actual_min / df.prior_min)
        self.model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=ALPHAS, n_estimators=300, learning_rate=0.05,
            max_depth=4, min_child_weight=5, subsample=0.9, tree_method="hist", random_state=self.s.seed, n_jobs=1,
        )
        self.model.fit(features(df), y)

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        """Sorted log-ratio quantiles, shape (n, 3)."""
        return np.sort(np.asarray(self.model.predict(features(df))).reshape(len(df), 3), axis=1)

    def _set_operator_stats(self, df: pd.DataFrame) -> None:
        q50 = self._raw(df)[:, 1]
        ratio = df.actual_min.to_numpy() / (df.prior_min.to_numpy() * np.exp(q50))
        g = pd.DataFrame({"op": df.operator_id.to_numpy(), "r": ratio}).groupby("op").r.agg(["count", "mean"])
        self.op_stats = {op: (int(r["count"]), float(r["mean"])) for op, r in g.iterrows()}

    def op_factor(self, operator_id: Optional[str]) -> tuple[float, int]:
        n, rbar = self.op_stats.get(operator_id, (0, 1.0))
        k = self.s.operator_k
        return (n * rbar + k) / (n + k), n

    def _log_band(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._raw(df)
        lf = np.log([self.op_factor(o)[0] for o in df.operator_id])
        return raw + lf[:, None]

    def _conformal(self, df: pd.DataFrame) -> float:
        y = np.log(df.actual_min / df.prior_min).to_numpy()
        band = self._log_band(df)
        scores = np.maximum(band[:, 0] - y, y - band[:, 2])
        n, alpha = len(scores), 1 - self.s.coverage_target
        level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
        return float(np.quantile(scores, level, method="higher"))

    def fit_evaluate(self, tasks: pd.DataFrame) -> dict:
        """Time split 60/20/20 by shift: train, calibrate, test. Stores test metrics, then refits on
        train+cal for serving. The serving band keeps the out-of-sample qhat: recomputing it on data the
        serving model was trained on would make the band too narrow."""
        n_sh = int(tasks.shift_index.max()) + 1
        tr = tasks[tasks.shift_index < int(0.6 * n_sh)]
        ca = tasks[(tasks.shift_index >= int(0.6 * n_sh)) & (tasks.shift_index < int(0.8 * n_sh))]
        te = tasks[tasks.shift_index >= int(0.8 * n_sh)]

        self._fit_model(tr)
        self._set_operator_stats(tr)
        self.qhat = self._conformal(ca)
        self.metrics = self._test_metrics(te, n_train=len(tr), n_cal=len(ca))

        # serving model: all history before the test period, so the reported metrics describe it honestly
        self._fit_model(pd.concat([tr, ca]))
        self._set_operator_stats(pd.concat([tr, ca]))
        return self.metrics

    def _test_metrics(self, te: pd.DataFrame, n_train: int, n_cal: int) -> dict:
        y = te.actual_min.to_numpy()
        prior = te.prior_min.to_numpy()
        band = self._log_band(te)
        raw = prior[:, None] * np.exp(band)
        cal = prior[:, None] * np.exp(band + np.array([-self.qhat, 0, self.qhat]))
        base = prior[:, None] * np.array([0.8, 1.0, 1.2])
        cover = lambda b: float(np.mean((y >= b[:, 0]) & (y <= b[:, 2])))
        pin = lambda b: round(float(np.mean([pinball(y, b[:, i], a) for i, a in enumerate(ALPHAS)])), 3)
        return {
            "split": "by shift in time: first 60% train, next 20% calibrate, last 20% test",
            "n_train": n_train, "n_cal": n_cal, "n_test": int(len(te)),
            "coverage_target": self.s.coverage_target,
            "coverage_p10_p90_raw": round(cover(raw), 3),
            "coverage_p10_p90_calibrated": round(cover(cal), 3),
            "coverage_prior_pm20pct": round(cover(base), 3),
            "mae_p50_min_model": round(float(np.mean(np.abs(y - cal[:, 1]))), 2),
            "mae_p50_min_prior": round(float(np.mean(np.abs(y - prior))), 2),
            "pinball_model": pin(cal), "pinball_prior_pm20pct": pin(base),
            "median_band_width_pct": round(float(np.median((cal[:, 2] - cal[:, 0]) / cal[:, 1])) * 100, 1),
            "qhat_log": round(self.qhat, 4),
        }

    # -- serving -----------------------------------------------------------------------------
    def predict_one(self, task_type: str, quantity_m3: float, cond: dict, start_hour: int,
                    operator_id: Optional[str], experience_h: float, slope_deg: float,
                    trucks_available: float) -> dict:
        """Returns p10/p50/p90 minutes and the step-by-step breakdown."""
        prior = prior_minutes(task_type, quantity_m3, self.s)
        row = {"task_type": task_type, "start_hour": start_hour, "experience_h": experience_h,
               "slope_deg": slope_deg, "trucks_available": trucks_available if task_type == "truck_loading" else 1.0,
               "operator_id": operator_id, **{k: cond[k] for k in ("rain_mm_h", "visibility_m", "ground", "temp_c", "humidity_pct")}}
        variants = {"actual": row}
        ref = {**row, **{k: v for k, v in REFERENCE.items() if k != "trucks_available"}, "slope_deg": 0.0,
               "trucks_available": 1.0}
        variants["reference"] = ref
        # one-at-a-time contributions from the reference, to name what the conditions step is made of
        named = {
            "rain and ground": {"rain_mm_h": row["rain_mm_h"], "ground": row["ground"]},
            "visibility": {"visibility_m": row["visibility_m"]},
            "slope": {"slope_deg": row["slope_deg"]},
            "trucks": {"trucks_available": row["trucks_available"]},
            "heat": {"temp_c": row["temp_c"], "humidity_pct": row["humidity_pct"]},
        }
        for name, change in named.items():
            variants[name] = {**ref, **change}
        df = pd.DataFrame(list(variants.values()))
        raw = self._raw(df)
        f_op, n = self.op_factor(operator_id)
        lo, mid, hi = raw[0] + math.log(f_op) + np.array([-self.qhat, 0, self.qhat])
        p10, p50, p90 = (prior * math.exp(v) for v in (lo, mid, hi))
        cond_p50 = prior * math.exp(raw[0, 1])
        ref_p50 = prior * math.exp(raw[1, 1])
        contribs = {name: prior * math.exp(raw[i + 2, 1]) - ref_p50 for i, name in enumerate(named)}
        top = [f"{k} {v:+.0f} min" for k, v in sorted(contribs.items(), key=lambda kv: -abs(kv[1])) if abs(v) >= 1][:2]
        steps = [
            EtaStep(label=f"Formula prior ({quantity_m3:.0f} m3 at {60 * quantity_m3 / prior:.0f} m3/h)", minutes=round(prior, 1)),
            EtaStep(label="Conditions" + (f" ({', '.join(top)})" if top else ""), minutes=round(cond_p50 - prior, 1)),
            EtaStep(label=(f"Your history (x{f_op:.2f} from {n} tasks)" if n else "Your history (no tasks yet, x1.00)"),
                    minutes=round(p50 - cond_p50, 1)),
            EtaStep(label=f"Range P10-P90 ({p10:.0f}-{p90:.0f} min)", minutes=round(p90 - p10, 1)),
        ]
        return {"p10": round(p10, 1), "p50": round(p50, 1), "p90": round(p90, 1), "prior": round(prior, 1),
                "breakdown": steps, "op_factor": round(f_op, 3)}
