import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


TRAIN_CSV = Path("data/processed/NASDAQ_wide_train.csv")
FUTURE_CSV = Path("data/processed/NASDAQ_wide_future.csv")
TARGET_COLUMN = "Close"
VAL_RATIO = 0.2
HORIZON = 12
STRIDE = 12
SEASONAL_PERIOD = 12
OUTPUT_PREDICTIONS = Path("outputs/NASDAQ_ETS_predictions.csv")
OUTPUT_METRICS = Path("outputs/NASDAQ_ETS_metrics.json")
VERBOSE_PROGRESS = False
MAX_HISTORY = 1440
MAX_VALIDATION_WINDOWS = 48


def log_progress(prefix: str, index: int, total: int, extra: str = "") -> None:
    if not VERBOSE_PROGRESS:
        return
    if total <= 0:
        return
    interval = max(1, total // 10)
    is_boundary = index == 0 or index == total - 1
    is_interval = ((index + 1) % interval) == 0
    if not (is_boundary or is_interval):
        return
    message = "[ETS] {} {}/{}".format(prefix, index + 1, total)
    if extra:
        message += " | " + extra
    print(message, flush=True)


def load_target_series(path: Path, target_column: str) -> np.ndarray:
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise KeyError("Column '{}' not found in {}.".format(target_column, path))
    series = pd.to_numeric(df[target_column], errors="coerce")
    if series.isna().any():
        raise ValueError("Target column '{}' contains non-numeric values.".format(target_column))
    return series.to_numpy(dtype=float)


def split_train_val(series: np.ndarray, val_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if len(series) < 10:
        raise ValueError("Series is too short for ETS validation.")
    split_idx = max(5, int(len(series) * (1.0 - val_ratio)))
    split_idx = min(split_idx, len(series) - 1)
    return series[:split_idx], series[split_idx:]


def trim_history(history: np.ndarray) -> np.ndarray:
    if MAX_HISTORY <= 0 or len(history) <= MAX_HISTORY:
        return history
    return history[-MAX_HISTORY:]


def select_window_starts(total_length: int) -> List[int]:
    starts = list(range(0, total_length, STRIDE))
    if MAX_VALIDATION_WINDOWS > 0 and len(starts) > MAX_VALIDATION_WINDOWS:
        sampled = np.linspace(0, len(starts) - 1, num=MAX_VALIDATION_WINDOWS, dtype=int)
        starts = [starts[idx] for idx in sampled.tolist()]
    return starts


def validate_forecast(forecast: np.ndarray, history: np.ndarray) -> np.ndarray:
    if not np.isfinite(forecast).all():
        raise ValueError("ETS forecast produced non-finite values.")

    history_scale = float(np.max(np.abs(history))) if len(history) > 0 else 0.0
    allowed_scale = max(50.0, history_scale * 20.0)
    if float(np.max(np.abs(forecast))) > allowed_scale:
        raise ValueError(
            "ETS forecast appears unstable: max_abs_forecast={} allowed_scale={}.".format(
                float(np.max(np.abs(forecast))),
                allowed_scale,
            )
        )
    return forecast


def build_candidates(train_length: int) -> List[Dict]:
    candidates: List[Dict] = [
        {"name": "naive", "mode": "naive"},
        {
            "name": "ets_level",
            "trend": None,
            "damped_trend": False,
            "seasonal": None,
            "seasonal_periods": None,
        },
        {
            "name": "ets_add_trend",
            "trend": "add",
            "damped_trend": False,
            "seasonal": None,
            "seasonal_periods": None,
        },
        {
            "name": "ets_damped_trend",
            "trend": "add",
            "damped_trend": True,
            "seasonal": None,
            "seasonal_periods": None,
        },
    ]
    if train_length >= 2 * SEASONAL_PERIOD:
        candidates.extend(
            [
                {
                    "name": "ets_add_seasonal",
                    "trend": None,
                    "damped_trend": False,
                    "seasonal": "add",
                    "seasonal_periods": SEASONAL_PERIOD,
                },
                {
                    "name": "ets_trend_seasonal",
                    "trend": "add",
                    "damped_trend": False,
                    "seasonal": "add",
                    "seasonal_periods": SEASONAL_PERIOD,
                },
            ]
        )
    return candidates


def fit_and_forecast(history: np.ndarray, steps: int, params: Dict) -> np.ndarray:
    if len(history) == 0:
        raise ValueError("History cannot be empty.")
    if params.get("mode") == "naive":
        return np.repeat(history[-1], steps).astype(float)
    history = trim_history(history)

    model = ExponentialSmoothing(
        history,
        trend=params.get("trend"),
        damped_trend=params.get("damped_trend", False),
        seasonal=params.get("seasonal"),
        seasonal_periods=params.get("seasonal_periods"),
        initialization_method="heuristic",
    )
    fitted = model.fit(optimized=True, use_brute=False, remove_bias=False)
    forecast = np.asarray(fitted.forecast(steps), dtype=float)
    if forecast.shape[0] != steps:
        raise ValueError("ETS forecast returned {} steps, expected {}.".format(forecast.shape[0], steps))
    return validate_forecast(forecast, history)


def score_candidate(train_values: np.ndarray, val_values: np.ndarray, params: Dict) -> float:
    preds: List[float] = []
    truths: List[float] = []
    window_starts = select_window_starts(len(val_values))
    total_windows = len(window_starts)
    for window_idx, start in enumerate(window_starts):
        log_progress(
            prefix="validation windows",
            index=window_idx,
            total=total_windows,
            extra="candidate={}".format(params["name"]),
        )
        history = np.concatenate([train_values, val_values[:start]])
        steps = min(HORIZON, len(val_values) - start)
        if steps <= 0:
            break
        try:
            forecast = fit_and_forecast(history, steps, params)
        except Exception:
            return float("inf")
        preds.extend(forecast.tolist())
        truths.extend(val_values[start : start + steps].tolist())
    if not preds:
        return float("inf")
    diff = np.asarray(preds) - np.asarray(truths)
    return float(np.mean(diff * diff))


def select_best_params(train_values: np.ndarray, val_values: np.ndarray) -> Dict:
    best_params = {"name": "naive", "mode": "naive"}
    best_mse = float("inf")
    candidates = build_candidates(len(train_values))
    total_candidates = len(candidates)
    for candidate_idx, params in enumerate(candidates):
        if VERBOSE_PROGRESS:
            print(
                "[ETS] candidate {}/{} | name={}".format(candidate_idx + 1, total_candidates, params["name"]),
                flush=True,
            )
        mse = score_candidate(train_values, val_values, params)
        if VERBOSE_PROGRESS:
            print(
                "[ETS] candidate result | name={} | val_mse={:.6f}".format(params["name"], mse),
                flush=True,
            )
        if mse < best_mse:
            best_mse = mse
            best_params = params
    best_params = dict(best_params)
    best_params["val_mse"] = best_mse
    return best_params


def rolling_forecast(train_values: np.ndarray, future_values: np.ndarray, params: Dict) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    train_len = len(train_values)
    offsets = list(range(0, len(future_values), STRIDE))
    total_windows = len(offsets)
    for window_idx, offset in enumerate(offsets):
        log_progress(
            prefix="future windows",
            index=window_idx,
            total=total_windows,
            extra="candidate={}".format(params["name"]),
        )
        history = np.concatenate([train_values, future_values[:offset]])
        steps = min(HORIZON, len(future_values) - offset)
        if steps <= 0:
            break
        try:
            forecast = fit_and_forecast(history, steps, params)
        except Exception:
            forecast = np.repeat(history[-1], steps).astype(float)
        truth = future_values[offset : offset + steps]
        for horizon_idx in range(steps):
            rows.append(
                {
                    "origin_index": int(train_len + offset),
                    "target_index": int(train_len + offset + horizon_idx),
                    "horizon": int(horizon_idx + 1),
                    "prediction": float(forecast[horizon_idx]),
                    "truth": float(truth[horizon_idx]),
                }
            )
    return pd.DataFrame(rows)


def compute_metrics(prediction_df: pd.DataFrame) -> Dict[str, float]:
    pred = prediction_df["prediction"].to_numpy(dtype=float)
    truth = prediction_df["truth"].to_numpy(dtype=float)
    diff = pred - truth
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(diff) / np.clip(np.abs(truth), 1e-6, None)) * 100.0)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "num_predictions": int(len(prediction_df)),
    }


def main() -> None:
    train_values = load_target_series(TRAIN_CSV, TARGET_COLUMN)
    future_values = load_target_series(FUTURE_CSV, TARGET_COLUMN)
    train_split, val_split = split_train_val(train_values, VAL_RATIO)

    best_params = select_best_params(train_split, val_split)
    prediction_df = rolling_forecast(train_values, future_values, best_params)
    metrics = compute_metrics(prediction_df)
    metrics.update(
        {
            "train_csv": str(TRAIN_CSV),
            "future_csv": str(FUTURE_CSV),
            "target_column": TARGET_COLUMN,
            "best_params": best_params,
        }
    )

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    with OUTPUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("ETS benchmark finished.")
    print("best_params:", best_params)
    print("predictions:", str(OUTPUT_PREDICTIONS))
    print("metrics:", str(OUTPUT_METRICS))


if __name__ == "__main__":
    main()
