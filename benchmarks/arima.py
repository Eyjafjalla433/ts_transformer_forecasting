import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


TRAIN_CSV = Path("data/processed/NASDAQ_wide_train.csv")
FUTURE_CSV = Path("data/processed/NASDAQ_wide_future.csv")
TARGET_COLUMN = "Close"
VAL_RATIO = 0.2
HORIZON = 12
STRIDE = 12
OUTPUT_PREDICTIONS = Path("outputs/NASDAQ_ARIMA_predictions.csv")
OUTPUT_METRICS = Path("outputs/NASDAQ_ARIMA_metrics.json")
VERBOSE_PROGRESS = False
MAX_HISTORY = 1440
MAX_VALIDATION_WINDOWS = 48
ORDER_GRID: List[Tuple[int, int, int]] = [
    (0, 1, 0),
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (1, 1, 2),
    (2, 1, 1),
]


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
    message = "[ARIMA] {} {}/{}".format(prefix, index + 1, total)
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
        raise ValueError("Series is too short for ARIMA validation.")
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
        raise ValueError("ARIMA forecast produced non-finite values.")

    history_scale = float(np.max(np.abs(history))) if len(history) > 0 else 0.0
    allowed_scale = max(50.0, history_scale * 20.0)
    if float(np.max(np.abs(forecast))) > allowed_scale:
        raise ValueError(
            "ARIMA forecast appears unstable: max_abs_forecast={} allowed_scale={}.".format(
                float(np.max(np.abs(forecast))),
                allowed_scale,
            )
        )
    return forecast


def fit_and_forecast(history: np.ndarray, steps: int, order: Tuple[int, int, int]) -> np.ndarray:
    if len(history) == 0:
        raise ValueError("History cannot be empty.")
    history = trim_history(history)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(history, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    forecast = np.asarray(fitted.forecast(steps=steps), dtype=float)
    if forecast.shape[0] != steps:
        raise ValueError("ARIMA forecast returned {} steps, expected {}.".format(forecast.shape[0], steps))
    return validate_forecast(forecast, history)


def naive_last_value_forecast(history: np.ndarray, steps: int) -> np.ndarray:
    if len(history) == 0:
        raise ValueError("History cannot be empty for naive fallback.")
    return np.repeat(float(history[-1]), steps).astype(float)


def safe_forecast(history: np.ndarray, steps: int, order: Tuple[int, int, int]) -> Tuple[np.ndarray, bool, str]:
    try:
        return fit_and_forecast(history, steps, order), False, ""
    except Exception as exc:
        return naive_last_value_forecast(history, steps), True, str(exc)


def score_order(train_values: np.ndarray, val_values: np.ndarray, order: Tuple[int, int, int]) -> float:
    preds: List[float] = []
    truths: List[float] = []
    window_starts = select_window_starts(len(val_values))
    total_windows = len(window_starts)
    for window_idx, start in enumerate(window_starts):
        log_progress(
            prefix="validation windows",
            index=window_idx,
            total=total_windows,
            extra="order={}".format(order),
        )
        history = np.concatenate([train_values, val_values[:start]])
        steps = min(HORIZON, len(val_values) - start)
        if steps <= 0:
            break
        try:
            forecast = fit_and_forecast(history, steps, order)
        except Exception:
            return float("inf")
        preds.extend(forecast.tolist())
        truths.extend(val_values[start : start + steps].tolist())
    if not preds:
        return float("inf")
    diff = np.asarray(preds) - np.asarray(truths)
    return float(np.mean(diff * diff))


def select_best_order(train_values: np.ndarray, val_values: np.ndarray) -> Dict:
    best_order: Tuple[int, int, int] = (0, 1, 0)
    best_mse = float("inf")
    total_orders = len(ORDER_GRID)
    for order_idx, order in enumerate(ORDER_GRID):
        if VERBOSE_PROGRESS:
            print(
                "[ARIMA] candidate {}/{} | order={}".format(order_idx + 1, total_orders, order),
                flush=True,
            )
        mse = score_order(train_values, val_values, order)
        if VERBOSE_PROGRESS:
            print(
                "[ARIMA] candidate result | order={} | val_mse={:.6f}".format(order, mse),
                flush=True,
            )
        if mse < best_mse:
            best_mse = mse
            best_order = order
    return {"order": list(best_order), "val_mse": best_mse}


def rolling_forecast(train_values: np.ndarray, future_values: np.ndarray, order: Tuple[int, int, int]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    train_len = len(train_values)
    offsets = list(range(0, len(future_values), STRIDE))
    total_windows = len(offsets)
    fallback_windows = 0
    fallback_reasons: Dict[str, int] = {}
    for window_idx, offset in enumerate(offsets):
        log_progress(
            prefix="future windows",
            index=window_idx,
            total=total_windows,
            extra="order={}".format(order),
        )
        history = np.concatenate([train_values, future_values[:offset]])
        steps = min(HORIZON, len(future_values) - offset)
        if steps <= 0:
            break
        forecast, used_fallback, fallback_reason = safe_forecast(history, steps, order)
        if used_fallback:
            fallback_windows += 1
            fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1
        truth = future_values[offset : offset + steps]
        for horizon_idx in range(steps):
            rows.append(
                {
                    "origin_index": int(train_len + offset),
                    "target_index": int(train_len + offset + horizon_idx),
                    "horizon": int(horizon_idx + 1),
                    "prediction": float(forecast[horizon_idx]),
                    "truth": float(truth[horizon_idx]),
                    "used_naive_fallback": int(used_fallback),
                }
            )
    prediction_df = pd.DataFrame(rows)
    prediction_df.attrs["fallback_windows"] = int(fallback_windows)
    prediction_df.attrs["total_windows"] = int(total_windows)
    prediction_df.attrs["fallback_reasons"] = dict(sorted(fallback_reasons.items()))
    return prediction_df


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

    best = select_best_order(train_split, val_split)
    order = tuple(best["order"])
    prediction_df = rolling_forecast(train_values, future_values, order)
    metrics = compute_metrics(prediction_df)
    metrics.update(
        {
            "train_csv": str(TRAIN_CSV),
            "future_csv": str(FUTURE_CSV),
            "target_column": TARGET_COLUMN,
            "best_order": best,
            "fallback_windows": int(prediction_df.attrs.get("fallback_windows", 0)),
            "total_windows": int(prediction_df.attrs.get("total_windows", 0)),
            "fallback_ratio": float(
                prediction_df.attrs.get("fallback_windows", 0) / max(1, prediction_df.attrs.get("total_windows", 0))
            ),
            "fallback_reasons": prediction_df.attrs.get("fallback_reasons", {}),
        }
    )

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    with OUTPUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("ARIMA benchmark finished.")
    print("best_order:", best)
    print("predictions:", str(OUTPUT_PREDICTIONS))
    print("metrics:", str(OUTPUT_METRICS))


if __name__ == "__main__":
    main()
