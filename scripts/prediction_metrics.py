import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def resolve_prediction_path(prediction_path: str) -> Path:
    path = Path(prediction_path)
    if path.exists():
        return path

    outputs_dir = path.parent if path.parent != Path("") else Path("outputs")
    candidates = sorted(outputs_dir.glob("*_Prediction.csv"))
    if candidates:
        candidate_text = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            "Prediction CSV not found: {}. Available prediction files: {}".format(path, candidate_text)
        )

    raise FileNotFoundError("Prediction CSV not found: {}".format(path))


def collect_value_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    pred_columns = sorted(col for col in df.columns if col.startswith("pred_"))
    truth_columns = sorted(col for col in df.columns if col.startswith("truth_"))
    if not pred_columns and not truth_columns:
        if "prediction" in df.columns and "truth" in df.columns:
            return ["prediction"], ["truth"]
    if not pred_columns or not truth_columns:
        raise ValueError(
            "CSV must contain columns like pred_0/truth_0 or prediction/truth. Got columns: {}".format(list(df.columns))
        )

    pred_suffixes = {col[len("pred_") :]: col for col in pred_columns}
    truth_suffixes = {col[len("truth_") :]: col for col in truth_columns}
    shared_suffixes = sorted(set(pred_suffixes) & set(truth_suffixes), key=_sort_key)
    if not shared_suffixes:
        raise ValueError(
            "No matching pred_/truth_ column pairs were found. pred columns={}, truth columns={}".format(
                pred_columns, truth_columns
            )
        )

    return [pred_suffixes[suffix] for suffix in shared_suffixes], [truth_suffixes[suffix] for suffix in shared_suffixes]


def _sort_key(value: str) -> Tuple[int, str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def load_arrays(prediction_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[str], int]:
    df = pd.read_csv(prediction_path)
    pred_columns, truth_columns = collect_value_columns(df)

    useful_columns = list(pred_columns) + list(truth_columns)
    if "target_index" in df.columns:
        useful_columns = ["target_index"] + useful_columns

    clean_df = df[useful_columns].copy()
    if "target_index" in clean_df.columns:
        clean_df["target_index"] = pd.to_numeric(clean_df["target_index"], errors="coerce")

    for column in pred_columns + truth_columns:
        clean_df[column] = pd.to_numeric(clean_df[column], errors="coerce")

    clean_df = clean_df.dropna(subset=pred_columns + truth_columns)
    if "target_index" in clean_df.columns:
        clean_df = clean_df.sort_values("target_index").reset_index(drop=True)

    if clean_df.empty:
        raise ValueError("Prediction CSV has no valid rows after cleaning: {}".format(prediction_path))

    pred = clean_df[pred_columns].to_numpy(dtype=float)
    truth = clean_df[truth_columns].to_numpy(dtype=float)

    if "target_index" in clean_df.columns:
        mase_truth_df = (
            clean_df[["target_index"] + truth_columns]
            .drop_duplicates(subset=["target_index"], keep="first")
            .sort_values("target_index")
            .reset_index(drop=True)
        )
        truth_for_mase = mase_truth_df[truth_columns].to_numpy(dtype=float)
    else:
        truth_for_mase = truth

    return pred, truth, truth_for_mase, pred_columns, len(clean_df)


def compute_metrics(pred: np.ndarray, truth: np.ndarray, truth_for_mase: np.ndarray, seasonality: int) -> dict:
    if pred.shape != truth.shape:
        raise ValueError("Prediction shape {} does not match truth shape {}.".format(pred.shape, truth.shape))
    if pred.size == 0:
        raise ValueError("Prediction CSV is empty after filtering.")
    if seasonality < 1:
        raise ValueError("seasonality must be >= 1, but got {}.".format(seasonality))

    pred_flat = pred.reshape(-1)
    truth_flat = truth.reshape(-1)
    diff = pred_flat - truth_flat
    h = diff.shape[0]

    mae = float(np.sum(np.abs(diff)) / h)
    rmse = float(np.sqrt(np.sum(diff * diff) / h))
    mape = float(np.sum(np.abs(diff) / np.clip(np.abs(truth_flat), 1e-6, None)) / h * 100.0)

    if truth_for_mase.shape[0] <= seasonality:
        raise ValueError(
            "MASE requires more than seasonality={} reference rows, but got {}.".format(
                seasonality, truth_for_mase.shape[0]
            )
        )

    seasonal_diff = truth_for_mase[seasonality:] - truth_for_mase[:-seasonality]
    naive_mae = float(np.mean(np.abs(seasonal_diff)))
    if naive_mae <= 1e-12:
        raise ValueError("MASE is undefined because the naive forecast denominator is 0.")

    mase = float(mae / naive_mae)
    return {"mae": mae, "mape": mape, "rmse": rmse, "mase": mase, "h": int(h), "M": int(truth_for_mase.shape[0]), "S": int(seasonality)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read a prediction CSV and print MAE, MAPE, RMSE, and MASE for pred/truth columns."
    )
    parser.add_argument(
        "--prediction-path",
        default="outputs/Weather_Prediction.csv",
        help="Path to the prediction CSV. Default: outputs/Weather_Prediction.csv",
    )
    parser.add_argument(
        "--seasonality",
        type=int,
        default=1,
        help="Seasonality S used in the MASE denominator. Default: 1",
    )
    args = parser.parse_args()

    prediction_path = resolve_prediction_path(args.prediction_path)
    pred, truth, truth_for_mase, pred_columns, num_rows = load_arrays(prediction_path)
    metrics = compute_metrics(pred, truth, truth_for_mase, seasonality=args.seasonality)

    print("prediction_path:", prediction_path)
    print("num_rows:", num_rows)
    print("num_pred_columns:", len(pred_columns))
    print("forecast_h:", metrics["h"])
    print("mase_reference_length:", metrics["M"])
    print("seasonality:", metrics["S"])
    print("mae:", "{:.6f}".format(metrics["mae"]))
    print("mape(%):", "{:.6f}".format(metrics["mape"]))
    print("rmse:", "{:.6f}".format(metrics["rmse"]))
    print("mase:", "{:.6f}".format(metrics["mase"]))


if __name__ == "__main__":
    main()
