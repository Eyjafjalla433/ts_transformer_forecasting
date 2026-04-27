import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def infer_value_columns(
    df: pd.DataFrame,
    pred_column: Optional[str],
    truth_column: Optional[str],
) -> Tuple[str, str]:
    if pred_column and truth_column:
        missing = [col for col in (pred_column, truth_column) if col not in df.columns]
        if missing:
            raise ValueError("Columns not found in CSV: {}".format(", ".join(missing)))
        return pred_column, truth_column

    pred_candidates = sorted(col for col in df.columns if col.startswith("pred"))
    truth_candidates = sorted(col for col in df.columns if col.startswith("truth"))
    if not pred_candidates or not truth_candidates:
        raise ValueError(
            "Prediction CSV must contain prediction and truth columns such as pred_0 and truth_0. "
            "Got columns: {}".format(list(df.columns))
        )

    pred_column = pred_column or pred_candidates[0]
    truth_column = truth_column or truth_candidates[0]
    if pred_column not in df.columns or truth_column not in df.columns:
        raise ValueError(
            "Unable to resolve value columns. pred_column={}, truth_column={}".format(
                pred_column, truth_column
            )
        )
    return pred_column, truth_column


def prepare_series(
    prediction_path: str,
    target_column: str,
    pred_column: Optional[str],
    truth_column: Optional[str],
) -> Tuple[pd.DataFrame, str, str]:
    path = Path(prediction_path)
    if not path.exists():
        raise FileNotFoundError("Prediction CSV not found: {}".format(path))

    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(
            "Prediction CSV must contain target index column '{}'. Got: {}".format(
                target_column, list(df.columns)
            )
        )

    pred_column, truth_column = infer_value_columns(df, pred_column, truth_column)
    plot_df = df[[target_column, pred_column, truth_column]].copy()
    plot_df = plot_df.dropna(subset=[target_column, pred_column, truth_column])
    plot_df[target_column] = pd.to_numeric(plot_df[target_column], errors="coerce")
    plot_df[pred_column] = pd.to_numeric(plot_df[pred_column], errors="coerce")
    plot_df[truth_column] = pd.to_numeric(plot_df[truth_column], errors="coerce")
    plot_df = plot_df.dropna(subset=[target_column, pred_column, truth_column])

    # Average duplicated target indices so the script still works when multiple
    # predictions map to the same timestamp.
    plot_df = (
        plot_df.groupby(target_column, as_index=False)[[pred_column, truth_column]]
        .mean()
        .sort_values(target_column)
        .reset_index(drop=True)
    )
    if plot_df.empty:
        raise ValueError("Prediction CSV has no valid rows after cleaning: {}".format(path))

    return plot_df, pred_column, truth_column


def select_best_window(
    df: pd.DataFrame,
    target_column: str,
    pred_column: str,
    truth_column: str,
    window_size: int,
) -> pd.DataFrame:
    if window_size <= 0:
        raise ValueError("window_size must be positive, got {}".format(window_size))
    if len(df) < window_size:
        raise ValueError(
            "Not enough valid points to select {} continuous points. Only {} points found.".format(
                window_size, len(df)
            )
        )

    best_start = None
    best_score = None

    segment_start = 0
    target_values = df[target_column].to_numpy()
    for idx in range(1, len(df) + 1):
        is_break = idx == len(df) or target_values[idx] - target_values[idx - 1] != 1
        if not is_break:
            continue

        segment = df.iloc[segment_start:idx].copy().reset_index(drop=True)
        if len(segment) >= window_size:
            segment["abs_error"] = (segment[pred_column] - segment[truth_column]).abs()
            rolling_error = segment["abs_error"].rolling(window=window_size).mean()
            segment_best_end = rolling_error.idxmin()
            segment_best_score = float(rolling_error.iloc[segment_best_end])
            segment_best_start = segment_start + int(segment_best_end) - window_size + 1

            if best_score is None or segment_best_score < best_score:
                best_score = segment_best_score
                best_start = segment_best_start

        segment_start = idx

    if best_start is None or best_score is None:
        raise ValueError(
            "No segment contains {} continuous target points under '{}'.".format(
                window_size, target_column
            )
        )

    best_window = df.iloc[best_start : best_start + window_size].copy().reset_index(drop=True)
    best_window["abs_error"] = (best_window[pred_column] - best_window[truth_column]).abs()
    return best_window


def plot_best_window(
    window_df: pd.DataFrame,
    target_column: str,
    pred_column: str,
    truth_column: str,
    output_path: str,
    title: Optional[str],
    dpi: int,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mae = float(window_df["abs_error"].mean())
    start_idx = int(window_df[target_column].iloc[0])
    end_idx = int(window_df[target_column].iloc[-1])
    plot_index = range(1, len(window_df) + 1)

    plt.figure(figsize=(14, 6))
    plt.plot(
        plot_index,
        window_df[truth_column],
        label="Ground Truth",
        linewidth=2.0,
        color="#1f77b4",
    )
    plt.plot(
        plot_index,
        window_df[pred_column],
        label="Prediction",
        linewidth=1.8,
        color="#ff7f0e",
        alpha=0.9,
    )
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(
        title
        or "Result Comparison"
    )
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=dpi)
    plt.close()


def build_output_path(prediction_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    prediction_file = Path(prediction_path)
    stem = prediction_file.stem
    return str(prediction_file.with_name("{}_best_200_curve.png".format(stem)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot prediction and ground-truth curves for the best continuous window."
    )
    parser.add_argument(
        "--prediction-path",
        default="outputs/prediction.csv",
        help="Path to prediction CSV generated by inference.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to save the visualization. Defaults to a PNG next to the CSV.",
    )
    parser.add_argument(
        "--target-column",
        default="target_index",
        help="Column name that represents the time or sample index.",
    )
    parser.add_argument(
        "--pred-column",
        default=None,
        help="Prediction column to plot. Defaults to the first column starting with 'pred'.",
    )
    parser.add_argument(
        "--truth-column",
        default=None,
        help="Ground-truth column to plot. Defaults to the first column starting with 'truth'.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Number of continuous points to plot.",
    )
    parser.add_argument(
        "--title",
        default="Result Comparison",
        help="Optional chart title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output image DPI.",
    )
    args = parser.parse_args()

    series_df, pred_column, truth_column = prepare_series(
        prediction_path=args.prediction_path,
        target_column=args.target_column,
        pred_column=args.pred_column,
        truth_column=args.truth_column,
    )
    best_window = select_best_window(
        df=series_df,
        target_column=args.target_column,
        pred_column=pred_column,
        truth_column=truth_column,
        window_size=args.window_size,
    )
    output_path = build_output_path(args.prediction_path, args.output_path)
    plot_best_window(
        window_df=best_window,
        target_column=args.target_column,
        pred_column=pred_column,
        truth_column=truth_column,
        output_path=output_path,
        title=args.title,
        dpi=args.dpi,
    )

    mae = float(best_window["abs_error"].mean())
    print("Saved prediction curve:", output_path)
    print("Read prediction CSV:", args.prediction_path)
    print("Plot columns: pred='{}', truth='{}'".format(pred_column, truth_column))
    print(
        "Selected range: {} ~ {} ({} points)".format(
            int(best_window[args.target_column].iloc[0]),
            int(best_window[args.target_column].iloc[-1]),
            len(best_window),
        )
    )
    print("Window MAE: {:.6f}".format(mae))


if __name__ == "__main__":
    main()
