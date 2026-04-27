import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prediction_metrics


DEFAULT_DATASETS = ["AQShunyi", "Electricity", "METR-LA", "NASDAQ", "Weather"]
DEFAULT_BENCHMARKS = ["ARIMA", "ETS", "RNN", "TCN"]


def validate_choices(values: Sequence[str], allowed: Sequence[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    cleaned: List[str] = []
    for value in values:
        if value not in allowed_set:
            raise ValueError("Unsupported {} '{}'. Allowed values: {}".format(label, value, list(allowed)))
        cleaned.append(value)
    return cleaned


def resolve_prediction_path(predictions_dir: Path, dataset: str, benchmark: str) -> Path:
    path = predictions_dir / "{}_{}_predictions.csv".format(dataset, benchmark)
    if not path.exists():
        raise FileNotFoundError("Prediction CSV not found: {}".format(path))
    return path


def compute_row(dataset: str, benchmark: str, prediction_path: Path, seasonality: int) -> Dict[str, object]:
    pred, truth, truth_for_mase, _, num_rows = prediction_metrics.load_arrays(prediction_path)
    metrics = prediction_metrics.compute_metrics(pred, truth, truth_for_mase, seasonality=seasonality)
    row: Dict[str, object] = {
        "dataset": dataset,
        "benchmark": benchmark,
        "prediction_path": str(prediction_path),
        "num_rows": int(num_rows),
        "seasonality": int(seasonality),
    }
    row.update(metrics)
    return row


def save_long_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "dataset",
        "benchmark",
        "mae",
        "mape",
        "rmse",
        "mase",
        "h",
        "M",
        "S",
        "num_rows",
        "seasonality",
        "prediction_path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_wide_csv(rows: Sequence[Dict[str, object]], output_path: Path, datasets: Sequence[str], benchmarks: Sequence[str]) -> None:
    metric_names = ["mae", "mape", "rmse", "mase"]
    rows_by_key = {(row["dataset"], row["benchmark"]): row for row in rows}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["benchmark", "metric"] + list(datasets)
        writer.writerow(header)
        for benchmark in benchmarks:
            for metric_name in metric_names:
                line = [benchmark, metric_name]
                for dataset in datasets:
                    row = rows_by_key.get((dataset, benchmark))
                    value = "" if row is None else row.get(metric_name, "")
                    line.append(value)
                writer.writerow(line)


def save_json(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": list(rows)}, f, ensure_ascii=False, indent=2)


def print_table(rows: Sequence[Dict[str, object]], datasets: Sequence[str], benchmarks: Sequence[str]) -> None:
    metrics = ["mae", "mape", "rmse", "mase"]
    rows_by_key = {(row["dataset"], row["benchmark"]): row for row in rows}
    for benchmark in benchmarks:
        print("[summary] {}".format(benchmark))
        for metric_name in metrics:
            values = []
            for dataset in datasets:
                row = rows_by_key.get((dataset, benchmark))
                if row is None:
                    values.append("{}=NA".format(dataset))
                else:
                    values.append("{}={:.6f}".format(dataset, float(row[metric_name])))
            print("  {} | {}".format(metric_name, " | ".join(values)))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read benchmark prediction CSVs, run prediction_metrics, and export summary tables."
    )
    parser.add_argument(
        "--predictions-dir",
        default="outputs/benchmark_suite",
        help="Directory containing *_predictions.csv files from run_benchmark_suite.py",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to summarize. Default: all five datasets.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=DEFAULT_BENCHMARKS,
        help="Benchmarks to summarize. Default: ARIMA ETS RNN TCN",
    )
    parser.add_argument(
        "--seasonality",
        type=int,
        default=1,
        help="Seasonality passed to prediction_metrics for MASE. Default: 1",
    )
    parser.add_argument(
        "--long-output",
        default="outputs/benchmark_suite/prediction_metrics_long.csv",
        help="Path for the long-form metrics CSV.",
    )
    parser.add_argument(
        "--wide-output",
        default="outputs/benchmark_suite/prediction_metrics_wide.csv",
        help="Path for the wide-form metrics CSV.",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/benchmark_suite/prediction_metrics_summary.json",
        help="Path for the JSON summary.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    datasets = validate_choices(args.datasets, DEFAULT_DATASETS, label="dataset")
    benchmarks = validate_choices(args.benchmarks, DEFAULT_BENCHMARKS, label="benchmark")
    predictions_dir = PROJECT_ROOT / args.predictions_dir

    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for benchmark in benchmarks:
            prediction_path = resolve_prediction_path(predictions_dir, dataset, benchmark)
            row = compute_row(
                dataset=dataset,
                benchmark=benchmark,
                prediction_path=prediction_path,
                seasonality=int(args.seasonality),
            )
            rows.append(row)

    long_output = PROJECT_ROOT / args.long_output
    wide_output = PROJECT_ROOT / args.wide_output
    json_output = PROJECT_ROOT / args.json_output

    save_long_csv(rows, long_output)
    save_wide_csv(rows, wide_output, datasets=datasets, benchmarks=benchmarks)
    save_json(rows, json_output)
    print_table(rows, datasets=datasets, benchmarks=benchmarks)
    print("[summary] long_output={}".format(long_output))
    print("[summary] wide_output={}".format(wide_output))
    print("[summary] json_output={}".format(json_output))


if __name__ == "__main__":
    main()
