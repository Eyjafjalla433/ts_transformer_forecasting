import argparse
import gc
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prediction_metrics


DEFAULT_DATASETS = [
    "NASDAQ",
    "AQShunyi",
    "Electricity",
    "METR-LA",
    "Weather",
]

BENCHMARK_MODULES = {
    "arima": "benchmarks.arima",
    "ets": "benchmarks.ets",
    "rnn": "benchmarks.rnn",
    "tcn": "benchmarks.tcn",
}

TRAINABLE_BENCHMARKS = {"rnn", "tcn"}


def parse_override_pairs(values: Sequence[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("Override '{}' must use DATASET=VALUE format.".format(raw))
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError("Override '{}' must use non-empty DATASET=VALUE format.".format(raw))
        parsed[key] = value
    return parsed


def validate_choices(values: Sequence[str], allowed: Sequence[str], label: str) -> List[str]:
    allowed_set = set(allowed)
    cleaned: List[str] = []
    for value in values:
        if value not in allowed_set:
            raise ValueError("Unsupported {} '{}'. Allowed values: {}".format(label, value, list(allowed)))
        cleaned.append(value)
    return cleaned


def resolve_dataset_files(dataset: str) -> Tuple[Path, Path]:
    train_csv = PROJECT_ROOT / "data" / "processed" / "{}_wide_train.csv".format(dataset)
    future_csv = PROJECT_ROOT / "data" / "processed" / "{}_wide_future.csv".format(dataset)
    if not train_csv.exists():
        raise FileNotFoundError("Train CSV not found for dataset '{}': {}".format(dataset, train_csv))
    if not future_csv.exists():
        raise FileNotFoundError("Future CSV not found for dataset '{}': {}".format(dataset, future_csv))
    return train_csv, future_csv


def resolve_target_column(train_csv: Path, override: str = "") -> str:
    if override:
        return override

    with train_csv.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().strip()
    if not header:
        raise ValueError("CSV header is empty: {}".format(train_csv))

    columns = [column.strip() for column in header.split(",")]
    if not columns:
        raise ValueError("Could not parse columns from header: {}".format(train_csv))
    return columns[-1]


def load_benchmark_module(name: str):
    return importlib.import_module(BENCHMARK_MODULES[name])


def configure_module(
    module,
    dataset: str,
    benchmark: str,
    train_csv: Path,
    future_csv: Path,
    target_column: str,
    output_dir: Path,
    checkpoint_dir: Path,
    device_override: str,
) -> Dict[str, Path]:
    prediction_path = output_dir / "{}_{}_predictions.csv".format(dataset, benchmark.upper())
    benchmark_metrics_path = output_dir / "{}_{}_benchmark_metrics.json".format(dataset, benchmark.upper())

    module.TRAIN_CSV = train_csv
    module.FUTURE_CSV = future_csv
    module.TARGET_COLUMN = target_column
    module.OUTPUT_PREDICTIONS = prediction_path
    module.OUTPUT_METRICS = benchmark_metrics_path

    checkpoint_path = None
    train_log_path = None
    if benchmark in TRAINABLE_BENCHMARKS:
        checkpoint_path = checkpoint_dir / "{}_{}.pt".format(dataset, benchmark.upper())
        train_log_path = output_dir / "{}_{}_train_log.csv".format(dataset, benchmark.upper())
        module.CHECKPOINT_PATH = checkpoint_path
        module.OUTPUT_TRAIN_LOG = train_log_path
        if device_override:
            module.DEVICE = device_override

    return {
        "prediction_path": prediction_path,
        "benchmark_metrics_path": benchmark_metrics_path,
        "checkpoint_path": checkpoint_path,
        "train_log_path": train_log_path,
    }


def compute_prediction_metrics(prediction_path: Path, seasonality: int) -> Dict[str, float]:
    pred, truth, truth_for_mase, _, _ = prediction_metrics.load_arrays(prediction_path)
    return prediction_metrics.compute_metrics(pred, truth, truth_for_mase, seasonality=seasonality)


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def save_summary_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "dataset",
        "benchmark",
        "status",
        "target_column",
        "seasonality",
        "prediction_path",
        "benchmark_metrics_path",
        "prediction_metrics_path",
        "checkpoint_path",
        "train_log_path",
        "mae",
        "mape",
        "rmse",
        "mase",
        "h",
        "M",
        "S",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                text = "" if value is None else str(value)
                if any(token in text for token in [",", "\"", "\n"]):
                    text = "\"" + text.replace("\"", "\"\"") + "\""
                values.append(text)
            f.write(",".join(values) + "\n")


def maybe_release_torch_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def run_suite(
    datasets: Sequence[str],
    benchmarks: Sequence[str],
    seasonality: int,
    output_dir: Path,
    checkpoint_dir: Path,
    target_overrides: Dict[str, str],
    device_override: str,
    dry_run: bool,
    fail_fast: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total_runs = len(datasets) * len(benchmarks)
    run_index = 0

    for dataset in datasets:
        train_csv, future_csv = resolve_dataset_files(dataset)
        target_column = resolve_target_column(train_csv, override=target_overrides.get(dataset, ""))
        for benchmark in benchmarks:
            run_index += 1
            print(
                "[suite] run {}/{} | dataset={} | benchmark={} | target={}".format(
                    run_index, total_runs, dataset, benchmark, target_column
                ),
                flush=True,
            )
            configured = {
                "prediction_path": output_dir / "{}_{}_predictions.csv".format(dataset, benchmark.upper()),
                "benchmark_metrics_path": output_dir / "{}_{}_benchmark_metrics.json".format(dataset, benchmark.upper()),
                "checkpoint_path": checkpoint_dir / "{}_{}.pt".format(dataset, benchmark.upper()) if benchmark in TRAINABLE_BENCHMARKS else None,
                "train_log_path": output_dir / "{}_{}_train_log.csv".format(dataset, benchmark.upper()) if benchmark in TRAINABLE_BENCHMARKS else None,
            }
            prediction_metrics_path = output_dir / "{}_{}_prediction_metrics.json".format(dataset, benchmark.upper())

            row: Dict[str, object] = {
                "dataset": dataset,
                "benchmark": benchmark,
                "status": "pending",
                "target_column": target_column,
                "seasonality": seasonality,
                "prediction_path": configured["prediction_path"],
                "benchmark_metrics_path": configured["benchmark_metrics_path"],
                "prediction_metrics_path": prediction_metrics_path,
                "checkpoint_path": configured["checkpoint_path"],
                "train_log_path": configured["train_log_path"],
            }

            if dry_run:
                row["status"] = "dry_run"
                rows.append(row)
                print("  train_csv={}".format(train_csv), flush=True)
                print("  future_csv={}".format(future_csv), flush=True)
                print("  prediction_path={}".format(configured["prediction_path"]), flush=True)
                continue

            try:
                module = load_benchmark_module(benchmark)
                configured = configure_module(
                    module=module,
                    dataset=dataset,
                    benchmark=benchmark,
                    train_csv=train_csv,
                    future_csv=future_csv,
                    target_column=target_column,
                    output_dir=output_dir,
                    checkpoint_dir=checkpoint_dir,
                    device_override=device_override,
                )
                row["prediction_path"] = configured["prediction_path"]
                row["benchmark_metrics_path"] = configured["benchmark_metrics_path"]
                row["checkpoint_path"] = configured["checkpoint_path"]
                row["train_log_path"] = configured["train_log_path"]

                module.main()
                pred_metrics = compute_prediction_metrics(configured["prediction_path"], seasonality=seasonality)
                pred_metrics_payload = {
                    "dataset": dataset,
                    "benchmark": benchmark,
                    "target_column": target_column,
                    "prediction_path": str(configured["prediction_path"]),
                    "seasonality": seasonality,
                }
                pred_metrics_payload.update(pred_metrics)
                save_json(pred_metrics_payload, prediction_metrics_path)

                row.update(pred_metrics)
                row["status"] = "ok"
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = "{}: {}".format(type(exc).__name__, exc)
                print("[suite] failed dataset={} benchmark={}".format(dataset, benchmark), flush=True)
                print(traceback.format_exc(), flush=True)
                if fail_fast:
                    rows.append(row)
                    raise
            finally:
                maybe_release_torch_memory()

            rows.append(row)

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ARIMA/ETS/RNN/TCN benchmarks across multiple processed datasets and summarize prediction_metrics."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Datasets to run. Default: all five processed datasets.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=list(BENCHMARK_MODULES.keys()),
        help="Benchmarks to run. Default: arima ets rnn tcn",
    )
    parser.add_argument(
        "--seasonality",
        type=int,
        default=1,
        help="Seasonality passed to prediction_metrics for MASE. Default: 1",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/benchmark_suite",
        help="Directory for prediction files, metric JSONs, and summary tables.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints/benchmark_suite",
        help="Directory for RNN/TCN checkpoints.",
    )
    parser.add_argument(
        "--target-column-override",
        nargs="*",
        default=[],
        help="Optional dataset-specific target overrides in DATASET=TARGET format.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Optional device override for trainable benchmarks, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved runs without executing benchmarks.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one benchmark run fails.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    datasets = validate_choices(args.datasets, DEFAULT_DATASETS, label="dataset")
    benchmarks = validate_choices(args.benchmarks, list(BENCHMARK_MODULES.keys()), label="benchmark")
    target_overrides = parse_override_pairs(args.target_column_override)

    output_dir = PROJECT_ROOT / args.output_dir
    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    rows = run_suite(
        datasets=datasets,
        benchmarks=benchmarks,
        seasonality=args.seasonality,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        target_overrides=target_overrides,
        device_override=args.device,
        dry_run=bool(args.dry_run),
        fail_fast=bool(args.fail_fast),
    )

    summary_json_path = output_dir / "benchmark_suite_summary.json"
    summary_csv_path = output_dir / "benchmark_suite_summary.csv"
    save_json({"rows": rows}, summary_json_path)
    save_summary_csv(rows, summary_csv_path)

    ok_count = sum(1 for row in rows if row.get("status") == "ok")
    failed_count = sum(1 for row in rows if row.get("status") == "failed")
    print("[suite] completed runs={} ok={} failed={}".format(len(rows), ok_count, failed_count))
    print("[suite] summary_json={}".format(summary_json_path), flush=True)
    print("[suite] summary_csv={}".format(summary_csv_path), flush=True)


if __name__ == "__main__":
    main()
