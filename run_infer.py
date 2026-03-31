import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from engine.infer import (
    autoregressive_forecast,
    build_model_from_config,
    denormalize_tensor,
    export_predictions_to_csv,
    load_input_window_from_csv,
    load_model_weights,
    load_normalization_stats,
    prepare_source_tensor,
    resolve_device,
)
from utils.config import load_config


def load_wide_series_from_csv(path: str, has_header: bool = True) -> torch.Tensor:
    """Load full wide table [T, F] from CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError("Wide CSV not found: {}".format(csv_path))

    rows: List[List[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])

    if not rows:
        raise ValueError("Wide CSV is empty: {}".format(csv_path))
    return torch.tensor(rows, dtype=torch.float32)


def resolve_column_slices(
    series_width: int,
    src_dim: int,
    tgt_dim: int,
    out_dim: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Match run_train.py column policy for consistent train/infer behavior."""
    if src_dim <= 0 or tgt_dim <= 0 or out_dim <= 0:
        raise ValueError("src_dim, tgt_dim, out_dim must all be positive.")
    if series_width < src_dim:
        raise ValueError("Series has {} columns but src_dim is {}.".format(series_width, src_dim))
    if series_width < max(tgt_dim, out_dim):
        raise ValueError(
            "Series has {} columns but target/output dimensions require at least {} columns."
            .format(series_width, max(tgt_dim, out_dim))
        )

    src_cols = list(range(src_dim))
    tgt_cols = list(range(series_width - tgt_dim, series_width))
    out_cols = list(range(series_width - out_dim, series_width))
    return src_cols, tgt_cols, out_cols


def compute_metrics(pred: torch.Tensor, truth: torch.Tensor) -> Dict[str, float]:
    """Compute mse/rmse/mae/mape on [N, out_dim] tensors."""
    if pred.numel() == 0 or truth.numel() == 0:
        raise ValueError("Prediction or truth is empty, cannot compute metrics.")
    if pred.shape != truth.shape:
        raise ValueError("Prediction shape {} != truth shape {}.".format(tuple(pred.shape), tuple(truth.shape)))

    diff = pred - truth
    mse = torch.mean(diff * diff).item()
    mae = torch.mean(torch.abs(diff)).item()
    rmse = mse ** 0.5
    denom = truth.abs().clamp_min(1e-6)
    mape = torch.mean(torch.abs(diff) / denom).item() * 100.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}


def export_offline_predictions(
    records: Sequence[Dict[str, float]],
    output_path: str,
) -> None:
    """Export rolling-eval predictions with origin/horizon/target index metadata."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        headers = ["origin_index", "target_index", "horizon"]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return

    headers = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)


def export_metrics(metrics: Dict[str, float], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def run_offline_eval_from_future(
    model: torch.nn.Module,
    device: torch.device,
    model_cfg: Dict,
    data_cfg: Dict,
    infer_cfg: Dict,
    input_mean: Optional[torch.Tensor],
    input_std: Optional[torch.Tensor],
    target_mean: Optional[torch.Tensor],
    target_std: Optional[torch.Tensor],
) -> None:
    """Run rolling offline evaluation over future_wide and print metrics."""
    input_length = int(data_cfg["input_length"])
    pred_length = int(data_cfg["pred_length"])
    src_dim = int(model_cfg["src_dim"])
    tgt_dim = int(model_cfg["tgt_dim"])
    out_dim = int(model_cfg["out_dim"])
    stride = int(infer_cfg.get("eval_stride", pred_length))
    if stride <= 0:
        raise ValueError("infer.eval_stride must be positive.")

    future_path = infer_cfg["future_path"]
    future_has_header = bool(infer_cfg.get("future_has_header", True))
    future_series = load_wide_series_from_csv(path=future_path, has_header=future_has_header)

    context_path = infer_cfg.get("context_path")
    context_has_header = bool(infer_cfg.get("context_has_header", True))
    if context_path:
        context_series = load_wide_series_from_csv(path=context_path, has_header=context_has_header)
        full_series = torch.cat([context_series, future_series], dim=0)
        future_start = int(context_series.size(0))
    else:
        full_series = future_series
        future_start = 0

    if int(full_series.size(0)) < input_length + 1:
        raise ValueError(
            "Need at least input_length + 1 rows in evaluation source, got {}."
            .format(int(full_series.size(0)))
        )

    src_cols, _, out_cols = resolve_column_slices(
        series_width=int(full_series.size(1)),
        src_dim=src_dim,
        tgt_dim=tgt_dim,
        out_dim=out_dim,
    )

    eval_start = max(future_start, input_length)
    eval_end = int(full_series.size(0))
    if eval_start >= eval_end:
        raise ValueError(
            "No evaluation anchors available. eval_start={} eval_end={}."
            .format(eval_start, eval_end)
        )

    if target_mean is not None:
        target_mean = target_mean.to(device)
    if target_std is not None:
        target_std = target_std.to(device)

    pred_chunks: List[torch.Tensor] = []
    truth_chunks: List[torch.Tensor] = []
    records: List[Dict[str, float]] = []

    for origin in range(eval_start, eval_end, stride):
        src_window = full_series[origin - input_length : origin, src_cols]
        src, src_mask = prepare_source_tensor(
            src_window=src_window,
            device=device,
            input_mean=input_mean,
            input_std=input_std,
        )

        pred_norm = autoregressive_forecast(
            model=model,
            src=src,
            src_mask=src_mask,
            pred_length=pred_length,
            tgt_dim=tgt_dim,
            out_dim=out_dim,
            start_token_mode=infer_cfg.get("start_token_mode", "zeros"),
        )
        pred = denormalize_tensor(pred_norm, target_mean, target_std)[0].detach().cpu()

        available = min(pred_length, eval_end - origin)
        truth = full_series[origin : origin + available, out_cols].detach().cpu()
        pred = pred[:available]

        pred_chunks.append(pred)
        truth_chunks.append(truth)

        for h in range(available):
            row: Dict[str, float] = {
                "origin_index": int(origin),
                "target_index": int(origin + h),
                "horizon": int(h + 1),
            }
            for j in range(out_dim):
                row["pred_{}".format(j)] = float(pred[h, j].item())
                row["truth_{}".format(j)] = float(truth[h, j].item())
            records.append(row)

    pred_all = torch.cat(pred_chunks, dim=0)
    truth_all = torch.cat(truth_chunks, dim=0)
    metrics = compute_metrics(pred_all, truth_all)
    metrics["num_eval_points"] = int(pred_all.shape[0])
    metrics["out_dim"] = int(out_dim)
    metrics["eval_stride"] = int(stride)
    metrics["future_rows"] = int(future_series.size(0))
    metrics["has_context"] = bool(context_path)

    pred_output = infer_cfg.get("output_path", "outputs/predictions_offline.csv")
    metrics_output = infer_cfg.get("metrics_output_path", "outputs/offline_metrics.json")
    export_offline_predictions(records, pred_output)
    export_metrics(metrics, metrics_output)

    print("Offline evaluation completed.")
    print("device:", str(device))
    print("future_path:", future_path)
    print("context_path:", context_path if context_path else "<none>")
    print("future_rows:", int(future_series.size(0)))
    print("eval_stride:", stride)
    print("num_eval_points:", int(pred_all.shape[0]))
    print("mse:", metrics["mse"])
    print("rmse:", metrics["rmse"])
    print("mae:", metrics["mae"])
    print("mape(%):", metrics["mape"])
    print("predictions_output:", pred_output)
    print("metrics_output:", metrics_output)


def main():
    """Run single-window inference or offline rolling evaluation from config."""
    cfg = load_config("configs/default.yaml")
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    infer_cfg = cfg.get("infer", {})

    device = resolve_device(infer_cfg.get("device", "cpu"))
    model = build_model_from_config(model_cfg, device)
    load_model_weights(model, infer_cfg["checkpoint_path"], device)

    stats = load_normalization_stats(infer_cfg.get("normalization_stats_path"))
    input_mean = stats["input_mean"] if stats else None
    input_std = stats["input_std"] if stats else None
    target_mean = stats["target_mean"] if stats else None
    target_std = stats["target_std"] if stats else None

    if infer_cfg.get("future_path"):
        run_offline_eval_from_future(
            model=model,
            device=device,
            model_cfg=model_cfg,
            data_cfg=data_cfg,
            infer_cfg=infer_cfg,
            input_mean=input_mean,
            input_std=input_std,
            target_mean=target_mean,
            target_std=target_std,
        )
        return

    src_window = load_input_window_from_csv(
        input_path=infer_cfg["input_path"],
        src_dim=model_cfg["src_dim"],
        input_length=data_cfg["input_length"],
        has_header=infer_cfg.get("input_has_header", True),
    )

    src, src_mask = prepare_source_tensor(
        src_window=src_window,
        device=device,
        input_mean=input_mean,
        input_std=input_std,
    )

    pred_norm = autoregressive_forecast(
        model=model,
        src=src,
        src_mask=src_mask,
        pred_length=data_cfg["pred_length"],
        tgt_dim=model_cfg["tgt_dim"],
        out_dim=model_cfg["out_dim"],
        start_token_mode=infer_cfg.get("start_token_mode", "zeros"),
    )

    if target_mean is not None:
        target_mean = target_mean.to(device)
    if target_std is not None:
        target_std = target_std.to(device)
    pred = denormalize_tensor(pred_norm, target_mean, target_std)

    output_path = infer_cfg.get("output_path", "outputs/predictions.csv")
    export_predictions_to_csv(pred, output_path)

    print("Inference completed.")
    print("device:", str(device))
    print("src shape:", tuple(src.shape))
    print("pred shape:", tuple(pred.shape))
    print("output:", output_path)


if __name__ == "__main__":
    main()
