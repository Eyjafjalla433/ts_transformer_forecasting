import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from data import TimeSeriesWindowDataset
from engine import Batch, SimpleLossCompute, TrainState, run_epoch
from models.model import make_model
from utils.checkpoint import save_checkpoint
from utils.config import load_config


def load_series_from_csv(path: str, has_header: bool = True) -> torch.Tensor:
    """Load a numeric [T, F] time-series table from CSV."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError("Training CSV not found: {}".format(csv_path))

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
        raise ValueError("Training CSV is empty: {}".format(csv_path))

    return torch.tensor(rows, dtype=torch.float32)


def build_synthetic_series(total_steps: int, src_dim: int) -> torch.Tensor:
    """Build a deterministic synthetic series for pipeline validation."""
    t = torch.linspace(0, 20, total_steps)
    base = torch.sin(t) + 0.1 * torch.cos(3 * t)
    cols = []
    for i in range(src_dim):
        shift = 0.02 * i
        noise = 0.01 * torch.randn(total_steps)
        cols.append((base + shift + noise).unsqueeze(1))
    return torch.cat(cols, dim=1)


def build_batch_iter(loader: DataLoader, device: torch.device) -> Iterable[Batch]:
    """Convert DataLoader dict batches into Batch objects expected by run_epoch."""
    for sample in loader:
        src = sample["src"].to(device)
        tgt_full = sample["tgt_full"].to(device)
        yield Batch(src, tgt_full, pad_value=None)


def resolve_column_slices(
    series_width: int,
    src_dim: int,
    tgt_dim: int,
    out_dim: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Use the first src_dim columns as input and the last target columns as outputs."""
    if src_dim <= 0 or tgt_dim <= 0 or out_dim <= 0:
        raise ValueError("src_dim, tgt_dim, and out_dim must be positive.")
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


def split_series_for_train_val(
    series: torch.Tensor,
    val_ratio: float,
    input_length: int,
    pred_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the raw [T, F] table chronologically for train/validation.

    Validation keeps an input_length overlap so its first window can look back
    into the training history without leaking validation targets into training.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    total_steps = int(series.size(0))
    min_train_steps = input_length + pred_length
    min_total_steps = input_length + 2 * pred_length
    if total_steps < min_total_steps:
        raise ValueError(
            "Need at least {} timesteps for chronological train/val splitting, but got {}."
            .format(min_total_steps, total_steps)
        )

    val_steps = max(pred_length, int(total_steps * val_ratio))
    val_start = total_steps - val_steps
    val_start = max(val_start, min_train_steps)
    val_start = min(val_start, total_steps - pred_length)

    train_series = series[:val_start]
    val_series = series[val_start - input_length:]

    if train_series.size(0) < min_train_steps:
        raise ValueError("Training split is too short for the requested window lengths.")
    if val_series.size(0) < min_train_steps:
        raise ValueError("Validation split is too short for the requested window lengths.")

    return train_series, val_series


def _open_metrics_writer(metrics_path: str, append: bool = False):
    """Open CSV writer for per-epoch metrics and write header when needed."""
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    mode = "a" if append else "w"
    f = path.open(mode, encoding="utf-8", newline="")
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "epoch_seconds",
        "best_val_loss_so_far",
        "is_best_epoch",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if (not append) or (append and not file_exists):
        writer.writeheader()
    return f, writer


def main():
    """Train the model for multiple epochs and save the best checkpoint."""
    cfg = load_config("configs/default.yaml")
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("train", {})
    infer_cfg = cfg.get("infer", {})

    device_name = train_cfg.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    src_dim = int(model_cfg.get("src_dim", 6))
    tgt_dim = int(model_cfg.get("tgt_dim", 1))
    out_dim = int(model_cfg.get("out_dim", 1))
    if out_dim != tgt_dim:
        raise ValueError("Training setup expects out_dim == tgt_dim for shifted targets.")

    input_length = int(data_cfg.get("input_length", 24))
    pred_length = int(data_cfg.get("pred_length", 12))
    batch_size = int(data_cfg.get("batch_size", 16))

    train_csv = data_cfg.get("train_csv")
    train_csv_has_header = bool(data_cfg.get("train_csv_has_header", True))
    synthetic_steps = int(data_cfg.get("synthetic_total_steps", 1000))

    if train_csv:
        series = load_series_from_csv(train_csv, has_header=train_csv_has_header)
    else:
        series = build_synthetic_series(synthetic_steps, src_dim)

    src_cols, tgt_cols, out_cols = resolve_column_slices(
        series_width=int(series.size(1)),
        src_dim=src_dim,
        tgt_dim=tgt_dim,
        out_dim=out_dim,
    )

    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    train_series, val_series = split_series_for_train_val(
        series=series,
        val_ratio=val_ratio,
        input_length=input_length,
        pred_length=pred_length,
    )

    # Legacy normalization logic moved to preprocessing.py.
    # train_series = (train_series - mean) / std
    # val_series = (val_series - mean) / std

    train_set = TimeSeriesWindowDataset(
        series=train_series,
        input_length=input_length,
        pred_length=pred_length,
        src_cols=src_cols,
        tgt_cols=tgt_cols,
        out_cols=out_cols,
    )
    val_set = TimeSeriesWindowDataset(
        series=val_series,
        input_length=input_length,
        pred_length=pred_length,
        src_cols=src_cols,
        tgt_cols=tgt_cols,
        out_cols=out_cols,
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = make_model(
        src_dim=src_dim,
        tgt_dim=tgt_dim,
        out_dim=out_dim,
        N=int(model_cfg.get("N", 2)),
        d_model=int(model_cfg.get("d_model", 64)),
        d_ff=int(model_cfg.get("d_ff", 128)),
        h=int(model_cfg.get("h", 4)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    ).to(device)

    num_epochs = int(train_cfg.get("epochs", 20))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = int(train_cfg.get("warmup_steps", max(100, int(total_steps * 0.05))))
    warmup_steps = max(1, min(warmup_steps, total_steps))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.05))
    min_lr_ratio = max(0.0, min(min_lr_ratio, 1.0))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_compute = SimpleLossCompute()

    accum_iter = int(train_cfg.get("accum_iter", 1))
    train_state = TrainState()

    best_val_loss = float("inf")
    best_epoch = -1
    run_start = time.perf_counter()

    checkpoint_path = infer_cfg.get("checkpoint_path", "checkpoints/model.pt")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_path = str(train_cfg.get("metrics_csv_path", "outputs/train_metrics.csv"))
    metrics_append = bool(train_cfg.get("metrics_append", False))
    metrics_file, metrics_writer = _open_metrics_writer(metrics_path=metrics_path, append=metrics_append)

    print(
        "Loaded series shape: {} | train steps: {} | val steps: {} | train windows: {} | val windows: {}"
        .format(tuple(series.shape), train_series.size(0), val_series.size(0), len(train_set), len(val_set))
    )

    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.perf_counter()

            model.train()
            train_loss, train_state = run_epoch(
                data_iter=build_batch_iter(train_loader, device),
                model=model,
                loss_compute=loss_compute,
                optimizer=optimizer,
                scheduler=scheduler,
                mode="train",
                accum_iter=accum_iter,
                train_state=train_state,
            )

            model.eval()
            with torch.no_grad():
                val_loss, _ = run_epoch(
                    data_iter=build_batch_iter(val_loader, device),
                    model=model,
                    loss_compute=loss_compute,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    mode="eval",
                    accum_iter=accum_iter,
                    train_state=train_state,
                )

            is_best_epoch = 0
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                is_best_epoch = 1
                save_checkpoint(
                    path=str(checkpoint_path),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                    extra={"model_config": model_cfg},
                )

            epoch_seconds = time.perf_counter() - epoch_start
            lr = float(optimizer.param_groups[0]["lr"])
            metrics_row: Dict[str, float] = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": lr,
                "epoch_seconds": float(epoch_seconds),
                "best_val_loss_so_far": float(best_val_loss),
                "is_best_epoch": is_best_epoch,
            }
            metrics_writer.writerow(metrics_row)
            metrics_file.flush()

            print(
                "Epoch {:03d} | train_loss {:.6f} | val_loss {:.6f}".format(
                    epoch, train_loss, val_loss
                )
            )
    finally:
        metrics_file.close()

    print("Training completed.")
    print("best_epoch:", best_epoch)
    print("best_val_loss:", best_val_loss)
    print("saved_checkpoint:", str(checkpoint_path))
    print("metrics_csv:", metrics_path)
    print("total_seconds:", time.perf_counter() - run_start)


if __name__ == "__main__":
    main()


