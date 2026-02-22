import csv
from pathlib import Path
from typing import Iterable, List

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

from data import TimeSeriesWindowDataset
from engine import Batch, SimpleLossCompute, TrainState, run_epoch
from models.model import make_model
from utils.checkpoint import save_checkpoint
from utils.config import load_config


def load_series_from_csv(path: str, has_header: bool = True) -> torch.Tensor:
    """Load numeric time-series table from CSV into a float tensor [T, F]."""
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

    if series.size(1) < src_dim:
        raise ValueError("Series has {} columns but src_dim is {}.".format(series.size(1), src_dim))

    src_cols = list(range(src_dim))
    tgt_cols = [src_dim - 1 + i for i in range(tgt_dim)]
    out_cols = [src_dim - 1 + i for i in range(out_dim)]
    if max(tgt_cols + out_cols) >= series.size(1):
        raise ValueError(
            "Series has {} columns, but target/output columns require at least {} columns."
            .format(series.size(1), max(tgt_cols + out_cols) + 1)
        )

    dataset = TimeSeriesWindowDataset(
        series=series,
        input_length=input_length,
        pred_length=pred_length,
        src_cols=src_cols,
        tgt_cols=tgt_cols,
        out_cols=out_cols,
    )

    val_ratio = float(train_cfg.get("val_ratio", 0.2))
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Not enough samples to split into train/val.")

    split_seed = int(train_cfg.get("split_seed", 42))
    generator = torch.Generator().manual_seed(split_seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

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

    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)))
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    loss_compute = SimpleLossCompute()

    num_epochs = int(train_cfg.get("epochs", 20))
    accum_iter = int(train_cfg.get("accum_iter", 1))
    train_state = TrainState()

    best_val_loss = float("inf")
    best_epoch = -1

    checkpoint_path = infer_cfg.get("checkpoint_path", "checkpoints/model.pt")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
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

        print(
            "Epoch {:03d} | train_loss {:.6f} | val_loss {:.6f}".format(
                epoch, train_loss, val_loss
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(
                path=str(checkpoint_path),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_loss=best_val_loss,
                extra={"model_config": model_cfg},
            )

    print("Training completed.")
    print("best_epoch:", best_epoch)
    print("best_val_loss:", best_val_loss)
    print("saved_checkpoint:", str(checkpoint_path))


if __name__ == "__main__":
    main()
