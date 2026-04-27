import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


TRAIN_CSV = Path("data/processed/NASDAQ_wide_train.csv")
FUTURE_CSV = Path("data/processed/NASDAQ_wide_future.csv")
TARGET_COLUMN = "Close"
INPUT_LENGTH = 30
HORIZON = 12
STRIDE = 12
VAL_RATIO = 0.2
BATCH_SIZE = 64
EPOCHS = 24
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.08
TCN_CHANNELS = [32, 32, 32]
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = Path("checkpoints/NASDAQ_TCN.pt")
OUTPUT_PREDICTIONS = Path("outputs/NASDAQ_TCN_predictions.csv")
OUTPUT_METRICS = Path("outputs/NASDAQ_TCN_metrics.json")
OUTPUT_TRAIN_LOG = Path("outputs/NASDAQ_TCN_train_log.csv")


def format_epoch_progress(epoch: int, total_epochs: int, width: int = 20) -> str:
    ratio = 0.0 if total_epochs <= 0 else float(epoch) / float(total_epochs)
    filled = min(width, max(0, int(round(width * ratio))))
    bar = "#" * filled + "-" * (width - filled)
    return "[{}] {:>3}/{:<3} {:>5.1f}%".format(bar, epoch, total_epochs, ratio * 100.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frames(train_csv: Path, future_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_csv)
    future_df = pd.read_csv(future_csv)
    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in future_df.columns:
        raise KeyError("Target column '{}' must exist in both train/future CSVs.".format(TARGET_COLUMN))
    if list(train_df.columns) != list(future_df.columns):
        raise ValueError("Train/future columns do not match.")
    numeric_train = train_df.apply(pd.to_numeric, errors="coerce")
    numeric_future = future_df.apply(pd.to_numeric, errors="coerce")
    if numeric_train.isna().any().any() or numeric_future.isna().any().any():
        raise ValueError("Train/future CSVs must be fully numeric.")
    return numeric_train, numeric_future


def split_train_val(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    min_train_rows = INPUT_LENGTH + HORIZON
    if len(df) < min_train_rows + 1:
        raise ValueError("NASDAQ training split is too short for TCN.")
    val_rows = max(HORIZON, int(len(df) * VAL_RATIO))
    val_start = len(df) - val_rows
    val_start = max(val_start, min_train_rows)
    val_start = min(val_start, len(df) - 1)
    train_part = df.iloc[:val_start].reset_index(drop=True)
    val_part = df.iloc[val_start - INPUT_LENGTH :].reset_index(drop=True)
    return train_part, val_part


def compute_stats(train_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    feature_mean = train_df.mean(axis=0).to_numpy(dtype=np.float32)
    feature_std = train_df.std(axis=0).replace(0.0, 1e-6).fillna(1e-6).to_numpy(dtype=np.float32)
    target_mean = float(train_df[TARGET_COLUMN].mean())
    target_std = float(train_df[TARGET_COLUMN].std())
    target_std = target_std if target_std > 0 else 1e-6
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, target: np.ndarray):
        self.features = features.astype(np.float32)
        self.target = target.astype(np.float32)
        self.length = len(self.features) - INPUT_LENGTH - HORIZON + 1
        if self.length <= 0:
            raise ValueError("Not enough rows to build TCN windows.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        x = self.features[idx : idx + INPUT_LENGTH]
        y = self.target[idx + INPUT_LENGTH : idx + INPUT_LENGTH + HORIZON]
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y).unsqueeze(-1),
        }


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TCNForecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = input_dim
        for level, out_channels in enumerate(TCN_CHANNELS):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=2 ** level,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(TCN_CHANNELS[-1], horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        features = self.tcn(x)[:, :, -1]
        out = self.head(features)
        return out.unsqueeze(-1)


def build_loaders(train_df: pd.DataFrame, val_df: pd.DataFrame, stats: Dict[str, np.ndarray]) -> Tuple[DataLoader, DataLoader]:
    train_features = ((train_df.to_numpy(dtype=np.float32) - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
    val_features = ((val_df.to_numpy(dtype=np.float32) - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
    train_target = ((train_df[TARGET_COLUMN].to_numpy(dtype=np.float32) - stats["target_mean"]) / stats["target_std"]).astype(np.float32)
    val_target = ((val_df[TARGET_COLUMN].to_numpy(dtype=np.float32) - stats["target_mean"]) / stats["target_std"]).astype(np.float32)

    train_set = WindowDataset(train_features, train_target)
    val_set = WindowDataset(val_features, val_target)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    count = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss_sum += float(loss.item()) * x.size(0)
            count += x.size(0)
    return loss_sum / max(count, 1)


def train_model(train_loader: DataLoader, val_loader: DataLoader, input_dim: int, device: torch.device) -> Tuple[nn.Module, List[Dict[str, float]]]:
    model = TCNForecaster(input_dim=input_dim, horizon=HORIZON, dropout=DROPOUT).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    logs: List[Dict[str, float]] = []
    no_improve_epochs = 0
    run_start = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * x.size(0)
            sample_count += x.size(0)

        train_loss = train_loss_sum / max(sample_count, 1)
        val_loss = evaluate(model, val_loader, device)
        epoch_seconds = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - run_start
        remaining_epochs = max(EPOCHS - epoch, 0)
        eta_seconds = (elapsed / max(epoch, 1)) * remaining_epochs
        logs.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": min(best_val, val_loss),
                "epoch_seconds": epoch_seconds,
                "eta_seconds": eta_seconds,
            }
        )

        improved = val_loss < (best_val - EARLY_STOPPING_MIN_DELTA)
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        print(
            "[TCN] {} | train_loss {:.6f} | val_loss {:.6f} | best {:.6f} | epoch {:.1f}s | eta {:.1f}m".format(
                format_epoch_progress(epoch, EPOCHS),
                train_loss,
                val_loss,
                best_val,
                epoch_seconds,
                eta_seconds / 60.0,
            ),
            flush=True,
        )

        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print(
                "[TCN] early stopping at epoch {} after {} non-improving epochs.".format(
                    epoch, no_improve_epochs
                ),
                flush=True,
            )
            break

    model.load_state_dict(best_state)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    return model, logs


def run_future_forecast(model: nn.Module, train_df: pd.DataFrame, future_df: pd.DataFrame, stats: Dict[str, np.ndarray], device: torch.device) -> pd.DataFrame:
    history_features = ((train_df.to_numpy(dtype=np.float32) - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
    future_features = ((future_df.to_numpy(dtype=np.float32) - stats["feature_mean"]) / stats["feature_std"]).astype(np.float32)
    future_target = future_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    rows: List[Dict[str, float]] = []
    train_len = len(train_df)
    model.eval()

    for offset in range(0, len(future_df), STRIDE):
        context = np.concatenate([history_features, future_features[:offset]], axis=0)
        if len(context) < INPUT_LENGTH:
            continue
        window = context[-INPUT_LENGTH:]
        x = torch.from_numpy(window).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(x).squeeze(0).squeeze(-1).cpu().numpy()
        pred = pred_norm * stats["target_std"] + stats["target_mean"]
        steps = min(HORIZON, len(future_df) - offset)
        truth = future_target[offset : offset + steps]
        for horizon_idx in range(steps):
            rows.append(
                {
                    "origin_index": int(train_len + offset),
                    "target_index": int(train_len + offset + horizon_idx),
                    "horizon": int(horizon_idx + 1),
                    "prediction": float(pred[horizon_idx]),
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


def save_logs(logs: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(logs).to_csv(output_path, index=False)


def main() -> None:
    set_seed(SEED)
    train_df, future_df = load_frames(TRAIN_CSV, FUTURE_CSV)
    train_part, val_part = split_train_val(train_df)
    stats = compute_stats(train_part)
    train_loader, val_loader = build_loaders(train_part, val_part, stats)

    device = torch.device(DEVICE)
    model, logs = train_model(train_loader, val_loader, input_dim=train_df.shape[1], device=device)
    prediction_df = run_future_forecast(model, train_df, future_df, stats, device)
    metrics = compute_metrics(prediction_df)
    metrics.update(
        {
            "train_csv": str(TRAIN_CSV),
            "future_csv": str(FUTURE_CSV),
            "target_column": TARGET_COLUMN,
            "input_length": INPUT_LENGTH,
            "horizon": HORIZON,
            "checkpoint_path": str(CHECKPOINT_PATH),
        }
    )

    OUTPUT_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    with OUTPUT_METRICS.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    save_logs(logs, OUTPUT_TRAIN_LOG)

    print("TCN benchmark finished.")
    print("checkpoint:", str(CHECKPOINT_PATH))
    print("predictions:", str(OUTPUT_PREDICTIONS))
    print("metrics:", str(OUTPUT_METRICS))


if __name__ == "__main__":
    main()
