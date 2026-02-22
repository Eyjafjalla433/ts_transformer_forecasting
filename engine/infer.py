import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from models.model import make_model
from utils.checkpoint import load_model_weights as load_model_weights_from_ckpt
from .train import Batch, subsequent_mask


TensorLike = Union[torch.Tensor, Sequence[Sequence[float]]]


def resolve_device(device_name: str) -> torch.device:
    """Resolve runtime device with a safe CUDA fallback."""
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def build_model_from_config(model_cfg: Dict, device: torch.device) -> torch.nn.Module:
    """Construct a model from config fields used by make_model."""
    model = make_model(
        src_dim=model_cfg["src_dim"],
        tgt_dim=model_cfg["tgt_dim"],
        out_dim=model_cfg["out_dim"],
        N=model_cfg["N"],
        d_model=model_cfg["d_model"],
        d_ff=model_cfg["d_ff"],
        h=model_cfg["h"],
        dropout=model_cfg["dropout"],
    )
    return model.to(device)


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    """Load model weights via the shared checkpoint utility."""
    load_model_weights_from_ckpt(model=model, path=checkpoint_path, map_location=device, strict=True)
    model.eval()


def load_normalization_stats(stats_path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    """Load optional normalization stats from JSON file."""
    if not stats_path:
        return None

    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError("Normalization file not found: {}".format(path))

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    required = ["input_mean", "input_std", "target_mean", "target_std"]
    for key in required:
        if key not in payload:
            raise KeyError("Missing key '{}' in normalization stats.".format(key))

    return {
        "input_mean": torch.tensor(payload["input_mean"], dtype=torch.float32),
        "input_std": torch.tensor(payload["input_std"], dtype=torch.float32),
        "target_mean": torch.tensor(payload["target_mean"], dtype=torch.float32),
        "target_std": torch.tensor(payload["target_std"], dtype=torch.float32),
    }


def load_input_window_from_csv(
    input_path: str,
    src_dim: int,
    input_length: int,
    has_header: bool = True,
) -> torch.Tensor:
    """Load the most recent input window [L_in, src_dim] from a numeric CSV file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError("Input file not found: {}".format(path))

    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])

    if not rows:
        raise ValueError("Input CSV is empty: {}".format(path))

    tensor = torch.tensor(rows, dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError("Input CSV must produce a 2D tensor.")
    if tensor.size(1) < src_dim:
        raise ValueError("Input CSV has {} columns but src_dim is {}.".format(tensor.size(1), src_dim))
    if tensor.size(0) < input_length:
        raise ValueError("Input CSV has {} rows but input_length is {}.".format(tensor.size(0), input_length))

    return tensor[-input_length:, :src_dim]


def normalize_tensor(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply feature-wise normalization with numerical stability."""
    if mean is None or std is None:
        return x
    return (x - mean) / std.clamp_min(1e-6)


def denormalize_tensor(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    """Undo feature-wise normalization."""
    if mean is None or std is None:
        return x
    return x * std + mean


def prepare_source_tensor(
    src_window: TensorLike,
    device: torch.device,
    input_mean: Optional[torch.Tensor] = None,
    input_std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create model-ready src and src_mask.

    Returns:
    - src: [1, L_in, src_dim]
    - src_mask: [1, 1, L_in]
    """
    src = torch.as_tensor(src_window, dtype=torch.float32, device=device)
    if src.dim() != 2:
        raise ValueError("src_window must be 2D with shape [L_in, src_dim].")

    if input_mean is not None:
        input_mean = input_mean.to(device)
    if input_std is not None:
        input_std = input_std.to(device)
    src = normalize_tensor(src, input_mean, input_std)
    src = src.unsqueeze(0)
    src_mask = Batch.make_src_mask(src, pad_value=None)
    return src, src_mask


def make_decoder_start_token(
    src: torch.Tensor,
    tgt_dim: int,
    mode: str = "zeros",
) -> torch.Tensor:
    """Create the first decoder token [B, 1, tgt_dim]."""
    batch_size = src.size(0)
    device = src.device

    if mode == "zeros":
        return torch.zeros(batch_size, 1, tgt_dim, dtype=src.dtype, device=device)
    if mode == "last":
        if src.size(-1) < tgt_dim:
            raise ValueError("Cannot use mode='last' when src_dim < tgt_dim.")
        return src[:, -1:, :tgt_dim]
    raise ValueError("Unsupported start_token_mode: {}".format(mode))


def autoregressive_forecast(
    model: torch.nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    pred_length: int,
    tgt_dim: int,
    out_dim: int,
    start_token_mode: str = "zeros",
) -> torch.Tensor:
    """Run autoregressive decoding and return predictions [B, L_out, out_dim]."""
    if out_dim != tgt_dim:
        raise ValueError("Autoregressive decoding requires out_dim == tgt_dim.")
    if pred_length <= 0:
        raise ValueError("pred_length must be positive.")

    tgt_seq = make_decoder_start_token(src, tgt_dim=tgt_dim, mode=start_token_mode)
    preds: List[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(pred_length):
            tgt_len = tgt_seq.size(1)
            tgt_mask = subsequent_mask(tgt_len, device=src.device).expand(src.size(0), -1, -1)
            out = model(src, tgt_seq, src_mask, tgt_mask)
            next_step = out[:, -1:, :]
            preds.append(next_step)
            tgt_seq = torch.cat([tgt_seq, next_step], dim=1)

    return torch.cat(preds, dim=1)


def export_predictions_to_csv(pred: torch.Tensor, output_path: str) -> None:
    """Export [1, L_out, out_dim] predictions to CSV."""
    if pred.dim() != 3 or pred.size(0) != 1:
        raise ValueError("Expected prediction shape [1, L_out, out_dim].")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    values = pred[0].detach().cpu().tolist()
    out_dim = pred.size(-1)
    header = ["step"] + ["pred_{}".format(i) for i in range(out_dim)]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step, row in enumerate(values):
            writer.writerow([step] + row)
