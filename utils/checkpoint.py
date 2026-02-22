from pathlib import Path
from typing import Any, Dict, Optional

import torch


def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    """Load a checkpoint with a safe default for modern PyTorch versions.

    Returns a normalized dictionary. If the file only stores a raw state_dict,
    the return value is wrapped as {"model_state_dict": state_dict}.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint file not found: {}".format(ckpt_path))

    try:
        payload = torch.load(str(ckpt_path), map_location=map_location, weights_only=True)
    except TypeError:
        # Backward compatibility for older PyTorch versions.
        payload = torch.load(str(ckpt_path), map_location=map_location)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload
    if isinstance(payload, dict):
        return {"model_state_dict": payload}
    raise TypeError("Unsupported checkpoint format in {}.".format(ckpt_path))


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model and optional training state into a single checkpoint."""
    payload: Dict[str, Any] = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if best_val_loss is not None:
        payload["best_val_loss"] = float(best_val_loss)
    if extra:
        payload.update(extra)

    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(ckpt_path))


def load_model_weights(
    model: torch.nn.Module,
    path: str,
    map_location=None,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model weights from checkpoint and return full checkpoint payload."""
    ckpt = load_checkpoint(path=path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    return ckpt


def restore_training_state(
    checkpoint: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> None:
    """Restore optimizer/scheduler state when available."""
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
