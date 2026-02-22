from typing import Dict

import torch

from .train import Batch


def evaluate_regression(model: torch.nn.Module, data_loader, device: torch.device) -> Dict[str, float]:
    """Evaluate a forecasting model on a dataloader of {'src','tgt_full','y'} batches.

    Returns metric dictionary with mse/rmse/mae.
    """
    model.eval()

    mse_sum = 0.0
    mae_sum = 0.0
    count = 0

    with torch.no_grad():
        for sample in data_loader:
            src = sample["src"].to(device)
            tgt_full = sample["tgt_full"].to(device)

            batch = Batch(src, tgt_full, pad_value=None)
            pred = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            target = batch.tgt_y

            diff = pred - target
            mse_sum += torch.sum(diff * diff).item()
            mae_sum += torch.sum(torch.abs(diff)).item()
            count += target.numel()

    if count == 0:
        raise ValueError("Evaluation loader produced zero targets.")

    mse = mse_sum / count
    mae = mae_sum / count
    rmse = mse ** 0.5
    return {"mse": mse, "rmse": rmse, "mae": mae}
