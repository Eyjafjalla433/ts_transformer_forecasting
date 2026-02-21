import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def subsequent_mask(size: int, device=None) -> torch.Tensor:
    """Mask out subsequent positions (causal mask)."""
    attn_shape = (1, size, size)
    # True means visible, False means masked.
    return torch.tril(torch.ones(attn_shape, dtype=torch.bool, device=device))


class Batch:
    """Hold a batch of time-series tensors and corresponding masks.

    Expected shapes:
    - src: [B, L_in, src_dim]
    - tgt: [B, L_out, tgt_dim] (full target sequence before shift)
    """

    def __init__(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None, pad_value=None):
        self.src = src
        self.src_mask = self.make_src_mask(src, pad_value)

        if tgt is not None:
            self.tgt = tgt[:, :-1, :]
            self.tgt_y = tgt[:, 1:, :]
            self.tgt_mask = self.make_tgt_mask(self.tgt, pad_value)
            self.ntokens = self.count_valid_values(self.tgt_y, pad_value)

    @staticmethod
    def make_src_mask(src: torch.Tensor, pad_value=None) -> torch.Tensor:
        if pad_value is None:
            return torch.ones(src.size(0), 1, src.size(1), dtype=torch.bool, device=src.device)
        valid = (src != pad_value).any(dim=-1)
        return valid.unsqueeze(1)

    @staticmethod
    def make_tgt_mask(tgt: torch.Tensor, pad_value=None) -> torch.Tensor:
        if pad_value is None:
            tgt_mask = torch.ones(tgt.size(0), 1, tgt.size(1), dtype=torch.bool, device=tgt.device)
        else:
            tgt_mask = (tgt != pad_value).any(dim=-1).unsqueeze(1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(1), device=tgt.device)
        return tgt_mask

    @staticmethod
    def count_valid_values(tgt_y: torch.Tensor, pad_value=None) -> torch.Tensor:
        if pad_value is None:
            count = tgt_y.size(0) * tgt_y.size(1) * tgt_y.size(2)
            return torch.tensor(count, device=tgt_y.device)
        valid_steps = (tgt_y != pad_value).any(dim=-1, keepdim=True)
        valid_values = valid_steps.expand_as(tgt_y).sum()
        return valid_values.clamp_min(1)


@dataclass
class TrainState:
    """Track number of steps, examples, and values processed."""

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0


class SimpleLossCompute:
    """MSE loss compute for time-series regression with mask-aware normalization."""

    def __init__(self):
        self.criterion = nn.MSELoss(reduction="none")

    def __call__(self, out: torch.Tensor, y: torch.Tensor, ntokens: torch.Tensor):
        loss_raw = self.criterion(out, y)
        loss_sum = loss_raw.sum()
        loss_node = loss_sum / ntokens
        return loss_sum.item(), loss_node


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Run one epoch for time-series Transformer."""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += int(batch.ntokens.item())

            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += int(batch.ntokens.item())
        tokens += int(batch.ntokens.item())

        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %8.6f "
                    + "| Values / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / max(int(batch.ntokens.item()), 1), tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0

        del loss
        del loss_node

    return total_loss / max(total_tokens, 1), train_state
