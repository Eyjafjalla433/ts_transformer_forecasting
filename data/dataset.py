from typing import Iterable, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset


IndexLike = Optional[Union[Sequence[int], torch.Tensor]]


def _normalize_indices(indices: IndexLike, width: int) -> Sequence[int]:
    if indices is None:
        return list(range(width))
    if isinstance(indices, torch.Tensor):
        return indices.tolist()
    return list(indices)


class TimeSeriesWindowDataset(Dataset):
    """Convert a raw time series into sliding-window samples.

    Input series must be 2D with shape [T, F], where:
    - T: number of timesteps
    - F: number of features per timestep

    Each sample returns:
    - src: [L_in, src_dim]
    - tgt_full: [L_out + 1, tgt_dim]
    - y: [L_out, out_dim]
    """

    def __init__(
        self,
        series: Union[torch.Tensor, Iterable[Iterable[float]]],
        input_length: int,
        pred_length: int,
        src_cols: IndexLike = None,
        tgt_cols: IndexLike = None,
        out_cols: IndexLike = None,
        dtype: torch.dtype = torch.float32,
    ):
        if input_length <= 0 or pred_length <= 0:
            raise ValueError("input_length and pred_length must be positive.")

        tensor = torch.as_tensor(series, dtype=dtype)
        if tensor.dim() != 2:
            raise ValueError("series must be a 2D tensor/array with shape [T, F].")

        total_steps, feature_dim = tensor.shape
        src_cols = _normalize_indices(src_cols, feature_dim)
        tgt_cols = _normalize_indices(tgt_cols, feature_dim)
        out_cols = _normalize_indices(out_cols if out_cols is not None else tgt_cols, feature_dim)

        self.series = tensor
        self.input_length = input_length
        self.pred_length = pred_length
        self.src_cols = src_cols
        self.tgt_cols = tgt_cols
        self.out_cols = out_cols

        self.src_series = self.series[:, self.src_cols]
        self.tgt_series = self.series[:, self.tgt_cols]
        self.out_series = self.series[:, self.out_cols]

        self.num_samples = total_steps - input_length - pred_length + 1
        if self.num_samples <= 0:
            raise ValueError(
                "Not enough timesteps for the requested windows: "
                "need at least input_length + pred_length."
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Sample index out of range.")

        src_start = idx
        src_end = src_start + self.input_length

        tgt_start = src_end - 1
        tgt_end = src_end + self.pred_length

        y_start = src_end
        y_end = src_end + self.pred_length

        src = self.src_series[src_start:src_end]
        tgt_full = self.tgt_series[tgt_start:tgt_end]
        y = self.out_series[y_start:y_end]

        return {"src": src, "tgt_full": tgt_full, "y": y}
