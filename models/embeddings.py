import torch
import torch.nn as nn
import math

# Time Series Embedding
class TimeSeriesEmbedding(nn.Module):
    """Map time-series features [B, L, input_dim] to model states [B, L, d_model]."""
    def __init__(self, input_dim, d_model):
        super(TimeSeriesEmbedding, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model) 
        self.position_encoding = PositionalEncoding(d_model, dropout=0.1)

    def forward(self, x):
        """x: [B, L, input_dim] -> [B, L, d_model]."""
        x = self.embedding(x)
        x = self.position_encoding(x) 
        return x

class PositionalEncoding(nn.Module):
    """Add sinusoidal positional encoding to [B, L, d_model]."""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [B, L, d_model] -> [B, L, d_model]."""
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
