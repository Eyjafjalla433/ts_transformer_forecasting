import torch.nn as nn
from .embeddings import TimeSeriesEmbedding
from .layers import Encoder, Decoder, EncoderLayer, DecoderLayer
from .attention import MultiHeadedAttention
from .layers import PositionwiseFeedForward

class TransformerTimeSeriesModel(nn.Module):
    """Encoder-decoder Transformer for time-series forecasting."""
    def __init__(self, src_dim, d_model, N, h, d_ff, dropout=0.1, tgt_dim=None, out_dim=1):
        super(TransformerTimeSeriesModel, self).__init__()
        if tgt_dim is None:
            tgt_dim = src_dim
        self.src_embed = TimeSeriesEmbedding(src_dim, d_model)
        self.tgt_embed = TimeSeriesEmbedding(tgt_dim, d_model)
        self.encoder = Encoder(EncoderLayer(d_model, MultiHeadedAttention(h, d_model), PositionwiseFeedForward(d_model, d_ff), dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, MultiHeadedAttention(h, d_model), MultiHeadedAttention(h, d_model), PositionwiseFeedForward(d_model, d_ff), dropout), N)
        self.generator = nn.Linear(d_model, out_dim)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Forward contract.

        src: [B, L_in, src_dim]
        tgt: [B, L_out, tgt_dim]
        src_mask: [B, 1, L_in]
        tgt_mask: [B, L_out, L_out]
        returns: [B, L_out, out_dim]
        """
        memory = self.encoder(self.src_embed(src), src_mask)
        output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        return self.generator(output)

def make_model(src_dim, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, tgt_dim=None, out_dim=1):
    model = TransformerTimeSeriesModel(src_dim, d_model, N, h, d_ff, dropout, tgt_dim=tgt_dim, out_dim=out_dim)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
