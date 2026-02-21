import torch
import torch.nn as nn

from engine import Batch
from models.model import make_model

B = 16
L_IN = 24
L_OUT = 12
SRC_DIM = 6
TGT_DIM = 1
OUT_DIM = 1

src = torch.randn(B, L_IN, SRC_DIM)
tgt_full = torch.randn(B, L_OUT + 1, TGT_DIM)
batch = Batch(src, tgt_full, pad_value=None)

model = make_model(
    src_dim=SRC_DIM,
    tgt_dim=TGT_DIM,
    out_dim=OUT_DIM,
    N=2,
    d_model=64,
    d_ff=128,
    h=4,
)

pred = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
loss = nn.MSELoss()(pred, batch.tgt_y)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()

print("one-batch ok")
print("src:", tuple(batch.src.shape))
print("tgt_in:", tuple(batch.tgt.shape))
print("y:", tuple(batch.tgt_y.shape))
print("src_mask:", tuple(batch.src_mask.shape))
print("tgt_mask:", tuple(batch.tgt_mask.shape))
print("pred:", tuple(pred.shape))
print("loss:", float(loss))
