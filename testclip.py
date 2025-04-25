import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_

model = nn.Linear(10, 1)
opt   = optim.SGD(model.parameters(), lr=1e-3)

# fake loss
x = torch.randn(4, 10)
y = torch.randn(4, 1)
loss = nn.MSELoss()(model(x), y)
loss.backward()

# compute gradient norm BEFORE clipping, and clip in-place
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

print("Total norm before clipping:", total_norm)
for p in model.parameters():
    # these are the clipped gradients
    print(p.grad.norm().item())
