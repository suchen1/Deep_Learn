import torch
import torch.nn as nn

# l =  0.5 x^2  (|x| < 1)
#      |x| - 0.5 others   SmoothL1loss

criterion = nn.SmoothL1Loss()

y_true = torch.tensor([1,0,1,0], dtype=torch.float32)
y_pred = torch.tensor([0.7,0.3,0.5,0.2],requires_grad=True,dtype=torch.float)
loss = criterion(y_pred,y_true)
print(f'loss: {loss}')