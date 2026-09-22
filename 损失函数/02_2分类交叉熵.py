import torch
import torch.nn as nn

# l = - y * log 预测  - （1 - y) * log （1 - 预测)

criterion = nn.BCELoss()

y_true = torch.tensor([1,0,1,0], dtype=torch.float32)
y_pred = torch.tensor([0.7,0.3,0.5,0.2],requires_grad=True,dtype=torch.float)
loss = criterion(y_pred,y_true)
print(f'loss: {loss}')