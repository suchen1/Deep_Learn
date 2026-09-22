import torch
import torch.nn as nn

# l = 糹y * log ( S ( f(x) ) )

criterion = nn.CrossEntropyLoss()

y_true = torch.tensor([1,2,1,0], dtype=torch.int64)
y_pred = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5], [0.4, 0.3, 0.3]],requires_grad=True,dtype=torch.float)
loss = criterion(y_pred,y_true)
print(f'loss: {loss}')