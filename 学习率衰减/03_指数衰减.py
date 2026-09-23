import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch import optim
import matplotlib.pyplot as plt

lr,epochs,iteration = 0.1,200,50
y_true = torch.tensor(1, dtype=torch.float32)
x = torch.tensor(1, dtype=torch.float32)
w = torch.tensor(1, dtype=torch.float32, requires_grad=True)

optimizer = optim.SGD([w], lr=lr,momentum=0.9)

scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

epoch_list,lr_list = [],[]

for epoch in range(epochs):
    epoch_list.append(epoch)
    lr_list.append(optimizer.param_groups[0]['lr'])
    for iteration in range(iteration):
        optimizer.zero_grad()
        y_pred = w * x
        loss = (y_pred - y_true) ** 2
        loss.backward()
        optimizer.step()
    scheduler.step()

    print(f"Epoch: {epoch}, Learning Rate: {optimizer.param_groups[0]['lr']}")

plt.plot(epoch_list,lr_list)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.show()