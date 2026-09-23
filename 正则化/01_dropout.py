import torch
import torch.nn as nn

t1 = torch.randint(0,10,size=(1,5))
print('原始张量:',t1)

linear1 = nn.Linear(5,5)
t1 = linear1(t1.float())
print('线性变换后的张量:',t1)

output = torch.relu(t1)
print('ReLU激活后的张量:',output)

dropout = nn.Dropout(0.5)
output = dropout(output)
print('Dropout后的张量:',output)