import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 必须在 import torch 之前，否则 Qt 绘图会崩（见 README 的 OpenMP 说明）
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

svg1 = plt.imread('./data/1.jpg')
plt.imshow(svg1)
plt.show()
print(f'svg1.shape = {svg1.shape}')
img = torch.tensor(svg1).permute(2,0,1).float()  # (1,3,1706, 1279)

pool = nn.MaxPool2d(kernel_size=2,stride=2)

out = pool(img)
print(f'out.shape = {out.shape}')
print(f'out = {out}')