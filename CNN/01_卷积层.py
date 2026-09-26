import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 必须在 import torch 之前，否则 Qt 绘图会崩（见 README 的 OpenMP 说明）
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

svg1 = plt.imread('./data/1.jpg')
plt.imshow(svg1)
plt.show()
print(f'svg1.shape = {svg1.shape}')
img = torch.tensor(svg1).permute(2,0,1).unsqueeze(0).float()  # (1,3,1706, 1279)
print(f'img.shape = {img.shape}')

conv = nn.Conv2d(in_channels=3,
                 out_channels=10,
                 kernel_size=3)

svg2 = conv(img)
print(f'svg2.shape = {svg2.shape}')

svg3 = svg2[0]
svg3 = svg3.permute(1,2,0).detach().numpy()  # (1704, 1277, 4)
print(f'svg3.shape = {svg3.shape}')

for i in range(10):
    plt.imshow(svg3[:,:,i])
    plt.show()