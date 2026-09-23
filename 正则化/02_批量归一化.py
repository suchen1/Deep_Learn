"""
BatchNormld:主要应用于全连接层或处理一维数据的网络，例如文本处理。它接收形状为(N，num_features)的张量作为输入。
BatchNorm2d:主要应用于卷积神经网络，处理二维图像数据或特征图。它接收形状为(N，C，H，W)的张量作为输入。
BatchNorm3d:主要用于三维卷积神经网络(3D CNN)，处理三维数据，例如视频或医学图像。它接收形状为(N，C，D，H，W)的张量作为输入。
"""

import torch
import torch.nn as nn

t1 = torch.randn(1,3,3,4)
print(t1)

# 创建一个BatchNorm2d层
bath2d = nn.BatchNorm2d(num_features=3, eps=1e-5, momentum=0.1, affine=True)

t2 = bath2d(t1)
print(t2)
