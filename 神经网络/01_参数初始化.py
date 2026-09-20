import torch.nn as nn

"""
参数初始化方法

Sigmoid/Tanh:Xavier 初始化ReLU/Leaky ReLU: kaiming 初始化
激活函数的选择:根据激活函数的类型选择对应的初始化方法
网络的深度
浅层网络:随机初始化即可
深层网络:需要考虑方差平衡，如Xavier或kaiming初始化
"""

def dm01_随机分布():
    linear = nn.Linear(5, 3)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)
    print("线性层权重:", linear.weight)
    print("线性层偏置:", linear.bias)

def dm02_固定分布():
    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 0.5)
    nn.init.constant_(linear.bias, 0.5)
    print("线性层权重:", linear.weight)
    print("线性层偏置:", linear.bias)

def dm03_全0分布():
    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print("线性层权重:", linear.weight)
    print("线性层偏置:", linear.bias)

def dm04_kaiming():
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
    print("线性层权重:", linear.weight)
    print("线性层偏置:", linear.bias)

def dm05_xavier():
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)
    print("线性层权重:", linear.weight)
    print("线性层偏置:", linear.bias)

if __name__ == "__main__":
    # dm01_随机分布()
    # dm02_固定分布()
    # dm03_全0分布()
    # dm04_kaiming()
    dm05_xavier()