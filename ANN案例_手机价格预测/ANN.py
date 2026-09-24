#导包
import torch #PyTorch框架，封装了张量的各种操作
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset #数据集对象. 数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader #数据加载器.
import torch.nn as nn #neural network  # 封装了神经网络的各种操作
import torch.optim as optim #优化器
from sklearn.model_selection import train_test_split # 训练集和测试集的划分
import matplotlib.pyplot as plt #绘图
import numpy as np # 数组(矩阵)操作
import torch.nn.functional as F # 封装了神经网络的各种操作
import pandas as pd # 数据处理
import time

def creat_dataset():
    data = pd.read_csv('./data/CellPhone_train.csv')
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    x = x.astype('float32')

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 114514,shuffle=True)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))
    return train_dataset, test_dataset,x.shape[1],len(y.unique()) # 返回训练集、测试集、输入特征数、输出特征值

class PhoneMode(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 1024)
        self.linear5 = nn.Linear(1024, 512)
        self.linear6 = nn.Linear(512, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.leaky_relu((self.linear1(x)))
        x = F.leaky_relu((self.linear2(x)))
        x = F.leaky_relu((self.linear3(x)))
        x = F.leaky_relu((self.linear4(x)))
        x = F.leaky_relu((self.linear5(x)))
        x = F.leaky_relu((self.linear6(x)))
        x = self.output(x)
        return x

def train(train_dataset, input_dim, out_dim):
    train_load = DataLoader(train_dataset, batch_size=50, shuffle=True) # 每批次160条，共1600/160=10批次
    model = PhoneMode(input_dim, out_dim)
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001,betas=[0.95, 0.999]) # Adam优化器 #0.9是动量法，0.999是自适应学习率

    epochs = 200

    for epoch in range(epochs):
        total_loss, batch_num = 0, 0
        start = time.time()
        for x, y in train_load:
            model.train() # 训练模式
            optimizer.zero_grad() # 梯度清零
            y_pred = model(x) # 前向传播
            loss = criterion(y_pred, y) # 计算损失
            loss.backward() # 反向传播
            optimizer.step() # 更新参数
            total_loss += loss.item()
            batch_num += 1
        end = time.time()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/batch_num:.4f}, Time: {end-start:.2f}s')

    torch.save(model.state_dict(), './model/phone_model.pth') # 保存模型参数

def model_test(test_dataset, input_dim, outdim):
    model = PhoneMode(input_dim, outdim)
    model.load_state_dict(torch.load('./model/phone_model.pth'))
    test_load = DataLoader(test_dataset, batch_size=50, shuffle=False)
    corrent = 0
    model.eval()  # 测试模式
    for x, y in test_load:
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)  # 取最大值的索引作为预测结果
        corrent += (y_pred == y).sum()
    print(f'测试集准确率: {corrent/len(test_dataset)*100:.2f}%')
if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = creat_dataset()
    print(f'训练集样本数: {len(train_dataset)}, \n 测试集样本数: {len(test_dataset)}, \n输入特征数: {input_dim}, \n输出特征值: {output_dim}')
    #train(train_dataset, input_dim, output_dim)
    model_test(test_dataset, input_dim, output_dim)


