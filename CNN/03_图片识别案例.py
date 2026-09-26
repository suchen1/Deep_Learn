#导包
import os

from sklearn.preprocessing import StandardScaler

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

batch_size = 10

def creat_data():
    train_data = CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    test_data = CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    # todo: 数据标准化
    return train_data, test_data

class ImgMode(nn.Module):
    def __init__(self):
        super(ImgMode, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=256 * 2 * 2, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        # print(f'x.shape = {x.shape}')  # (8,576)
        x = x.reshape(x.size(0), -1)  # 展平  从（(8,256,2,2)到(8,1024)
        # x.view(x.size(0), -1)
        # print(f'x.shape = {x.shape}')  # (8,576)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train(train_data,device):
    dataLoader = DataLoader(train_data, batch_size=batch_size)
    model = ImgMode().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    epochs = 20
    model.train()
    for epoch in range(epochs):
        total_loss,total_samples,correct,start = 0,0,0,time.time()
        for x,y in dataLoader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            total_samples += len(y)
            correct += (torch.max(y_pred, 1).indices == y).sum().item()
        print(f'epoch = {epoch}, loss = {total_loss/total_samples}, acc = {correct/total_samples * 100:.5f}%, time = {time.time()-start}s')

    torch.save(model.state_dict(), './model/ImgModel.pth')

def evaluate(test_data):
    dataLoader = DataLoader(test_data, batch_size=batch_size)
    model = ImgMode().to(device)
    model.load_state_dict(torch.load('./model/ImgModel.pth'))
    model.eval()
    total_samples, correct = 0, 0
    for x,y in dataLoader:
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        total_samples += len(y)
        correct += (torch.max(y_pred, 1).indices == y).sum().item()
    print(f'eval acc = {correct/total_samples * 100:.5f}%')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, test_data = creat_data()
    # print(f'train_data.shape = {train_data.data.shape}') #(50000, 32, 32, 3)
    # print(f'test_data.shape = {test_data.data.shape}') #(10000, 32, 32, 3)
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print(f'train_data.class_to_idx = {train_data.class_to_idx}')
    # summary(ImgModel, (3, 32, 32),batch_size=8)
    print('---------start train---------')
    train(train_data, device)
    print('---------end train---------') # 87.5%
    evaluate(test_data)  # 69.81%