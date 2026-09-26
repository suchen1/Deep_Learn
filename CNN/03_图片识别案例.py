#导包
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

batch_size = 8

def creat_data():
    train_data = CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False)
    test_data = CIFAR10(root='./data', train=False, transform=transforms.ToTensor(), download=False)
    return train_data, test_data

class ImgMode(nn.Module):
    def __init__(self):
        super(ImgMode, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=576, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        print(f'x.shape = {x.shape}')  # (8,576)
        x = x.reshape(x.size(0), -1)  # 展平  从（8,16,6,6）到（8,576）
        # x.view(x.size(0), -1)
        print(f'x.shape = {x.shape}')  # (8,576)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    train_data, test_data = creat_data()
    print(f'train_data.shape = {train_data.data.shape}') #(50000, 32, 32, 3)
    print(f'test_data.shape = {test_data.data.shape}') #(10000, 32, 32, 3)
    #{'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    print(f'train_data.class_to_idx = {train_data.class_to_idx}')
    ImgModel = ImgMode()
    summary(ImgModel, (3, 32, 32),batch_size=8)
