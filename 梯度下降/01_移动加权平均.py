"""
梯度下降优化算法中，可能会碰到以下情况:
1.碰到平缓区域，梯度值较小，参数优化变慢
2.碰到“鞍点”，梯度为0，参数无法优化
3.碰到局部最小值，参数不是最优                      动量法    自适应学习率   自适应学习率   自适应动量法
对于这些问题，出现了一些对梯度下降算法的优化方法，例如: Momentum   AdaGrad     RMSprop    Adam等
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import matplotlib.pyplot as plt

def dm01():
    torch.manual_seed(54)
    tem = torch.randn(size=[30,], dtype=torch.float32) * 5 + 25
    print(tem)
    day = torch.arange(start=1, end=31, step=1, dtype=torch.int)
    print(day)
    plt.plot(day, tem,color='r')
    plt.scatter(day, tem,color='b')
    plt.show()

def dm02(pot):
        torch.manual_seed(54)
        tem = torch.randn(size=[30, ], dtype=torch.float32) * 5 + 25
        print(tem)
        day = []
        x = -1
        for i in range(30):
            if x == -1:
                x = tem[i]
            else:
                x = pot * x + (1 - pot) * tem[i]
            day.append(x)
        days = torch.arange(start=1, end=31, step=1, dtype=torch.int)
        plt.plot(days, day, color='r')
        plt.scatter(days, tem, color='b')
        plt.show()
if __name__ == '__main__':
    dm01()
    dm02(0.5)
    dm02(0.9)