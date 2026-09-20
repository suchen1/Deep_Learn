

# arange(), linspace(), manual_seed(), randint()
# 涉及到的函数:
# torch.arange() 和torch.linspace()创建线性张量
# torch.random.initial_seed() 和 torch.random.manual_seed() 随机种子设置
# torch.rand/randn() 创建随机浮点类型张量
# torch.randint(low，high，size=())创建随机整数类型张量

import torch

t1 = torch.arange(0, 10, step=2)  # 创建一个从0到10，步长为2的线性张量
print(f"t1:{t1}, type:{type(t1)}")  # 输出

t2 = torch.linspace(0, 1, steps=5)  # 创建一个从0到1，包含5个点的线性张量
print(f"t2:{t2}, type:{type(t2)}")  # 输出

# torch.initial_seed() # 获取当前随机数生成器的种子值
torch.manual_seed(42) # 设置随机数生成器的种子值为42
t3 = torch.rand(size=(3,4)) # 创建一个3行4列的随机浮点类型张量，均匀分布[0,1)之间
print(f"t3:{t3}, type:{type(t3)}")


t4 = torch.randn((4,3)) # 创建一个4行3列的随机浮点类型张量，正态分布，均值为0，标准差为1
print(f"t4:{t4}, type:{type(t4)}")

t5 = torch.randint(0, 10, size=(3,4)) # 创建一个3行4列的随机整数类型张量，范围在[0,10)之间
print(f"t5:{t5}, type:{type(t5)}")
