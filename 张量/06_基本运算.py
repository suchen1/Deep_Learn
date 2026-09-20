import torch

# add(), sub(), mul(), div() neg()分别对应加、减、乘、除 取反操作 + - * /
# add_(), sub_(), mul_(), div_() neg_()分别对应加、减、乘、除 取反操作 修改源数据数据
t1 = torch.tensor([[1, 2], [3, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])
print(t1 + t2)  # 张量加法
print(t1 - t2)  # 张量减法
print(t1 * t2)  # 张量乘法
print(t1 / t2)  # 张量除法

print(-t1)  # 张量取反

# @矩阵乘法
print(t1 @ t2)  # 矩阵乘法