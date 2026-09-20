# 1. 索引操作
# 1.1 简单行列索引
import torch

torch.manual_seed(23)  # 设置随机数生成器的种子值为23
data = torch.randint(0, 10, (5,5))
print(data)

# 获取第一行
print(data[0])

# 获取第一列
print(data[:, 0])

# 获取元素
print(data[0, 0])
print('*'*50)
#1.2 列表索引
print(data[[1,2],[2,4]]) # 获取第1行第2列和第2行第4列的元素 前面是行，后面是列
print(data[[[1],[2]],[2,3,4]]) # 获取第1行和第2行的第2列-第4列的元素 前面是行，后面是列
print('*'*50)
#1.3 范围索引
print(data[:3, :2]) # 获取前3行和前2列的元素

print(data[1:,:2]) # 获取第1行到最后一行和前2列的元素

print(data[1::2, 1::2]) # 获取奇行奇列的元素
print('*'*100)
#1.4 布尔索引
print(data[data[:, 2] >= 5]) # 获取第3列大于5的行
print(data[:, data[1] >= 5]) # 获取第2行的大于5的元素
print('*'*100)
#1.5 多维索引
t2 = torch.randint(0, 10, (3, 4, 5))
print(t2)
print(t2[0]) # 获取第1个二维张量
print(t2[0, 1]) # 获取第1个二维张量的第2行
print(t2[0, 1, 2]) # 获取第1个二维张量的第2行第3列的元素
print(t2[:, 1, 2]) # 获取所有二维张量的第2行第3列的元素