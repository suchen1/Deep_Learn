# cat（）不改变维度数，除了拼接的维度外，其他维度必须保持一致
# stack()会增加一个维度，拼接的维度必须保持一致

import torch

t1 = torch.randint(0, 10, (2,3))
t2 = torch.randint(0, 10, (2,3))

print(t1)
print(t2)

# 使用 cat() 进行拼接
t3 = torch.cat([t1, t2], dim=0) # (2,3) + (2,3) = (4,3)
print(t3)

print(t3.shape)

# 使用 stack() 进行拼接
t4 = torch.stack([t1, t2], dim=0)
print(t4)
print(t4.shape)