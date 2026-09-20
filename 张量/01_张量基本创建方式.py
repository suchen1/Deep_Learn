import torch
import numpy as np


t1 = torch.tensor(10)
data = [[1,2,3],[4,5,6]]
t2 = torch.tensor(data,dtype=torch.float)
print(f't1: {t1}, t1 type: {type(t1)}')
print(f't2: {t2}, t2 type: {type(t2)}')
t3 = torch.Tensor(3,5)
print(f't3: {t3}, t3 type: {type(t3)}')

#场景3:numpy nd数组 >张量.
t4 = torch.IntTensor(data)
data = np.random.randint(0, 10, size=(2, 3))
print(f't4:{t4}, type:{type(t4)}')
#场景4:如果类型不匹配，会尝试自动转换类
t5 = torch.FloatTensor(data)
data = np.random.randint(0, 10, size=(2, 3))
print(f't5: {t5}, type: {type(t5)}')