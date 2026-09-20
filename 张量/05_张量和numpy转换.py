import torch
import numpy as np
# numpy() 方法实现张量和numpy数组的相互转换 共享内存
# tensor() 将numpy数组转换为张量    不共享内存
# item() 将单元素张量转换为Python数值

t1 = torch.tensor([[1, 2], [3, 4]])
print(f"t1: {t1}, type: {type(t1)}")
# 将张量转换为numpy数组
np_array = t1.numpy()
print(f"np_array: {np_array}, type: {type(np_array)}")

t2 = np.random.randn(3,4)
print(f"t2: {t2}, type: {type(t2)}")
# 将numpy数组转换为张量
tensor_from_np = torch.tensor(t2)
print(f"tensor_from_np: {tensor_from_np}, type: {type(tensor_from_np)}")

t3 = torch.tensor([42])
print(f"t3: {t3}, type: {type(t3)}")
# 将单元素张量转换为Python数值
python_value = t3.item()
print(f"python_value: {python_value}, type: {type(python_value)}")