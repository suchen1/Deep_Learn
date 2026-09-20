import torch

# torch.ones 和 torch.ones_like 创建全1张量
# torch.zeros 和 torch.zeros_like 创建全0张量
# torch.full和 torch.full_like 创建全为指定值张量



t1 = torch.ones(3,4) # 创建一个3行4列的张量，元素全为1
print(f"t1:{t1}, type:{type(t1)}")

t2 = torch.zeros(3,4) # 创建一个3行4列的张量，元素全为0
print(f"t2:{t2}, type:{type(t2)}")

t3 = torch.full((3,4), 5.0) # 创建一个3行4列的张量，元素全为5.0
print(f"t3:{t3}, type:{type(t3)}")

t4 = torch.ones_like(t3) # 创建一个与t3形状相同、元素全为1的张量
print(f"t4:{t4}, type:{type(t4)}")

t5 = torch.zeros_like(t3) # 创建一个与t3形状相同、元素全为0的张量
print(f"t5:{t5}, type:{type(t5)}")
