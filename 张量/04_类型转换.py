import torch
from nltk.book import text3

# type(torch支持的类型)
# half()/double()/float()/int()/long()/short()/char()/byte()/bool()

t1 = torch.tensor([1,2,3])
print(f"t1:{t1}, type:{type(t1)}, dtype:{t1.dtype}") # 输出张量t1的类型和数据类型

t1 = t1.type(torch.float32)
print(f"t1:{t1}, type:{type(t1)}, dtype:{t1.dtype}") # 输出张量t1的类型和数据test

t3= torch.tensor([1,2,3], dtype=torch.float32)