import torch

w = torch.tensor(10, requires_grad=True,dtype=torch.float)

for i in range(1,101):
    print(f"第{i}次迭代")
    loss = 2 * w **2
    print(f"更新前的 w.grad: {w.grad}, type: {type(w.grad)},w.data: {w.data}, type: {type(w.data)}, loss: {loss}, type: {type(loss)}")
    loss.backward()  # 计算梯度
    w.data = w.data - 0.1 * w.grad  # 更新权重
    print(f"更新后的 w.grad: {w.grad}, type: {type(w.grad)},w.data: {w.data}, type: {type(w.data)}")
    w.grad.zero_()  # 清空梯度
    print('*' * 20)