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

x = torch.tensor(1.0, requires_grad=True, dtype=torch.float32)

optimizer = torch.optim.RMSprop([x], lr=0.01,momentum=0.9,alpha=0.99)

for i in range(2):
    optimizer.zero_grad()
    loss = ((x ** 2) / 2.0)
    loss.backward()
    optimizer.step()
    print(f"Step {i+1}: x = {x.item()}, loss = {loss.item()}")
