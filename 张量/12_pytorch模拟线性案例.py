# 导入相关模块
import torch
from torch.utils.data import TensorDataset # 构造数据集对象
from torch.utils.data import DataLoader # 数据加载器
from torch import nn #nn模块中有平方损失函数和假设函数
from torch import optim # optim模块中有优化器函数
from sklearn.datasets import make_regression # 创建线性回归模型数据集
import matplotlib.pyplot as plt

# 让 matplotlib 正常显示中文和负号，否则标题会变成一串方块
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def data_preprocess():
    # 生成线性回归数据集
    X, y, coef = make_regression(n_samples=100,
                           n_features=1,
                           noise=10,
                           bias = 5,
                           coef = True,
                           random_state=42)

    # 将数据转换为张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)  # 转换为列向量

    x_np = X_tensor.numpy()
    y_np = y_tensor.numpy()
    print(f"x_np: {x_np}, x_np shape: {x_np.shape}")
    print(f"y_np: {y_np}, y_np shape: {y_np.shape}")
    return X_tensor, y_tensor, coef

# coef = 41.7411003148779 bias = 5.0

def train(x, y, coef):
    dataset = TensorDataset(x, y)  # 构造数据集对象
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)  # 构造数据加载器对象
    model = nn.Linear(1, 1)  # 1 个输入特征 -> 1 个输出（回归）
    criterion = nn.MSELoss()  # 损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器
    loss_list = []
    for epoch in range(100):
        total_loss, total_sample = 0, 0  # 每个 epoch 都要重置，否则算的是历史累计平均
        for x_train, y_train in dataloader:
            y_pred = model(x_train)  # 前向传播
            loss = criterion(y_pred, y_train.reshape(-1, 1))  # 计算损失
            optimizer.zero_grad()  # 梯度清零（放在 backward 之前）
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加损失
            total_sample += 1  # 累加 batch 数
        avg_loss = total_loss / total_sample  # 本 epoch 的平均损失
        loss_list.append(avg_loss)  # 保存平均损失

    print(f"训练完成，真实系数: {float(coef):.4f}, "
          f"模型系数: {model.weight.item():.4f}, 模型偏置: {model.bias.item():.4f}")

    # ---------- 第一张图：损失曲线 ----------
    plt.figure()
    plt.plot(range(1, len(loss_list) + 1), loss_list)
    plt.title('损失轮数曲线')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()

    # ---------- 第二张图：拟合结果 ----------
    plt.figure()
    plt.scatter(x.numpy(), y.numpy(), s=12, alpha=0.6, label='真实数据点')

    # 预测直线：必须先把 x 排序再连线，否则会画成锯齿
    x_sorted, _ = torch.sort(x.flatten())
    y_pred_sorted = model(x_sorted.reshape(-1, 1)).detach().flatten()
    plt.plot(x_sorted.numpy(), y_pred_sorted.numpy(), color='red', label='预测直线')

    # 真实直线：画生成数据时用的公式 y = coef * x + bias（bias=5），
    # 而不是把 y 那 100 个散点直接连起来
    x_line = torch.linspace(x.min(), x.max(), 100)
    y_line = float(coef) * x_line + 5.0
    plt.plot(x_line.numpy(), y_line.numpy(), color='green', linestyle='--', label='真实直线')

    plt.title('线性回归预测结果')
    plt.legend()
    plt.grid(True)
    plt.show()


if(__name__ == "__main__"):
    x_tensor, y_tensor, coef = data_preprocess()
    train(x_tensor, y_tensor, coef)