# Deep Learn

跟随 **黑马程序员《神经网络与深度学习》** 课程的学习代码与实践记录。

- 课程地址：<https://www.bilibili.com/video/BV1c5yrBcEEX/>
- 课程规模：共 108 P，总时长约 31.7 小时
- 内容覆盖：PyTorch 框架 → ANN 人工神经网络 → CNN 图像分类 → RNN 文本生成
- 分集进度：见 [`课程进度.md`](课程进度.md)

本仓库是个人学习记录。全部 108 P 已学完，代码按课程小节逐个练习整理并实际跑通，
三个实战案例都留下了可复现的结果与踩坑记录，用于复习和查漏补缺。

## 实战案例成果

| 案例 | 对应小节 | 模型结构 | 结果 |
| --- | --- | --- | --- |
| 手机价格分类（ANN） | P73–P79 | 全连接网络 + BatchNorm | 测试集准确率 **93%** |
| CIFAR-10 图像分类（CNN） | P92–P98 | 3 层卷积 + 3 层全连接 | 训练集 **97.10%** / 测试集 **72.67%** |
| AI 歌词生成（RNN） | P104–P108 | Embedding + RNN + Linear | 可生成连贯歌词片段 |

RNN 歌词生成实测输出（起始词「晴天」，20 个词）：

```
晴天
蝴蝶自在飞
花也布满天
一朵一朵因你而香
```

## 环境

| 项 | 版本 / 说明 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.14.6（Anaconda，`C:\ProgramData\anaconda3`） |
| PyTorch | **2.14.0+cu130**（CUDA 13.0） |
| torchvision | 0.29.0+cu130 |
| cuDNN | 9.24.0 |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.0 |
| pandas | 3.0.3 |
| jieba | 0.42.1 |
| torchsummary | 1.5.1 |
| GPU | NVIDIA GeForce RTX 5060 Ti（16 GB，sm_120 / Blackwell） |
| IDE | PyCharm 2026.1 |

> **关于 GPU**：初期使用 CPU 版 PyTorch（`2.14.0+cpu`），到 CNN 图像分类实战时训练太慢，
> 换成了 CUDA 版。API 完全一致，只有构建标签从 `+cpu` 变成 `+cu130`，
> **不写 `.to(device)` 的代码行为不变**（张量默认仍在 CPU 上）。
>
> RTX 5060 Ti 是 Blackwell 架构（sm_120），需要 CUDA 12.8 以上的构建。
> 只检查 `torch.cuda.is_available()` 是不够的，必须真的跑一次 GPU 矩阵乘法
> 才能确认 wheel 里编进了对应架构的 kernel。

## 目录结构

```
Deep Learn/
├── 张量/              12 个文件   Day01–Day03  张量创建、运算、索引、形状、拼接、自动微分、线性回归案例
├── 神经网络/           3 个文件   Day03–Day04  参数初始化、神经网络搭建、激活函数
├── 损失函数/           5 个文件   Day04        多分类/二分类交叉熵、MAE、MSE、SmoothL1
├── 梯度下降/           5 个文件   Day05        指数移动加权平均、动量法、AdaGrad、RMSProp、Adam
├── 学习率衰减/         3 个文件   Day05        等间隔、指定间隔、指数衰减
├── 正则化/             2 个文件   Day05        dropout、批量归一化 BatchNorm1d/2d/3d
├── ANN/                Day05–Day06  手机价格分类案例
│   ├── ANN案例_手机价格预测.py
│   ├── data/           CellPhone_train.csv、CellPhone_test.csv
│   └── model/          phone_model.pth
├── CNN/                Day06–Day07  卷积、池化、CIFAR-10 图像分类
│   ├── 01_卷积层.py
│   ├── 02_池化层.py
│   ├── 03_图片识别案例.py
│   ├── data/           1.jpg、cifar-10-batches-py/（CIFAR-10 Python 版数据集）
│   └── model/          ImgModel.pth
├── RNN/                Day07        词嵌入、RNN、AI 歌词生成器
│   ├── 01_词嵌入层.py
│   ├── 02_RNN.py
│   ├── 03_歌词生成.py
│   ├── data/           jaychou_lyrics.txt
│   └── model/          lyrics.pth
├── README.md
└── 课程进度.md          108 P 完整课程地图与进度清单
```

## 学习进度

完整的分集清单见 [`课程进度.md`](课程进度.md)。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| Day01 | 深度学习概述 + PyTorch 张量创建 | ✅ 已完成 |
| Day02 | 张量进阶（运算 / 索引 / 形状 / 拼接）+ 自动微分 | ✅ 已完成 |
| Day03 | 线性回归 + 神经网络入门 + 激活函数 | ✅ 已完成 |
| Day04 | 网络搭建 + 损失函数 + 反向传播 | ✅ 已完成 |
| Day05 | 优化器 + 学习率衰减 + 正则化 + ANN 案例 | ✅ 已完成 |
| Day06 | ANN 案例收尾 + CNN 卷积与池化 | ✅ 已完成 |
| Day07 | CNN 图像分类实战 + RNN 文本生成 | ✅ 已完成 |

## 运行方式

命令行直接运行单个脚本：

```bash
python 张量/01_张量基本创建方式.py
```

或在 PyCharm 中打开对应文件，按 `Shift+F10` 运行。

**三个实战案例请在各自目录下运行**，脚本里的数据路径（`./data/...`）是相对于工作目录的：

```bash
cd ANN  && python ANN案例_手机价格预测.py
cd CNN  && python 03_图片识别案例.py
cd RNN  && python 03_歌词生成.py
```

> 在 Windows 上如果 `python` 命令被 Microsoft Store 的应用别名占用了，用完整路径：
> `C:\ProgramData\anaconda3\python.exe`

### 数据说明

| 案例 | 数据来源 | 是否需要下载 |
| --- | --- | --- |
| 张量 / 神经网络 / 损失函数 / 梯度下降 / 学习率衰减 / 正则化 | `sklearn` 现场生成或随机初始化 | 不需要 |
| ANN 手机价格分类 | 仓库内 `ANN/data/*.csv` | 不需要 |
| CNN 卷积层 / 池化层 | 仓库内 `CNN/data/1.jpg` | 不需要 |
| CNN 图像分类 | 仓库内 `CNN/data/cifar-10-batches-py/` | 不需要（脚本里 `download=False`） |
| RNN 歌词生成 | 仓库内 `RNN/data/jaychou_lyrics.txt` | 不需要 |

所有数据都已随仓库提供，**克隆下来即可直接运行**。

### 依赖随课程推进逐步增加

| 依赖 | 从哪个文件开始需要 |
| --- | --- |
| `torch` | `张量/01` 起 |
| `numpy` | `张量/01` 起 |
| `scikit-learn` | `张量/12` 起（生成回归数据集） |
| `matplotlib` | `张量/12` 起（损失曲线、拟合结果可视化） |
| `torchsummary` | `神经网络/02` 起（打印模型结构与参数量） |
| `pandas` | `ANN/ANN案例_手机价格预测.py`（读取 CSV） |
| `jieba` | `RNN/01_词嵌入层.py` 起（中文分词） |

## 踩坑记录

这些是实际跑代码时撞到并解决的问题，按出现顺序排列。

### 1. OpenMP 运行时冲突（Windows）

本机同时存在**两份** `libiomp5md.dll`：

```
torch\lib\libiomp5md.dll            ← PyTorch 自带（约 1.54 MB）
Library\bin\libiomp5md.dll          ← Anaconda 的 intel-openmp 包（约 1.25 MB）
```

当 matplotlib 启用 **Qt 交互式后端**真的去绘图时，两份运行时会在同一进程内冲突，
报 `OMP: Error #15` 并**直接终止进程** —— `plt.show()` 根本执行不到，所以表现为"图不显示"。

绕过方式是在文件开头加两行，**必须在 `import torch` 之前**：

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

`梯度下降/`、`学习率衰减/`、`CNN/` 下的绘图脚本都已加上。
`张量/12_pytorch模拟线性案例.py` 也调用了 `plt.show()` 但尚未加这两行，
如果在你的机器上报 `OMP: Error #15`，照抄即可。

> 根治办法是让环境里只保留一份 OpenMP 运行时（或给深度学习单开一个干净的 conda 环境），
> 当前的绕过方式对本项目的简单绘图任务风险极低。

### 2. GPU 版 PyTorch 的安装与验证

国内直连 `download.pytorch.org` 只有 3–50 kB/s，2 GB 的 wheel 基本下不动。
改用阿里云的镜像（注意它是**扁平目录**，要用 `-f` 而不是 `-i`）：

```bash
pip install "torch==2.14.0+cu130" "torchvision==0.29.0+cu130" \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu130/ --timeout 120 --retries 10
```

装完必须做**三步验证**，第三步是关键：

```python
import torch
print(torch.__version__, torch.version.cuda)     # 2.14.0+cu130 13.0
print(torch.cuda.get_device_name(0))             # NVIDIA GeForce RTX 5060 Ti
a = torch.randn(1000, 1000, device='cuda')       # ← 真的算一次
print((a @ a).sum().item())
```

前两步只说明"识别到了显卡"，**只有第三步跑通才证明这个 wheel 里编了 sm_120 的 kernel**。
否则会抛 `no kernel image is available for execution on the device`。

### 3. batch_size 太小会把显卡饿死

CNN 图像分类一开始用 `batch_size=8`，每个 epoch 要 19 秒。实测把 batch 单独扫一遍：

| batch | 单步耗时 | 吞吐 | 一个 epoch |
| --- | --- | --- | --- |
| 8 | 1.82 ms | 4401 张/秒 | **19.17 s** |
| 16 | 2.07 ms | 7720 张/秒 | — |
| 64 | 2.10 ms | 30537 张/秒 | 3.80 s |
| 128 | 2.99 ms | 42759 张/秒 | 3.07 s |
| **256** | 5.68 ms | **45055 张/秒** | **2.17 s** |

注意 batch 从 8 加到 64，**单步耗时几乎不变**（1.82 → 2.10 ms）——
算 8 张图和算 64 张图一样慢，说明时间全花在"启动 CUDA kernel"这个固定开销上，
显卡根本没在算数。这就是 GPU 利用率低的典型长相。

**只改一个数字（`batch_size` 8 → 256），一个 epoch 从 19.17 s 降到 2.17 s，快 8.8 倍。**

### 4. Windows 上 DataLoader 的 `num_workers` 反而更慢

同一份代码，实测对比：

| 配置 | 一个 epoch |
| --- | --- |
| `num_workers=0` | 19.17 s |
| `num_workers=4` | **34.29 s**（更慢） |
| `num_workers=4` + `persistent_workers=True` | 第 1 个 epoch 16.52 s，之后 1.95 s |

原因是两点叠加：

1. CIFAR-10 的数据**本来就已经全在内存里**（`train_data.data` 是个 153 MB 的 numpy 数组），
   单进程读就有 24000 张/秒，开子进程没有任何收益
2. Windows 上 `num_workers>0` 走 **spawn**，每个子进程都要**重新 `import torch`**（约 2 秒），
   还要把**整个数据集 pickle 过去**（153 MB × 4 份），而且**每个 epoch 都重来一次**

所以这个场景下 `num_workers=0` 才是最优解。加 `persistent_workers=True` 可以让子进程只 spawn 一次，
但要跑 70 个 epoch 以上才回本。

### 5. DataLoader 最后一批不满，会让 `init_hidden(batch_size)` 崩掉

`DataLoader` 的 `drop_last` 默认是 `False`，所以**最后一批装不满**。实测（RNN 歌词生成）：

```
len(dataset) = 49118          (= 语料 49134 个词 - 序列长度 16)
batch_size   = 64
batch 总数    = 768
最后一批量    = 30
```

于是最后一批输入是 `(30, 16)`，而 hidden 是按固定 64 造的：

```
RuntimeError: Expected hidden size (1, 30, 256), got [1, 64, 256]
```

**这会在第一个 epoch 的最后一个 batch 才爆**，非常难查。正确写法是从数据里取真实批大小：

```python
hidden = model.init_hidden(x.size(0))    # 不是 init_hidden(batch_size)
```

### 6. `CrossEntropyLoss` 要求类别维在 dim=1

RNN 的输出是 `(batch, seq_len, vocab)`，直接喂给损失函数会报：

```
RuntimeError: Expected target size [4, 94], got [4, 6]
```

因为 `CrossEntropyLoss` 把 `(N, C, d1, ...)` 里的 **dim=1 当成类别维**，
所以它以为你有 6 个类别。两种改法（结果完全等价）：

```python
loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))   # 拍平成 2D（常用）
loss = criterion(y_pred.permute(0, 2, 1), y)                # 把类别维换到 dim=1
```

> **关键区分**：`batch_first=True` 管的是"RNN 的 batch 和 seq 谁在前"，
> `CrossEntropyLoss` 管的是"类别维在第几维"——**两件不相干的事，谁也替代不了谁**。
> `batch_first=True` 省掉的是进出 RNN 的两次 `transpose`，**省不掉 loss 的 reshape**。

另外：`view` 要求内存连续，`permute` 之后必须用 `reshape`：

```python
y_pred.permute(0, 2, 1).view(-1, vocab_size)      # ✗ RuntimeError: view size is not compatible
y_pred.permute(0, 2, 1).reshape(-1, vocab_size)   # ✓
```

### 7. RNN 的 hidden 每个 batch 都要重置（TBPTT）

每个 batch 重新 `init_hidden` 不是随手写的习惯，而是**截断时序反向传播（TBPTT）**的标准做法。
实测不重置的后果：

```
[不重置] loss 计算图节点数:  47 → 78 → 109 → 140 → 171   每批 +31，线性增长
[每批重置] loss 计算图节点数: 47 → 47 →  47 →  47 →  47   恒定
```

一个 epoch 768 个 batch，不重置的话最后一个 loss 的计算图上挂着 768 个 batch 的全部中间结果，
显存直接爆。而且梯度**真的会一路回传**——实测只对最后一个 loss 反传，第 0 个 hidden 拿到了梯度，
范数是 `1.76e-09`（才 5 层就已经梯度消失到十亿分之一）。

如果确实需要长程记忆，正确写法是**传 hidden 但 `detach()`**：

```python
hidden = model.init_hidden(x.size(0))
for x, y in dataLoader:
    y_pred, hidden = model(x, hidden)
    hidden = hidden.detach()      # 保留数值，断开计算图
    ...
```

实测这样计算图恒定在 47 个节点，但记忆能跨 batch 保留。**注意必须配合 `shuffle=False`**，
否则 batch 顺序随机，记忆接不上。

### 8. 其他小坑

- **matplotlib 中文显示成方框**：加 `plt.rcParams['font.sans-serif'] = ['SimHei']`
  和 `plt.rcParams['axes.unicode_minus'] = False`
- **`plt.plot` 画出绿色锯齿**：x 被打乱后折线按数组顺序连接。正确做法是用
  `torch.linspace` 画公式曲线，预测点先排序再画
- **累计 loss 曲线假性平滑**：`total_loss, total_samples` 的重置必须放在 **epoch 循环里面**
- **`nn.Linear(1, 2)` 用在回归任务上**：只会给一个 `UserWarning` 然后静默广播，MSE 完全算错
- **`torch.save` 不会自动建目录**：`model/` 目录必须提前存在

## 说明

- 代码为跟随课程逐步录入、逐个运行验证的练习，每个文件对应课程中的一个小节
- 涉及 matplotlib 绘图时已配置 `SimHei` 字体，保证中文标题与图例正常显示
- `data/` 与 `model/` 目录下的数据集和模型权重已随仓库提交，方便直接复现
- 课程资料版权归黑马程序员所有，本仓库仅存放个人练习代码

## 参考

- [黑马程序员《神经网络与深度学习》](https://www.bilibili.com/video/BV1c5yrBcEEX/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
- [PyTorch 中文文档](https://pytorch.cadn.net.cn/)
