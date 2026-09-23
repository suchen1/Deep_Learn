# Deep Learn

跟随 **黑马程序员《神经网络与深度学习》** 课程的学习代码与笔记。

- 课程地址：<https://www.bilibili.com/video/BV1c5yrBcEEX/>
- 课程规模：共 108 P，总时长约 31.7 小时
- 内容覆盖：PyTorch 框架 → ANN 人工神经网络 → CNN 图像分类 → RNN 文本生成

本仓库是个人学习记录，代码按课程小节逐个练习整理，用于复习和查漏补缺。

## 环境

| 项 | 版本 / 说明 |
| --- | --- |
| Python | 3.14.6 |
| PyTorch | 2.14.0+cpu |
| NumPy | 2.4.6 |
| scikit-learn | 1.9.0 |
| Matplotlib | 3.11.0 |
| torchsummary | 1.5.1 |
| 解释器 | Anaconda（`C:\ProgramData\anaconda3`） |
| IDE | PyCharm |

> **注意**：当前安装的是 **CPU 版** PyTorch，`torch.cuda.is_available()` 返回 `False`。
> 学习张量、神经网络部分完全够用；到 Day06 的 CNN 图像分类实战时训练速度会明显偏慢，届时可考虑改装 CUDA 版。

## 目录结构

```
Deep Learn/
├── 张量/          # Day01–Day03  张量创建、运算、索引、形状、拼接、自动微分、线性回归案例（12 个文件）
├── 神经网络/       # Day04        参数初始化、神经网络搭建（2 个文件）
├── 损失函数/       # Day04        多分类/二分类交叉熵、MAE、MSE、SmoothL1（5 个文件）
├── 梯度下降/       # Day05        指数移动加权平均、动量法、AdaGrad、RMSProp、Adam（5 个文件）
├── 学习率衰减/     # Day05        等间隔、指定间隔、指数衰减（3 个文件）
├── 正则化/         # Day05        dropout、批量归一化（进行中）
├── main.py
└── 课程进度.md     # 108 P 完整课程地图与进度勾选清单
```

后续会按课程推进新增目录：`ANN案例/`、`CNN/`、`RNN/`。

## 学习进度

完整的分集进度见 [`课程进度.md`](课程进度.md)。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| Day01 | 深度学习概述 + PyTorch 张量创建 | ✅ 已完成 |
| Day02 | 张量进阶（运算 / 索引 / 形状 / 拼接）+ 自动微分 | ✅ 已完成 |
| Day03 | 线性回归 + 神经网络入门 + 激活函数 | ✅ 已完成 |
| Day04 | 网络搭建 + 损失函数 + 反向传播 | ✅ 已完成 |
| Day05 | 优化器 + 学习率衰减 + 正则化 + ANN 案例 | 🔵 进行中 |
| Day06 | ANN 案例收尾 + CNN 卷积与池化 | ⬜ 未开始 |
| Day07 | CNN 图像分类实战 + RNN 文本生成 | ⬜ 未开始 |

> 当前进度：**Day05 P68 已完成**（优化器、学习率衰减），正在学 P69 dropout 正则化。

## 运行方式

命令行直接运行单个脚本：

```bash
python 张量/01_张量基本创建方式.py
```

或在 PyCharm 中打开对应文件，按 `Shift+F10` 运行。

所有脚本均**不需要下载外部数据集**，数据由 `sklearn.datasets` 现场生成或随机初始化，克隆下来即可直接运行。

依赖随课程推进逐步增加：

| 依赖 | 从哪个文件开始需要 |
| --- | --- |
| `torch` | `张量/01` 起 |
| `numpy` | `张量/05` 起 |
| `scikit-learn` | `张量/12` 起（生成回归数据集、对照闭式解） |
| `matplotlib` | `张量/12` 起（损失曲线、拟合结果可视化） |
| `torchsummary` | `神经网络/02` 起（打印模型结构与参数量） |

## 说明

- 代码为跟随课程逐步录入并验证的练习，每个文件对应课程中的一个小节
- 环境为 CPU 版本，无 GPU 相关依赖，可在任意装有 PyTorch 的机器上运行
- 涉及 matplotlib 绘图时已配置 `SimHei` 字体，保证中文标题与图例正常显示
- 课程资料版权归黑马程序员所有，本仓库仅存放个人练习代码

### 已知环境问题：OpenMP 冲突

本机同时存在**两份** `libiomp5md.dll`：

```
torch\lib\libiomp5md.dll            ← PyTorch 自带（1.54 MB）
Library\bin\libiomp5md.dll          ← Anaconda 的 intel-openmp 包（1.25 MB）
```

当 matplotlib 启用 **Qt 交互式后端**真的去绘图时，两份运行时会在同一进程内冲突，
报 `OMP: Error #15` 并**直接终止进程**（`plt.show()` 根本执行不到，所以图不显示）。

`梯度下降/` 和 `学习率衰减/` 下的脚本已在文件开头加了绕过：

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # 必须在 import torch 之前
```

**新建的绘图脚本如果报同样的错，照抄这两行即可。**

> 根治办法是让环境里只保留一份 OpenMP 运行时（或给深度学习单开一个干净的 conda 环境），
> 但当前绕过方式对本项目的简单绘图任务风险极低。

## 参考

- [黑马程序员《神经网络与深度学习》](https://www.bilibili.com/video/BV1c5yrBcEEX/)
- [PyTorch 官方文档](https://pytorch.org/docs/stable/index.html)
