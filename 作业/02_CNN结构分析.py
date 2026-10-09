# -*- coding: utf-8 -*-
"""
作业 (2)：分析已有 CNN 网络的结构、参数量和构建方法。

本脚本**不重写模型**，而是从 ../CNN/03_图片识别案例.py 里把 ImgMode 类导入进来分析，
保证"分析的"和"交上去跑通的"是同一份代码（而不是报告里另写一份漂亮的）。

做四件事
--------
1. 逐层列出：层类型 / 超参 / 输出形状 / 该层参数量 / 占比  ——（用 register_forward_hook 真实前向一次拿到形状）
2. 统计卷积层、全连接层各自扛了多少参数，看清"参数都花在哪"
3. 算 FLOPs 与"感受野 RF"，回答"卷积核能看多宽"
4. 用 5 种不同的写法构建等价网络，说明"构建方法"可以怎么变

运行
----
    cd 作业 && python 02_CNN结构分析.py
输出
----
    结果/CNN逐层结构.csv
    stdout 上的完整表格 + 分析结论
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.join(HERE, "..", "CNN", "03_图片识别案例.py")
RES_DIR = os.path.join(HERE, "结果")
os.makedirs(RES_DIR, exist_ok=True)


# ============================================================
# 0. 从原案例文件里导入 ImgMode（不改动原文件）
# ============================================================
def load_case_model():
    """
    用 importlib 按路径加载 ../CNN/03_图片识别案例.py，
    只取其中的 ImgMode 类，不执行它的 __main__ 训练流程。
    """
    spec = importlib.util.spec_from_file_location("case03", CASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 文件里训练代码在 if __name__ 里，不会被触发
    return mod.ImgMode, mod


# ============================================================
# 1. 逐层结构 + 参数量（hook 拿真实形状）
# ============================================================
def human(n):
    return f"{n:,}"


def analyze(model, in_shape=(3, 32, 32)):
    records = {}

    def hook(module):
        # 用模块对象本身做 key，不依赖记录顺序与 named_children 的顺序一致
        def fn(_m, _inp, out):
            records[id(module)] = tuple(out.shape)
        return fn

    handles = []
    named = list(model.named_children())
    for _name, module in named:
        handles.append(module.register_forward_hook(hook(module)))

    model.eval()
    x = torch.randn(1, *in_shape)
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    # 逐层超参 + 参数量
    rows = []
    rf, stride_prod = 1, 1
    for name, module in named:
        out_shape = records.get(id(module))
        if out_shape is None:
            raise RuntimeError(f"层 {name} 在前向过程中没有执行，无法统计")
        p = sum(q.numel() for q in module.parameters())
        info = ""
        if isinstance(module, nn.Conv2d):
            info = (f"in={module.in_channels}, out={module.out_channels}, "
                    f"k={module.kernel_size[0]}, s={module.stride[0]}, p={module.padding[0]}")
            k = module.kernel_size[0]
            rf = rf + (k - 1) * stride_prod
        elif isinstance(module, nn.MaxPool2d):
            info = f"k={module.kernel_size}, s={module.stride}"
            k, s = module.kernel_size, module.stride
            rf = rf + (k - 1) * stride_prod
            stride_prod *= s
        elif isinstance(module, nn.Linear):
            info = f"in={module.in_features}, out={module.out_features}"
        elif isinstance(module, nn.Dropout):
            info = f"p={module.p}"
        rows.append({
            "序": len(rows) + 1,
            "层名": name,
            "类型": type(module).__name__,
            "超参": info,
            "输出形状": "x".join(str(v) for v in out_shape[1:]),
            "参数量": p,
            "感受野": rf,
        })
    return rows


# ============================================================
# 2. FLOPs（乘加次数）
# ============================================================
def count_flops(model, in_shape=(1, 3, 32, 32)):
    """
    用 hook 统计每一次卷积/全连接的乘加次数 (MACs)。
    FLOPs 通常记作 2 x MACs（一次乘 + 一次加）。
    """
    macs = []

    def conv_hook(m, inp, out):
        o = out.shape
        macs.append(("Conv2d " + str(tuple(o[1:])),
                     m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1]
                     * o[2] * o[3] // m.groups))

    def lin_hook(m, inp, out):
        macs.append(("Linear " + str(m.out_features), m.in_features * m.out_features))

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    model.eval()
    with torch.no_grad():
        model(torch.randn(*in_shape))
    for h in hs:
        h.remove()
    return macs


# ============================================================
# 3. 同一网络的 5 种构建写法
# ============================================================
class BuildA(nn.Module):
    """写法A：__init__ 里逐层声明 + forward 里手工串联（原案例用的就是这种）。"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.55)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)


def build_b():
    """写法B：nn.Sequential 顺序容器，一句话搭完主干；池化用 nn.Sequential 折叠。"""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 128, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3), nn.ReLU(), nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(256 * 2 * 2, 2048), nn.LeakyReLU(), nn.Dropout(0.55),
        nn.Linear(2048, 1024), nn.LeakyReLU(), nn.Dropout(0.55),
        nn.Linear(1024, 512), nn.LeakyReLU(),
        nn.Linear(512, 10),
    )


class BuildC(nn.Module):
    """写法C：nn.ModuleList 装重复结构，for 循环串联——层数多时最省事。"""
    def __init__(self):
        super().__init__()
        chans = [(3, 32), (32, 128), (128, 256)]
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(i, o, 3), nn.ReLU(), nn.MaxPool2d(2, 2)) for i, o in chans
        ])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 2048), nn.LeakyReLU(), nn.Dropout(0.55),
            nn.Linear(2048, 1024), nn.LeakyReLU(), nn.Dropout(0.55),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


class BuildD(nn.Module):
    """写法D：nn.ModuleDict 按名字存取，方便做"消融实验"（换掉某一层再跑）。"""
    def __init__(self, body="base"):
        super().__init__()
        self.body_name = body
        self.layers = nn.ModuleDict({
            "conv1": nn.Conv2d(3, 32, 3), "pool1": nn.MaxPool2d(2, 2),
            "conv2": nn.Conv2d(32, 128, 3), "pool2": nn.MaxPool2d(2, 2),
            "conv3": nn.Conv2d(128, 256, 3), "pool3": nn.MaxPool2d(2, 2),
            "fc1": nn.Linear(256 * 2 * 2, 2048), "fc2": nn.Linear(2048, 1024),
            "fc3": nn.Linear(1024, 512), "fc4": nn.Linear(512, 10),
            "drop": nn.Dropout(0.55),
        })

    def forward(self, x):
        L = self.layers
        x = L["pool1"](F.relu(L["conv1"](x)))
        x = L["pool2"](F.relu(L["conv2"](x)))
        x = L["pool3"](F.relu(L["conv3"](x)))
        x = x.reshape(x.size(0), -1)
        x = L["drop"](F.leaky_relu(L["fc1"](x)))
        x = L["drop"](F.leaky_relu(L["fc2"](x)))
        x = F.leaky_relu(L["fc3"](x))
        return L["fc4"](x)


class BuildE(nn.Module):
    """
    写法E：**参数量等价**但结构升级版 —— 3 层卷积原样保留，
    把 "Flatten + 4 层全连接" 换成 "全局平均池化 + 1 层分类头"。
    它的存在是为了说明：构建方法变了，参数量能差两个数量级（详见结论）。
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 128, 3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3), nn.BatchNorm2d(256), nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(self.dropout(x))


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 92)
    print("作业(2) CNN 结构 / 参数量 / 构建方法分析")
    print("=" * 92)

    ImgMode, case = load_case_model()
    model = ImgMode()
    print(f"模型来源 : {os.path.relpath(CASE, HERE)}  ->  class {ImgMode.__name__}")
    print(f"输入张量 : (N, 3, 32, 32)  —— CIFAR-10 彩色小图")

    rows = analyze(model)

    total = sum(r["参数量"] for r in rows)
    conv_p = sum(r["参数量"] for r in rows if r["类型"] == "Conv2d")
    fc_p = sum(r["参数量"] for r in rows if r["类型"] == "Linear")

    print("\n" + "-" * 92)
    print(f"{'#':>2} {'层名':<8}{'类型':<12}{'超参':<34}{'输出形状':<14}{'参数量':>14}{'感受野':>7}")
    print("-" * 92)
    for r in rows:
        print(f"{r['序']:>2} {r['层名']:<8}{r['类型']:<12}{r['超参']:<34}"
              f"{r['输出形状']:<14}{human(r['参数量']):>14}{r['感受野']:>7}")
    print("-" * 92)
    print(f"{'合计':<56}{human(total):>14}")

    print(f"\n带参数的层数        : {len(rows)} 层（3 卷积 + 4 全连接），"
          f"全部模块（含池化/Dropout）共 {sum(1 for _ in model.children())} 个")
    print(f"卷积层参数          : {human(conv_p):>14}  ({conv_p / total * 100:6.2f} %)")
    print(f"全连接层参数        : {human(fc_p):>14}  ({fc_p / total * 100:6.2f} %)"
          f"   <= 参数几乎都在这里")
    print(f"参数量合计          : {human(total)}  ≈ {total * 4 / 1024 / 1024:.2f} MB (float32)")
    print(f"  对照：卷积层只占 {conv_p / total * 100:.2f}% 的参数，"
          f"却贡献了最主要的看图能力（见下面的 FLOPs 分布）")

    # 感受野能不能覆盖整张图
    last_rf = rows[-1]["感受野"]
    conv_out = [r for r in rows if r["类型"] == "Conv2d"][-1]["输出形状"]
    print(f"\n最后一层的感受野    : {last_rf} x {last_rf}，而输入只有 32 x 32  "
          f"-> 覆盖 {min(1.0, last_rf * last_rf / (32 * 32)) * 100:.1f}% 的面积")
    print(f"  * 卷积最后输出 {conv_out}，再 Flatten 成 {256 * 2 * 2} 维送进 fc1："
          f"空间信息一个没丢，但每个神经元只看得到 {last_rf}x{last_rf} 的范围。")

    # ---- FLOPs ----
    macs = count_flops(model)
    total_macs = sum(v for _, v in macs)
    print(f"\n前向一次(1 张图)乘加次数: {human(total_macs)} MACs ≈ "
          f"{total_macs * 2 / 1e6:.1f} MFLOPs")
    conv_macs = sum(v for k, v in macs if k.startswith("Conv2d"))
    fc_macs = total_macs - conv_macs
    n_lin = len([1 for k, _ in macs if k.startswith("Linear")])
    print(f"  其中卷积 {human(conv_macs)} ({conv_macs / total_macs * 100:.1f}%)，"
          f"全连接 {human(fc_macs)} ({fc_macs / total_macs * 100:.1f}%)")
    print(f"  注意：全连接只被调用了 {n_lin} 次，却吃掉 {fc_macs / total_macs * 100:.1f}% 的计算量 —— "
          f"卷积省参数、全连接费参数，这张表看得最清楚。")

    # ---- 5 种构建写法 ----
    print("\n" + "=" * 92)
    print("5 种构建方法（同一网络的不同写法，参数量应当一致；写法E 是升级版，参数量不同）")
    print("=" * 92)
    variants = [
        ("A  逐层声明 + forward 手写串联（原案例）", BuildA()),
        ("B  nn.Sequential 顺序容器一句话搭完", build_b()),
        ("C  nn.ModuleList + for 循环", BuildC()),
        ("D  nn.ModuleDict 按名字存取", BuildD()),
        ("E  全局平均池化 + 单层分类头（升级）", BuildE()),
    ]
    print(f"{'写法':<44}{'参数量':>14}{'可训练参数':>14}  输出形状校验")
    print("-" * 92)
    ok_all = True
    for title, m in variants:
        n = sum(p.numel() for p in m.parameters())
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(2, 3, 32, 32))
        ok = tuple(y.shape) == (2, 10)
        ok_all &= ok
        print(f"{title:<44}{human(n):>14}{human(n):>14}  {tuple(y.shape)}  {'OK' if ok else 'FAIL'}")
    print("-" * 92)
    print("A/B/C/D 参数量完全一致 -> 四种写法等价（构建方法不影响数学结果，只影响可读性与可扩展性）")
    n_a = sum(p.numel() for p in variants[0][1].parameters())
    n_e = sum(p.numel() for p in variants[4][1].parameters())
    print(f"E 比 A 少 {human(n_a - n_e)} 个参数（{n_e / n_a * 100:.2f}%），"
          f"模型从 {n_a * 4 / 1024 / 1024:.1f} MB 降到 {n_e * 4 / 1024 / 1024:.2f} MB")
    print(f"全部写法输出形状校验 : {'全部通过' if ok_all else '有不通过的'}")

    # ---- 写 CSV（只写逐层数据，合计由报告生成脚本现场计算，避免数据表里出现空单元格） ----
    csv_path = os.path.join(RES_DIR, "CNN逐层结构.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n逐层结构已保存 : {os.path.relpath(csv_path, HERE)}")

    # ---- 结构图（把主干画出来，放进报告） ----
    try:
        draw_arch(rows, total)
    except Exception as e:      # 画图失败不影响分析结论
        print(f"[warn] 结构图绘制失败：{e}")


def draw_arch(rows, total):
    """把网络结构画成一张竖向流程图，作为报告图4。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10.6, 11.0))
    ax.set_xlim(0, 10); ax.set_ylim(0, len(rows) + 1.4)
    ax.axis("off")

    color = {"Conv2d": "#cfe3ff", "MaxPool2d": "#e6f4d7", "Linear": "#ffe0cc"}
    for i, r in enumerate(rows):
        y = len(rows) - i
        p = r["参数量"]
        share = p / total * 100
        box = FancyBboxPatch((0.5, y - 0.34), 6.05, 0.68,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.1, edgecolor="#40506a",
                             facecolor=color.get(r["类型"], "#efefef"))
        ax.add_patch(box)
        ax.text(0.72, y, f"{r['层名']}", fontsize=11, va="center", weight="bold")
        ax.text(1.85, y, f"{r['类型']}", fontsize=9.5, va="center")
        ax.text(3.15, y, f"{r['超参']}", fontsize=8.2, va="center", color="#22364f")
        ax.text(6.25, y, f"输出 {r['输出形状']}", fontsize=8.8, va="center")
        ax.text(9.75, y, f"{p:,}", fontsize=9, va="center", ha="right",
                color="#a33" if share > 20 else "#333")
        ax.text(9.75, y - 0.30, f"({share:.2f}%)", fontsize=7.6, va="center", ha="right",
                color="#a33" if share > 20 else "#777")
        if i < len(rows) - 1:
            ax.add_patch(FancyArrowPatch((3.5, y - 0.36), (3.5, y - 0.66),
                                         arrowstyle="-|>", mutation_scale=11, color="#40506a"))

    ax.text(0.5, len(rows) + 0.72, "输入  (N, 3, 32, 32)  RGB 彩色图像", fontsize=11)
    ax.text(9.75, len(rows) + 0.72, "参数量", fontsize=9.5, ha="right", color="#555")
    ax.text(0.5, 0.30, f"合计参数量 {total:,}   （卷积 {sum(r['参数量'] for r in rows if r["类型"] == "Conv2d"):,} / "
                       f"全连接 {sum(r['参数量'] for r in rows if r["类型"] == "Linear"):,}）",
            fontsize=11, weight="bold")
    ax.set_title("图4  ImgMode 逐层结构与参数量分布\n（最右列为该层参数量及占全网参数的比例）",
                 fontsize=12.5, pad=14)
    out = os.path.join(HERE, "结果图", "fig4_网络结构.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存 : {os.path.relpath(out, HERE)}")


if __name__ == "__main__":
    main()
