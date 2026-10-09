# -*- coding: utf-8 -*-
"""
作业 (1) 配图：把三种滤波器核的系数画出来（报告图7）。

为什么单独一个脚本：只是想看清楚"核里到底装了什么数"，
正文脚本 01_图像滤波对比.py 只管滤波与差异度量，各管一件事。

运行
----
    cd 作业 && python 05_画滤波器核.py
输出
----
    结果图/fig7_滤波器核.png
    结果/核对齐与归一化.csv     —— 各核系数之和、是否对称、"核半径"占图宽比例
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "结果图")
RES_DIR = os.path.join(HERE, "结果")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


def box(k):
    return np.full((k, k), 1.0 / (k * k))


def gauss(k, sigma):
    r = k // 2
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    return g / g.sum()


KERNELS = [
    ("3x3 均值核（全部相等，箱子形状）", box(3)),
    ("5x5 均值核", box(5)),
    ("9x9 均值核", box(9)),
    ("7x7 高斯核 $\\sigma$=1.5（中间权重大）", gauss(7, 1.5)),
]


def axis_labels(k):
    """把行/列下标 0..k-1 翻译成相对核中心的偏移，例如 k=5 -> -2 -1 0 +1 +2。"""
    r = k // 2
    return ["0" if i == r else (f"+{i - r}" if i > r else f"{i - r}") for i in range(k)]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(15.6, 4.3))

    rows = []
    for ax, (name, K) in zip(axes, KERNELS):
        ax.imshow(K, cmap="Blues")
        ax.set_title(name, fontsize=11)
        k = K.shape[0]
        for i in range(k):
            for j in range(k):
                v = K[i, j]
                txt = f"{v:.3f}".rstrip("0").rstrip(".") if v < 1 else "1"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.2,
                        color="white" if v > K.max() * 0.7 else "black")
        lab = axis_labels(k)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels(lab, fontsize=8)
        ax.set_yticklabels(lab, fontsize=8)
        ax.set_xlabel("列偏移", fontsize=9, labelpad=1)
        ax.set_ylabel("行偏移", fontsize=9, labelpad=1)
        rows.append(dict(核=name, 边长k=k, 系数个数=k * k,
                         系数和=round(float(K.sum()), 8),
                         中心系数=round(float(K[k // 2, k // 2]), 6),
                         上下左右对称=bool(np.allclose(K, K[::-1, :]) and np.allclose(K, K[:, ::-1])),
                         半径占图宽=(k // 2) / 1279 * 100))

    fig.suptitle("图7  四种滤波器核的系数分布\n（坐标是相对核中心的偏移，系数之和都为 1 → 不改变图像整体亮度，只把邻域摊平）",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(FIG_DIR, "fig7_滤波器核.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"已保存 : {os.path.relpath(out, HERE)}")

    with open(os.path.join(RES_DIR, "核对齐与归一化.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['核']:<28} 系数和={r['系数和']}  对称={r['上下左右对称']}  "
              f"半径占图宽={r['半径占图宽']:.3f}%")

    print("\n要点：")
    print("  1) 四种核的系数之和都等于 1 —— 所以滤波不会整体变亮/变暗，只会把邻域摊平。")
    print("  2) 均值核在核内权重完全相同，对邻域一视同仁，所以边缘会产生台阶感。")
    print("  3) 高斯核中间权重大、四周衰减，越靠近中心越被信任，过渡更自然")
    print("     （图2 里 7x7 高斯的观感比 5x5/9x9 均值核更干净，而边缘保留率只有 42.0%）。")


if __name__ == "__main__":
    main()
