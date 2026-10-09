# -*- coding: utf-8 -*-
"""
作业 (1)：选取任意不同大小的两个（本脚本给出三个）滤波器核，实现图像滤波，
并比较两幅图像的差异。

设计要点
--------
1. 任意彩色图像：默认读取 ../CNN/data/1.jpg（1279x1706，满足"不小于 100x100"）。
2. 滤波**不使用任何库的"黑盒滤波函数"**（不用 cv2.blur / cv2.GaussianBlur /
   scipy.ndimage.convolve），而是用 numpy 自己实现"滑动窗口 + 逐点乘加"，
   把卷积的定义写清楚，便于讲清楚"核大小改变了什么"。
3. 对比的通道：灰度图（亮度）与彩色图（按通道分别滤波）都做。
4. 差异度量：
        MAE  平均绝对误差      —— 平均每个像素被"改动"了多少（0~255）
        PSNR 峰值信噪比        —— 越小说明与原图差得越多
        梯度能量 / 边缘保留率  —— 核越大，高频（细节）被抹掉得越多
        与核大小的关系         —— 量化"模糊半径"

运行
----
    cd 作业 && python 01_图像滤波对比.py
输出
----
    结果图/fig1_滤波对比.png     原图 / 各核滤波结果 / 差值图
    结果图/fig2_局部放大.png     高频区域的 4 倍局部放大对比
    结果图/fig3_指标曲线.png     各指标随核大小的变化
    结果/滤波指标.csv           全部量化结果
    stdout                       控制台表格
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # 见报告 5.1：规避 OpenMP 冲突

import sys
import csv
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ---------- 中文显示（见报告 5.8） ----------
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.path.join(HERE, "..", "CNN", "data", "1.jpg")
FIG_DIR = os.path.join(HERE, "结果图")
RES_DIR = os.path.join(HERE, "结果")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


# ============================================================
# 1. 自己实现二维卷积（"滑动窗口 + 逐点乘加"）
# ============================================================
def conv2d_channel(img2d, kernel, padding="same"):
    """
    单通道二维卷积。img2d: (H, W) float64；kernel: (k, k)。

    padding="same"  —— 核关于中心对称时，先对原图做零填充再卷积，输出与原图同尺寸。
                       零填充会让图像最外圈的像素偏暗（见报告 5.x 的"边界效应"），
                       所以下面用 padding="valid" 的裁剪版本报告指标。
    padding="valid" —— 不做任何填充，只计算"核能完整覆盖"的位置，
                       输出尺寸 (H-k+1, W-k+1)。卷积是严格定义，没有边界假设。
    """
    k = kernel.shape[0]
    assert kernel.shape == (k, k), "本脚本只处理方核"
    H, W = img2d.shape

    if padding == "same":
        p = k // 2
        padded = np.pad(img2d, ((p, p), (p, p)), mode="constant")
        out = np.zeros((H, W), dtype=np.float64)
        for i in range(k):
            for j in range(k):
                # 用切片一次算完"核系数 (i,j) 在所有位置的贡献"
                out += kernel[i, j] * padded[i:i + H, j:j + W]
        return out

    # valid：只取完整覆盖的位置
    oh, ow = H - k + 1, W - k + 1
    out = np.zeros((oh, ow), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            out += kernel[i, j] * img2d[i:i + oh, j:j + ow]
    return out


def filter_image(img, kernel, mode="valid"):
    """
    对彩色图像 (H, W, 3) 逐通道滤波。mode='valid' 时三个通道输出同尺寸。
    """
    chs = [conv2d_channel(img[:, :, c], kernel, padding=mode) for c in range(img.shape[2])]
    return np.stack(chs, axis=2)


# ============================================================
# 2. 三种核（"任意不同大小的两个滤波器核"，这里给三个）
# ============================================================
def kernel_box(k):
    """均值（盒式）核：k x k，每个系数 1/k²，系数和为 1。k 必须是奇数。"""
    assert k % 2 == 1, "k 取奇数，核才有唯一的中心"
    return np.full((k, k), 1.0 / (k * k), dtype=np.float64)


def kernel_gauss(k, sigma):
    """高斯核：系数按二维高斯分布取样后归一化，系数和为 1。"""
    assert k % 2 == 1
    r = k // 2
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(xs ** 2 + ys ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


KERNELS = [
    ("3x3 均值核", kernel_box(3), "3x3"),
    ("5x5 均值核", kernel_box(5), "5x5"),
    ("9x9 均值核", kernel_box(9), "9x9"),
    ("7x7 高斯核 sigma=1.5", kernel_gauss(7, 1.5), "7x7G"),
]


# ============================================================
# 3. 差异度量
# ============================================================
def mae(a, b):
    """平均绝对误差：平均每个通道值被改动了多少（0~255 量纲）。"""
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a, b):
    """峰值信噪比(dB)。差值越大 PSNR 越小。完全相同时返回 inf。"""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def gray_of(img):
    """按亮度权重合成灰度图（0.299R + 0.587G + 0.114B）。"""
    w = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    return img.astype(np.float64) @ w


def gradient_energy(gray):
    """
    梯度能量：用相邻像素差分近似 |∂I/∂x| + |∂I/∂y| 的总和。
    高频（边缘、纹理、噪点）越多，这个值越大。滤波把它压掉多少，就是"模糊"多少。
    """
    gy = np.abs(np.diff(gray, axis=0)).sum()
    gx = np.abs(np.diff(gray, axis=1)).sum()
    return float(gx + gy)


# ============================================================
# 4. 主流程
# ============================================================
def main():
    print("=" * 78)
    print("作业(1) 图像滤波：不同大小滤波器核的对比")
    print("=" * 78)

    img = np.asarray(Image.open(IMG_PATH).convert("RGB"), dtype=np.float64)
    H, W, _ = img.shape
    print(f"输入图像 : {os.path.relpath(IMG_PATH, HERE)}")
    print(f"图像尺寸 : {W} x {H} x 3  (宽x高x通道)  —— 不小于 100x100 ✓")

    gray = gray_of(img)
    base_ge = gradient_energy(gray)
    print(f"原图梯度能量 : {base_ge:.4e}   （作为 100% 基准）")

    rows = []
    results = {}

    for name, K, tag in KERNELS:
        k = K.shape[0]
        t0 = time.perf_counter()
        out_color = filter_image(img, K, mode="valid")      # 彩色：三通道分别滤波
        t1 = time.perf_counter()
        out_gray = conv2d_channel(gray, K, padding="valid")  # 灰度：直接滤亮度
        t2 = time.perf_counter()

        # 为"同尺寸"比较与画图，valid 结果按差值的整数位移对齐到原图
        p = k // 2
        ref = img[p:H - p, p:W - p, :]
        ref_gray = gray[p:H - p, p:W - p]

        m_color = mae(out_color, ref)
        m_gray = mae(out_gray, ref_gray)
        p_color = psnr(out_color, ref)
        ge = gradient_energy(out_gray)
        keep = ge / base_ge * 100.0

        rows.append(dict(
            核=name, k=k,
            彩色MAE=round(m_color, 4),
            灰度MAE=round(m_gray, 4),
            彩色PSNR_dB=round(p_color, 3),
            边缘保留率=round(keep, 3),
            彩色耗时ms=round((t1 - t0) * 1000, 1),
            灰度耗时ms=round((t2 - t1) * 1000, 1),
        ))
        results[name] = dict(kernel=K, color=out_color, gray=out_gray, keep=keep)

        print(f"\n[{name}]  输出尺寸 {out_color.shape[1]}x{out_color.shape[0]}"
              f"  (valid 卷积，比原图小 {k - 1} 像素)")
        print(f"    彩色图 MAE  = {m_color:8.4f}   PSNR = {p_color:6.3f} dB")
        print(f"    灰度图 MAE  = {m_gray:8.4f}")
        print(f"    边缘保留率  = {keep:7.3f} %   （原图梯度能量的百分比，越小越模糊）")
        print(f"    耗时        : 彩色 {(t1 - t0) * 1000:.1f} ms / 灰度 {(t2 - t1) * 1000:.1f} ms")

    # ---------- 写 CSV ----------
    csv_path = os.path.join(RES_DIR, "滤波指标.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n指标已保存 : {os.path.relpath(csv_path, HERE)}")

    # ---------- 核心结论 ----------
    print("\n" + "=" * 78)
    print("结论（用数据说话）")
    print("=" * 78)
    a, b = rows[0], rows[1]
    print(f"3x3 → 5x5：核面积从 {a['k']**2} 增到 {b['k']**2}（{b['k']**2/a['k']**2:.2f} 倍），"
          f"彩色 MAE {a['彩色MAE']} → {b['彩色MAE']}（{b['彩色MAE']/a['彩色MAE']:.2f} 倍），")
    print(f"           PSNR {a['彩色PSNR_dB']} → {b['彩色PSNR_dB']} dB，"
          f"边缘保留率 {a['边缘保留率']}% → {b['边缘保留率']}%（掉了 {a['边缘保留率']-b['边缘保留率']:.1f} 个百分点）")
    print("也就是说：核越大 → 与原图差得越多 → 高频细节被抹掉越多 → 图像越模糊。")

    # ============================================================
    # 图 1：原图 / 各核结果 / 差值图
    # ============================================================
    show = Image.open(IMG_PATH).convert("RGB")
    show.thumbnail((520, 520), Image.LANCZOS)
    show = np.asarray(show, dtype=np.float64)

    order = [("原图", None)] + [(n, results[n]) for n, _, _ in KERNELS]
    ncol = len(order)
    # 原图是竖构图（1279x1706，宽高比 0.75），子图给足高度，避免大片留白
    fig, axes = plt.subplots(3, ncol, figsize=(2.95 * ncol, 12.0))

    for j, (title, res) in enumerate(order):
        ax = axes[0][j]
        if res is None:
            ax.imshow(show.astype(np.uint8))
            ax.set_title("原图\n(裁到可与 valid 结果对齐的范围)", fontsize=11)
        else:
            k = res["kernel"].shape[0]
            p = k // 2
            cut = img[p:H - p, p:W - p]
            im = Image.fromarray(np.clip(cut, 0, 255).astype(np.uint8))
            im.thumbnail((560, 760), Image.LANCZOS)
            ax.imshow(np.asarray(im))
            ax.set_title(f"{title}\n滤波结果（彩色，逐通道）", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("auto")

    for j, (title, res) in enumerate(order):
        ax = axes[1][j]
        if res is None:
            ax.imshow(np.clip(show * 0.35 + 60, 0, 255).astype(np.uint8))
            ax.set_title("（原图本身差值为 0）", fontsize=10)
        else:
            k = res["kernel"].shape[0]
            p = k // 2
            ref = img[p:H - p, p:W - p]
            diff = np.abs(res["color"] - ref).max(axis=2)   # 三通道取最大差值，避免正负抵消
            dm = Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8))
            dm.thumbnail((560, 760), Image.NEAREST)
            ax.imshow(np.asarray(dm), cmap="inferno", vmin=0, vmax=60)
            ax.set_title("|滤波-原图| 差值图\n(色标上限 60)\nMAE={:.2f}".format(rows[j - 1]["彩色MAE"]),
                         fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("auto")

    for j, (title, res) in enumerate(order):
        ax = axes[2][j]
        if res is None:
            g = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
            g.thumbnail((560, 760), Image.LANCZOS)
            ax.imshow(np.asarray(g), cmap="gray")
            ax.set_title("原图灰度（0.299R+0.587G+0.114B）", fontsize=10)
        else:
            g = Image.fromarray(np.clip(res["gray"], 0, 255).astype(np.uint8))
            g.thumbnail((560, 760), Image.LANCZOS)
            ax.imshow(np.asarray(g), cmap="gray")
            ax.set_title("灰度滤波结果\n高频(边缘)保留率 = {:.1f}%".format(res["keep"]), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("auto")

    fig.suptitle("图1  不同大小滤波器核的滤波结果与差异对比\n（第一行：彩色滤波结果　第二行：与原图的差值图　第三行：灰度滤波结果）",
                 fontsize=13, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    f1 = os.path.join(FIG_DIR, "fig1_滤波对比.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    print(f"已保存 : {os.path.relpath(f1, HERE)}")

    # ============================================================
    # 图 2：高频区域的 4 倍局部放大
    # ============================================================
    # 在图像中部找一个"细节最密集"的窗口：用盒式滑动平均统计局部梯度能量，
    # 取能量最大的位置（照片的平坦区域如天空/墙面不是好样本，所以要按局部能量挑）。
    win, stride = 96, 16
    gy_ = np.abs(np.diff(gray, axis=0, prepend=gray[:1]))
    gx_ = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    detail = gx_ + gy_

    def box_sum(a, n):
        """用积分图求 a 的 n x n 窗口和，返回 (H-n+1, W-n+1)。"""
        ii = np.cumsum(np.cumsum(a, axis=0), axis=1)
        ii = np.pad(ii, ((1, 0), (1, 0)))
        return ii[n:, n:] - ii[:-n, n:] - ii[n:, :-n] + ii[:-n, :-n]

    sums = box_sum(detail, win)
    # 只在中间 20%~90% 的范围内找，避开照片边缘（避免选到边框/纯背景）
    y0, y1 = int(H * 0.15), int(H * 0.90)
    x0, x1 = int(W * 0.15), int(W * 0.90)
    sub = sums[y0:y1 - win, x0:x1 - win]
    sy, sx = np.unravel_index(np.argmax(sub), sub.shape)
    by, bx = y0 + int(sy), x0 + int(sx)
    by = by // stride * stride
    bx = bx // stride * stride
    peak = float(sub.max() / (win * win))
    print(f"局部放大区域 : 左上角 ({bx}, {by})，窗口 {win}x{win}，"
          f"窗口内平均梯度 {peak:.1f}（全图平均 {detail.mean():.1f}）")

    fig, axes = plt.subplots(1, len(order) + 1, figsize=(4.05 * (len(order) + 1), 5.2))
    crop_o = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).crop((bx, by, bx + win, by + win))
    axes[0].imshow(np.asarray(crop_o.resize((win * 4, win * 4), Image.NEAREST)))
    axes[0].set_title("原图局部 4x 放大\n(细节最密集的窗口)", fontsize=11)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for j, (name, K, tag) in enumerate(KERNELS, start=1):
        full = filter_image(img, K, mode="same")       # same 模式，便于按原坐标裁剪
        c = Image.fromarray(np.clip(full[by:by + win, bx:bx + win], 0, 255).astype(np.uint8))
        axes[j].imshow(np.asarray(c.resize((win * 4, win * 4), Image.NEAREST)))
        axes[j].set_title("{}\n4x 放大".format(name), fontsize=11)
        axes[j].set_xticks([]); axes[j].set_yticks([])

    fig.suptitle("图2  细节最密集区域的 4 倍放大：核越大，纹理/笔画被抹得越平（注意 same 模式的外圈边界效应）",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    f2 = os.path.join(FIG_DIR, "fig2_局部放大.png")
    fig.savefig(f2, dpi=160)
    plt.close(fig)
    print(f"已保存 : {os.path.relpath(f2, HERE)}")

    # ============================================================
    # 图 3：指标随核大小的变化
    # ============================================================
    ks = [r["k"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    axes[0].plot(ks, [r["彩色MAE"] for r in rows], "o-", color="tab:red", label="彩色 MAE")
    axes[0].plot(ks, [r["灰度MAE"] for r in rows], "s--", color="tab:blue", label="灰度 MAE")
    axes[0].set_xlabel("核边长 k"); axes[0].set_ylabel("平均绝对误差 MAE")
    axes[0].set_title("MAE 随核大小单调增大", fontsize=11)
    axes[0].set_xticks(ks); axes[0].grid(alpha=0.3); axes[0].legend()

    axes[1].plot(ks, [r["彩色PSNR_dB"] for r in rows], "^-", color="tab:green")
    axes[1].set_xlabel("核边长 k"); axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("PSNR 随核大小下降（越差越低）", fontsize=11)
    axes[1].set_xticks(ks); axes[1].grid(alpha=0.3)

    axes[2].plot(ks, [r["边缘保留率"] for r in rows], "D-", color="tab:purple")
    axes[2].set_xlabel("核边长 k"); axes[2].set_ylabel("边缘保留率 (%)")
    axes[2].set_title("边缘/细节保留率随核大小下降", fontsize=11)
    axes[2].set_xticks(ks); axes[2].grid(alpha=0.3)

    fig.suptitle("图3  三项指标随滤波器核大小的变化（横轴 k=3,5,9，另加 7x7 高斯核对照）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    f3 = os.path.join(FIG_DIR, "fig3_指标曲线.png")
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    print(f"已保存 : {os.path.relpath(f3, HERE)}")
    print("\n完成。")


if __name__ == "__main__":
    main()
