# -*- coding: utf-8 -*-
"""
算三种结构的感受野，给报告里"感受野对比表"提供真实数字。
感受野递推：RF_out = RF_in + (k - 1) * stride_prod；遇到 stride>1 时 stride_prod *= s。
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT = 32


def trace(model, name, in_shape=(3, 32, 32)):
    """跑一次前向，逐层累计感受野。"""
    rf, sp = 1, 1
    rows = []
    handles = []

    def hook(m):
        def fn(_m, _i, out):
            nonlocal rf, sp
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0]
                rf = rf + (k - 1) * sp
                sp *= m.stride[0]
            elif isinstance(m, nn.MaxPool2d):
                k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
                s = m.stride if isinstance(m.stride, int) else m.stride[0]
                rf = rf + (k - 1) * sp
                sp *= s
            rows.append((type(m).__name__, tuple(out.shape[1:]), rf))
        return fn

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d)):
            handles.append(m.register_forward_hook(hook(m)))
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, *in_shape))
    for h in handles:
        h.remove()

    final_rf = rows[-1][2] if rows else 1
    cover = min(1.0, (final_rf / INPUT) ** 2) * 100
    n_par = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} ===")
    print(f"  参数量 {n_par:,}   末层感受野 {final_rf}x{final_rf}   "
          f"对 {INPUT}x{INPUT} 输入的覆盖率 {cover:.1f}%")
    for t, sh, r in rows:
        print(f"    {t:<12} 输出 {str(sh):<14} 感受野 {r}")
    return dict(name=name, params=n_par, rf=final_rf, cover=cover)


# 1) 原模型：3 个 (conv+pool)
class Orig(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 128, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(256 * 2 * 2, 2048), nn.Linear(2048, 1024),
            nn.Linear(1024, 512), nn.Linear(512, 10))

    def forward(self, x):
        return self.m(x)


# 2) 换成 stride=2 卷积，不做池化（少一次下采样）
class Strided(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(), nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3), nn.ReLU(), nn.Conv2d(128, 256, 3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(256 * 5 * 5, 10))

    def forward(self, x):
        return self.m(x)


# 3) 改进模型：每个 stage 两个卷积 + 三个池化 + 全局平均池化
class Improved(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10))

    def forward(self, x):
        return self.f(x)


res = [trace(Orig(), "原模型 ImgMode（3 段 卷积+池化）"),
       trace(Strided(), "改用 stride=2 卷积（只有 2 次下采样）"),
       trace(Improved(), "改进模型（6 卷积 + 3 池化 + 全局平均池化）")]

print("\n" + "=" * 70)
print(f"{'结构':<40}{'参数量':>12}{'感受野':>10}{'覆盖率':>10}")
print("-" * 70)
for r in res:
    print(f"{r['name']:<40}{r['params']:>12,}{r['rf']:>10}{r['cover']:>9.1f}%")
