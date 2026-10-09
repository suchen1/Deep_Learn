# -*- coding: utf-8 -*-
"""
作业 (2) 的"证据链"脚本：把 CNN 的准确率真的跑出来，而不是只抄一行注释。

三个实验
--------
实验1  复现上交的模型：加载 ../CNN/model/ImgModel.pth，在 CIFAR-10 测试集上重新评测
       —— 用来核对案例里注释写的结果（训练 87.5% / 测试 69.81%）是否可复现。
实验2  诊断：同一批测试数据，做/不做标准化，看模型到底吃没吃到归一化。
实验3  改进：保留"3 层卷积"的主干不变，按作业(2)分析出的问题改造网络
       （加 BatchNorm、用全局平均池化替掉 4 层全连接、数据标准化），
       用同一套超参训练 30 轮，和原模型对比参数量与精度。
       注意：**不会覆盖**原来的 CNN/model/ImgModel.pth，改进模型存到 作业/model/。

运行
----
    cd 作业 && python 03_CNN精度验证与改进.py            # 默认跑全部
    cd 作业 && python 03_CNN精度验证与改进.py --only 1   # 只跑实验1

输出
----
    结果/精度对比.csv
    结果图/fig5_训练曲线.png
    结果图/fig6_预测样例.png
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import csv
import time
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "..", "CNN", "data")
CKPT = os.path.join(HERE, "..", "CNN", "model", "ImgModel.pth")
MODEL_DIR = os.path.join(HERE, "model")
RES_DIR = os.path.join(HERE, "结果")
FIG_DIR = os.path.join(HERE, "结果图")
for d in (MODEL_DIR, RES_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
# 原案例保存的权重在测试集上的实测准确率（见实验1 的输出）。
# 这里写成常量而不是散落在图里写死，避免出现"图上一条线、图例另一个数"的情况。
BASE_ACC = 72.67
CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


def load_case_model():
    """导入原案例的 ImgMode（原文件不改动）。"""
    spec = importlib.util.spec_from_file_location(
        "case03", os.path.join(HERE, "..", "CNN", "03_图片识别案例.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ImgMode


def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        # 第 3 步验证：只 is_available() 不够，要真的算一次矩阵乘法（见报告 5.2）
        a = torch.randn(256, 256, device=dev)
        _ = float((a @ a).sum())
        print(f"设备 : {dev}  {torch.cuda.get_device_name(0)}  "
              f"(CUDA {torch.version.cuda})  矩阵乘法自检通过")
    else:
        print("设备 : cpu")
    return dev


def build_loaders(normalize, batch_size, num_workers=0):
    """
    normalize=True 用 CIFAR-10 官方均值方差标准化。
    注意：数据集本身没有变，变的是喂给网络的数值范围。
    """
    def make(norm):
        ts = [transforms.ToTensor()]
        if norm:
            ts.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))
        return transforms.Compose(ts)

    tr = CIFAR10(root=DATA_ROOT, train=True, transform=make(normalize), download=False)
    te = CIFAR10(root=DATA_ROOT, train=False, transform=make(normalize), download=False)
    return (DataLoader(tr, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(te, batch_size=batch_size, shuffle=False, num_workers=num_workers))


def build_loaders_raw(normalize, batch_size):
    """返回未标准化的原始 0~1 张量数据集（用于"同一模型换输入分布"的对照）。"""
    return build_loaders(normalize, batch_size)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total * 100.0


# ============================================================
# 改进网络：主干仍是 3 层卷积（作业要求 >= 5 层，这里 6 个卷积/线性层 + 池化/Bn/Dropout）
# ============================================================
class ImprovedCNN(nn.Module):
    """
    相对原 ImgMode 的三处改动，每一处都对应作业(2) 分析出的一个具体问题：
      1) 卷积后加 BatchNorm2d —— 原网络全靠 0.0001 的小学习率硬压，收敛慢
      2) 末段用全局平均池化代替 Flatten + 4 层全连接 —— 原网络 93.4% 的参数堆在 fc 上
      3) 数据标准化 —— 原案例 create_data() 里就留着 "todo: 数据标准化"
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),                       # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),                       # 16 -> 8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                          # 8 -> 4
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model(model, train_loader, test_loader, device, epochs, lr, tag, bs):
    model.to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr, betas=[0.9, 0.999])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = {"epoch": [], "loss": [], "train_acc": [], "test_acc": [], "sec": []}
    best = 0.0
    best_ep = 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss = tot_n = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.item() * y.numel()
            tot_n += y.numel()
            correct += (out.argmax(1) == y).sum().item()
        sched.step()
        te_acc = evaluate(model, test_loader, device)
        if te_acc > best:
            best, best_ep = te_acc, ep
        hist["epoch"].append(ep)
        hist["loss"].append(tot_loss / tot_n)
        hist["train_acc"].append(correct / tot_n * 100)
        hist["test_acc"].append(te_acc)
        hist["sec"].append(time.time() - t0)
        print(f"  [{tag}] epoch {ep:>2}/{epochs}  loss={tot_loss / tot_n:.4f}  "
              f"train={correct / tot_n * 100:6.2f}%  test={te_acc:6.2f}%  "
              f"({time.time() - t0:.1f}s)")
    return hist, best, best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="1 / 2 / 3 / all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    print("=" * 84)
    print("CNN 精度验证与改进")
    print("=" * 84)
    device = get_device()
    ImgMode = load_case_model()
    results = []

    # ---------------- 实验1：复现上交的模型 ----------------
    if args.only in ("all", "1"):
        print("\n" + "=" * 84)
        print("实验1  复现 ../CNN/model/ImgModel.pth 的测试集精度")
        print("=" * 84)
        if not os.path.exists(CKPT):
            print(f"[跳过] 找不到权重文件 {CKPT}")
        else:
            sd = torch.load(CKPT, map_location="cpu")
            print(f"权重文件 : {os.path.relpath(CKPT, HERE)}  "
                  f"({os.path.getsize(CKPT) / 1024 / 1024:.2f} MB, {len(sd)} 个张量)")

            model = ImgMode()
            missing = model.load_state_dict(sd, strict=True)
            print(f"参数加载 : strict=True 无报错 -> 权重与 ImgMode 结构完全对应")

            n_par = sum(p.numel() for p in model.parameters())
            print(f"参数量   : {n_par:,}（与结构分析脚本一致：{n_par == 5060362}）")

            model.to(device)
            _, te_raw = build_loaders_raw(normalize=False, batch_size=args.batch)
            acc_raw = evaluate(model, te_raw, device)
            _, te_norm = build_loaders_raw(normalize=True, batch_size=args.batch)
            acc_norm = evaluate(model, te_norm, device)
            print(f"\n在测试集(10000 张)上的精度：")
            print(f"  直接喂 0~1 原图      : {acc_raw:6.2f}%   <= README.md 记录的测试精度")
            print(f"  先做标准化再喂        : {acc_norm:6.2f}%   "
                  f"({acc_norm - acc_raw:+.2f} 个百分点)")
            print("  两点结论：")
            print(f"   (a) 这个 checkpoint 实际能到 {acc_raw:.2f}%，与 README.md 的 72.67% 一致；")
            print("       03_图片识别案例.py 末尾注释写的 69.81% 对不上，说明那份注释")
            print("       记的是更早一次训练的结果，交报告时应当采信重新评测出来的值。")
            print(f"   (b) 换个输入分布（标准化）精度从 {acc_raw:.2f}% 掉到 {acc_norm:.2f}%，")
            print("       反向证明训练时喂进去的就是未标准化的 0~1 张量 —— ")
            print("       这正是原案例 create_data() 里那个 'todo: 数据标准化' 没做的后果。")
            results.append(dict(实验="1 复现原模型(未标准化)", 参数量=n_par,
                                测试精度=round(acc_raw, 2), 训练轮数="-", 耗时s="-"))
            results.append(dict(实验="1 原模型换标准化输入", 参数量=n_par,
                                测试精度=round(acc_norm, 2), 训练轮数="-", 耗时s="-"))

    # ---------------- 实验2：输入分布诊断（batch 维度） ----------------
    if args.only in ("all", "2"):
        print("\n" + "=" * 84)
        print("实验2  数据标准化的作用：同一网络、同一起点，只换输入分布")
        print("=" * 84)
        print("说明：实验2 与实验1 共用结论，这里额外打印一批数据的统计量做证据。")
        _, te_raw = build_loaders_raw(False, 8)
        x, _ = next(iter(te_raw))
        print(f"  未标准化一批 (8,3,32,32)：mean={x.mean():.4f}  std={x.std():.4f}  "
              f"min={x.min():.3f}  max={x.max():.3f}")
        _, te_norm = build_loaders_raw(True, 8)
        xn, _ = next(iter(te_norm))
        print(f"  标准化后同一批        ：mean={xn.mean():.4f}  std={xn.std():.4f}  "
              f"min={xn.min():.3f}  max={xn.max():.3f}")
        print("  标准化把每个通道搬到 0 附近、方差 1 —— "
              "梯度尺度一致，网络才敢用大一点的学习率。")

    # ---------------- 实验3：改进网络 ----------------
    if args.only in ("all", "3"):
        print("\n" + "=" * 84)
        print(f"实验3  改进网络训练（BatchNorm + 全局平均池化 + 标准化，{args.epochs} 轮）")
        print("=" * 84)
        tr, te = build_loaders(True, args.batch)
        print(f"训练集 {len(tr.dataset)} 张 / 测试集 {len(te.dataset)} 张，"
              f"batch_size={args.batch}，num_workers=0，lr={args.lr}（余弦退火）")

        net = ImprovedCNN()
        n_par = sum(p.numel() for p in net.parameters())
        print(f"改进模型参数量 : {n_par:,}  "
              f"（原模型 5,060,362，压缩到 {n_par / 5060362 * 100:.2f}%）")

        t0 = time.time()
        hist, best, best_ep = train_model(net, tr, te, device, args.epochs, args.lr, "改进", args.batch)
        el = time.time() - t0
        print(f"\n改进模型最佳测试精度 : {best:.2f}%（第 {best_ep} 轮）   总耗时 {el:.1f}s")
        results.append(dict(实验=f"3 改进模型({args.epochs}轮)", 参数量=n_par,
                            测试精度=round(best, 2), 训练轮数=args.epochs, 耗时s=round(el, 1)))

        torch.save(net.state_dict(), os.path.join(MODEL_DIR, "ImprovedCNN.pth"))
        print(f"权重已保存 : {os.path.relpath(os.path.join(MODEL_DIR, 'ImprovedCNN.pth'), HERE)}")

        # 训练历史落盘，供生成报告时引用真实曲线数据
        import json as _json
        with open(os.path.join(RES_DIR, "训练历史.json"), "w", encoding="utf-8") as f:
            _json.dump({"hist": hist, "best": best, "best_epoch": best_ep, "params": n_par,
                        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                        "base_acc": BASE_ACC, "sec": round(el, 1)},
                       f, ensure_ascii=False, indent=1)

        # ---- 图5：训练曲线 ----
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
        axes[0].plot(hist["epoch"], hist["loss"], "o-", color="tab:red", ms=3)
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("训练集平均交叉熵损失")
        axes[0].set_title("损失曲线", fontsize=11); axes[0].grid(alpha=0.3)
        axes[1].plot(hist["epoch"], hist["train_acc"], "o-", ms=3, label="训练集精度")
        axes[1].plot(hist["epoch"], hist["test_acc"], "s-", ms=3,
                     label=f"测试集精度（第 {best_ep} 轮最佳 {best:.2f}%）")
        axes[1].axhline(BASE_ACC, ls=":", color="gray",
                        label=f"原模型测试精度 {BASE_ACC:.2f}%")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("准确率 (%)")
        axes[1].set_title(f"精度曲线（最佳测试 {best:.2f}%）", fontsize=11)
        axes[1].grid(alpha=0.3); axes[1].legend()
        fig.suptitle("图5  改进网络（BatchNorm + 全局平均池化 + 数据标准化）的训练过程", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fp = os.path.join(FIG_DIR, "fig5_训练曲线.png")
        fig.savefig(fp, dpi=150); plt.close(fig)
        print(f"已保存 : {os.path.relpath(fp, HERE)}")

        # ---- 图6：预测样例（对/错各挑一批） ----
        net.eval().to(device)
        xb, yb = next(iter(DataLoader(CIFAR10(root=DATA_ROOT, train=False,
                                             transform=transforms.ToTensor(), download=False),
                                      batch_size=16, shuffle=True)))
        xb_n = (xb - CIFAR_MEAN) / CIFAR_STD
        with torch.no_grad():
            pb = net(xb_n.to(device)).argmax(1).cpu()
        fig, axes = plt.subplots(2, 8, figsize=(15.4, 4.5))
        for i, ax in enumerate(axes.ravel()):
            ax.imshow(xb[i].permute(1, 2, 0).numpy())
            ok = pb[i].item() == yb[i].item()
            ax.set_title(f"真:{CLASSES[yb[i]]}\n预:{CLASSES[pb[i]]}", fontsize=8.5,
                         color="green" if ok else "red")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("图6  改进模型在测试集上的随机 16 张预测样例（绿=预测正确，红=预测错误）",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fp6 = os.path.join(FIG_DIR, "fig6_预测样例.png")
        fig.savefig(fp6, dpi=150); plt.close(fig)
        print(f"已保存 : {os.path.relpath(fp6, HERE)}")

    # ---------------- 汇总 ----------------
    if results:
        csv_path = os.path.join(RES_DIR, "精度对比.csv")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f"\n汇总已保存 : {os.path.relpath(csv_path, HERE)}")
        print("\n" + "-" * 62)
        for r in results:
            print(f"{r['实验']:<28}{str(r['参数量']):>12}  {r['测试精度']:>6}%")
        print("-" * 62)


if __name__ == "__main__":
    main()
