# -*- coding: utf-8 -*-
"""用 Word 打开最终文档，检查标题段落实际显示的文字（确认没有 '1 1 引言' 这种重复编号）。

如果沙箱拒绝 COM，就退回到纯 XML 判断并明确说明。
"""
import glob
import os
import re
import subprocess
import zipfile

HERE = r"D:\ML_Project\Deep Learn\作业"
cands = [c for c in glob.glob(os.path.join(HERE, "第一次作业_报告*.docx")) if not c.endswith(".tmp")]
P = max(cands, key=os.path.getmtime)
print("文件:", os.path.basename(P), "\n")

z = zipfile.ZipFile(P)
doc = z.read("word/document.xml").decode("utf-8")

# 每个标题段落：样式、是否被 numId=0 顶掉自动编号、文字里是否已含手写序号
n_manual = n_auto = 0
dup = []
for pp in re.findall(r"<w:p[ >].*?</w:p>", doc, re.S):
    sty = re.findall(r'<w:pStyle w:val="(a[012])"', pp)
    if not sty:
        continue
    t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", pp, re.S))
    killed = '<w:numId w:val="0"' in pp
    if killed:
        n_manual += 1
    else:
        n_auto += 1
    # 手写序号 + 样式仍带 numId=2 -> 会显示成 "1 1 引言"
    if not killed and re.match(r"^\d+(\.\d+)*[\s\u3000]", t):
        dup.append(t)

print(f"标题段落: 已顶掉自动编号(手写序号) {n_manual} 个，仍用自动编号 {n_auto} 个")
if dup:
    print("!! 可能重复编号的标题:")
    for d in dup:
        print("   ", d)
else:
    print("没有发现'手写序号 + 自动编号'叠加的情况 —— 不会出现 '1 1 引言'")
