# -*- coding: utf-8 -*-
"""检查最终报告：标题编号是否连续、正文是否通顺。"""
import glob
import os
import re
import zipfile

HERE = r"D:\ML_Project\Deep Learn\作业"
cands = [c for c in glob.glob(os.path.join(HERE, "第一次作业_报告*.docx")) if not c.endswith(".tmp")]
P = max(cands, key=os.path.getmtime)
print("检查文件 :", os.path.basename(P), "\n")

z = zipfile.ZipFile(P)
doc = z.read("word/document.xml").decode("utf-8")

print("=" * 74)
print("标题结构（编号由模板的自动编号生成，这里按 ilvl 推演）")
print("=" * 74)
cnt = [0, 0, 0]
for pp in re.findall(r"<w:p[ >].*?</w:p>", doc, re.S):
    sty = re.findall(r'<w:pStyle w:val="(a[012])"', pp)
    if not sty:
        continue
    t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", pp, re.S))
    lvl = int(sty[0][1])
    cnt[lvl] += 1
    for k in range(lvl + 1, 3):
        cnt[k] = 0
    num = ".".join(str(c) for c in cnt[: lvl + 1])
    indent = "  " * lvl
    print(f"{indent}{num}  {t}")
