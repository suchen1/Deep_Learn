# -*- coding: utf-8 -*-
"""检查最终 docx 里有没有把 Python 数据结构（列表/元组/dict 的 repr）当成正文写进去。"""
import re
import zipfile

P = r"D:\ML_Project\Deep Learn\作业\第一次作业_报告.docx"
z = zipfile.ZipFile(P)
doc = z.read("word/document.xml").decode("utf-8")

paras = re.findall(r"<w:p[ >].*?</w:p>", doc, re.S)
bad = []
for i, pp in enumerate(paras):
    t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", pp, re.S))
    if re.search(r"[\[\(]\s*['\"]|'ea':|'bold':|'sz':|\{'", t):
        bad.append((i, t[:120]))

print(f"段落总数 : {len(paras)}")
print(f"可疑段落 : {len(bad)}")
for i, t in bad:
    print(f"  --- 第 {i} 段 ---")
    print(f"  {t}")

# 单独看摘要与关键词两段
for pp in paras:
    t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", pp, re.S))
    if "本报告完成" in t or "卷积神经网络；图像滤波" in t:
        print("\n实际写入的内容（前 120 字）:")
        print("  " + t[:120])
