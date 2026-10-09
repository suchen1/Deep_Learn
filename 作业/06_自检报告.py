# -*- coding: utf-8 -*-
"""生成后的自检：把 docx 当包来验证，别等 Word 报错才发现问题。"""
import glob
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))


def newest_report():
    """找最新的报告文件：优先 第一次作业_报告.docx，被占用时生成脚本会写 _v2/_v3…"""
    cands = glob.glob(os.path.join(HERE, "第一次作业_报告*.docx"))
    cands = [c for c in cands if not c.endswith(".tmp")]
    if not cands:
        raise SystemExit("找不到报告文件，请先运行 04_生成报告.py")
    return max(cands, key=os.path.getmtime)


P = newest_report()
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "ct": "http://schemas.openxmlformats.org/package/2006/content-types",
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}
problems = []

z = zipfile.ZipFile(P)
names = z.namelist()

# 1. 所有 xml 部件能否解析
for n in names:
    if n.endswith(".xml") or n.endswith(".rels"):
        try:
            ET.fromstring(z.read(n))
        except Exception as e:
            problems.append(f"[XML 解析失败] {n}: {e}")

# 2. Content_Types 是否覆盖所有部件扩展名
ct = ET.fromstring(z.read("[Content_Types].xml"))
defaults = {e.get("Extension").lower() for e in ct.findall("ct:Default", NS)}
overrides = {e.get("PartName") for e in ct.findall("ct:Override", NS)}
for n in names:
    if n == "[Content_Types].xml":
        continue
    ext = n.rsplit(".", 1)[-1].lower()
    if ext in ("xml", "rels"):
        if ext == "xml" and ("/" + n) not in overrides and ext not in defaults:
            problems.append(f"[Content_Types] 缺少 Override: {n}")
        continue
    if ext not in defaults:
        problems.append(f"[Content_Types] 缺少 Default Extension: {ext} ({n})")

# 3. document.xml 里引用的每个 r:id 是否都在 rels 里有定义，反之亦然
doc = z.read("word/document.xml").decode("utf-8")
paras = re.findall(r"<w:p[ >].*?</w:p>", doc, re.S)
rels = ET.fromstring(z.read("word/_rels/document.xml.rels"))
rel_ids = {r.get("Id"): (r.get("Target"), r.get("Type")) for r in rels.findall("rel:Relationship", NS)}
used = set(re.findall(r'r:(?:id|embed)="(rId\d+)"', doc))
missing = used - set(rel_ids)
if missing:
    problems.append(f"[关系缺失] document.xml 用到但 rels 里没有: {sorted(missing)}")

# 4. 每个图片 Target 是否真实存在（相对 word/ 解析）
for rid, (target, typ) in rel_ids.items():
    if "image" in typ or typ.endswith("/footer"):
        real = "word/" + target if not target.startswith("/") else target[1:]
        if real not in names:
            problems.append(f"[部件缺失] {rid} -> {real} 不在包里")

# 5. 未替换的占位符 / XML 转义残留
for token in ("{{", "}}", "&lt;w:"):
    if token in doc:
        problems.append(f"[内容可疑] 正文里出现了 {token!r}")

# 5b. 数据结构 repr 泄漏：把整段文本抽出来查，而不是查 XML。
#     查 XML 会漏掉 —— repr 里的单引号在 XML 里是 &apos;，而且 XML 本身满是 {' 之类。
#     典型症状：摘要那一段变成 "[('摘要：', {'ea': '黑体', ...}), (...)]"。
LEAK_PATTERNS = (r"'ea':", r"'bold':", r"'sz':", r"\bNone\b", r"^\s*[\[\(]\s*['\"]",
                 r"<class '", r"object at 0x")
for i, pp in enumerate(paras):
    t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", pp, re.S))
    if not t.strip():
        continue
    for pat in LEAK_PATTERNS:
        if re.search(pat, t):
            problems.append(f"[repr 泄漏] 第 {i} 段命中 {pat!r}：{t[:90]}")
            break

# 6. 统计
text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", doc, re.S))
tables = doc.count("<w:tbl>")
drawings = doc.count("<w:drawing>")
styles_used = set(re.findall(r'<w:pStyle w:val="([^"]+)"', doc))
style_defs = set(re.findall(r'w:styleId="([^"]+)"', z.read("word/styles.xml").decode("utf-8")))
undef = styles_used - style_defs

print("=" * 70)
print(f"文件        : {os.path.basename(P)}  {os.path.getsize(P) / 1024:.1f} KB")
print(f"部件数      : {len(names)}")
print(f"段落数      : {len(paras)}")
print(f"文字量      : {len(text):,} 字符（含表格、图题、代码块内文本）")
print(f"表格        : {tables} 个    插图: {drawings} 张")
print(f"用到的样式  : {sorted(styles_used)}")
print(f"样式是否都已定义 : {'是' if not undef else '否 -> ' + str(undef)}")
print(f"图片关系    : {sorted(t for t, ty in rel_ids.values() if 'image' in ty)}")
if undef:
    problems.append(f"[样式未定义] {undef}")
print("-" * 70)
if problems:
    print("发现的问题：")
    for x in problems:
        print("  ! " + x)
    sys.exit(1)
print("自检通过：XML 可解析、内容类型完整、关系齐备、无未替换占位符、样式全部已定义。")

# 顺带把纯文本导出一份，便于人工通读
out = os.path.join(os.path.dirname(P), "过程记录", "_报告正文校对.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    cur = []
    for pp in paras:
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", pp, re.S))
        sty = re.findall(r'<w:pStyle w:val="([^"]+)"', pp)
        if sty:
            f.write(f"\n[{sty[0]}] ")
        f.write(t + "\n")
print(f"正文纯文本已导出 : {out}")
