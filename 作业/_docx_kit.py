# -*- coding: utf-8 -*-
"""
生成《第一次作业》报告 Word 文档。

为什么要自己拼 OOXML，而不是用 python-docx
------------------------------------------
本机 pip 装不上 python-docx（沙箱下临时目录不可写，多次尝试均 Permission denied），
而报告必须严格套用《报告模板.docx》的样式（一级/二级/三级标题自动编号、字体字号、
页面边距、参考文献格式都在 styles.xml 里）。所以这里直接以模板的样式包为底，
用标准库 zipfile + 自己拼 document.xml 生成 docx：
  * word/styles.xml、numbering.xml、theme、fontTable 全部沿用模板原样
  * 只重写 word/document.xml 与图片关系，并追加所需图片
这样文档打开后就是"在模板里写的一样"，且不依赖任何第三方库。

格式（照抄模板 paragraphs 的直接格式）
------------------------------------
    标题      居中 18pt(36 半点)
    副标题    18pt 缩进
    作者/邮箱 居中 12pt(24 半点)
    摘要/关键词 楷体_GB2312 10pt，1.3 倍行距
    一级标题  样式 a0（宋体 12pt，自动编号）
    二级标题  样式 a1（黑体 9pt，自动编号）
    三级标题  样式 a2（楷体_GB2312 9pt，自动编号）
    正文      宋体 10.5pt，首行缩进 2 字符，1.15 倍行距
    表格正文  9pt，居中
    图题/表题 8pt / 9pt，居中
    参考文献  样式 af（宋体 7.5pt）
    页面      A4，上下 1440、左右 1800 twips（模板 sectPr 原值）

运行
----
    cd 作业 && python 04_生成报告.py
输出
----
    第一次作业_报告.docx
"""
import os
import io
import json
import shutil
import zipfile

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# 这两个路径由调用方（04_生成报告.py）在运行前覆盖，这里只留占位
TEMPLATE = os.path.join(HERE, "报告模板.docx")
OUT = os.path.join(HERE, "第一次作业_报告.docx")
FIG = os.path.join(HERE, "结果图")
RES = os.path.join(HERE, "结果")

CN = 'w:ascii="Times New Roman" w:eastAsia="%s" w:hAnsi="Times New Roman" w:cs="Times New Roman"'
EMU_PER_PX = 9525          # 96 dpi
MAX_W_TWIP = 8800          # 正文可用宽度约 15.5 cm（A4 减去左右各 1800 twips 边距）
MAX_H_TWIP = 11800         # 单张图不超过约 20.8 cm，保证整图能落在一页里


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# ============================================================
# 段落 / 文本
# ============================================================
def fld(instr):
    """插入域代码：用于生成可自动更新的"第 X 页 共 Y 页"。"""
    return (f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r><w:instrText xml:space="preserve"> {instr} </w:instrText></w:r>'
            f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r><w:t>1</w:t></w:r>'
            f'<w:r><w:fldChar w:fldCharType="end"/></w:r>')


def p(text="", *, style=None, jc=None, sz=None, ea="宋体", bold=False,
      first_line=None, left=None, line=None, before=None, after=None,
      ilvl=None, num_id=None, runs=None, page_break=False, keep_next=False,
      no_num=False):
    """
    生成一个 <w:p>。
    text     纯文本；也可以直接传 [(文本, {覆盖属性}), ...] 做段内混排（如"摘要："黑体 + 正文楷体）
    runs     同上，显式写法；与 text 二选一
    ilvl/num_id 用了就自动编号（与模板 a0/a1/a2 一致：numId=2, ilvl=0/1/2）
    """
    # ---- 归一化：把"混排内容"统一收到 runs 里 ----
    # 这里踩过坑：调用方按 p([(文本, 属性), ...]) 传进来，第一个位置参数是 text 不是 runs，
    # 于是 text 拿到的是整个列表、runs 仍是 None；后面 runs = [(text, {})] 就把这个列表
    # 当成了"文本"，再 str() 一下，整份数据结构被原样写进正文
    # （摘要那段变成 "[('摘要：', {'ea': '黑体', ...}), (...)]"）。
    if isinstance(text, (list, tuple)):
        if runs is not None:
            raise ValueError("text 与 runs 不能同时传混排内容")
        runs = text
        text = ""
    if runs is None:
        runs = [(text, {})]
    elif isinstance(runs, str):
        runs = [(runs, {})]
    elif not isinstance(runs, (list, tuple)):
        raise TypeError(f"runs 必须是 str / list / tuple，收到 {type(runs).__name__}")

    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if no_num:
        # 样式 a0/a1/a2 自带 numId=2 的自动编号。这里用 numId=0 把它顶掉，
        # 因为序号已经作为文字写进标题了（"1 引言"），不顶掉会变成"1 1 引言"。
        ppr.append('<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr>')
    elif num_id is not None:
        ppr.append(f'<w:numPr><w:ilvl w:val="{ilvl or 0}"/><w:numId w:val="{num_id}"/></w:numPr>')
    if keep_next:
        ppr.append('<w:keepNext/>')
    if page_break:
        ppr.append('<w:pageBreakBefore/>')
    if before is not None or after is not None or line is not None:
        a = f' w:before="{before}"' if before is not None else " "
        b = f' w:after="{after}"' if after is not None else " "
        l = f' w:line="{line}" w:lineRule="auto"' if line is not None else ""
        ppr.append(f'<w:spacing{a}{b}{l}/>')
    ind = []
    if first_line:
        ind.append(f'w:firstLine="{first_line}" w:firstLineChars="{int(first_line / 2.1)}"')
    if left:
        ind.append(f'w:left="{left}"')
    if ind:
        ppr.append('<w:ind ' + " ".join(ind) + "/>")
    if jc:
        ppr.append(f'<w:jc w:val="{jc}"/>')

    def rpr(d):
        size = d.get("sz", sz)
        size = 21 if size is None else size
        s = ['<w:rFonts %s w:hint="eastAsia"/>' % (CN % d.get("ea", ea))]
        if d.get("bold", bold):
            s.append("<w:b/><w:bCs/>")
        if d.get("mono"):
            s.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="宋体"/>')
        s.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
        if d.get("color"):
            s.append(f'<w:color w:val="{d["color"]}"/>')
        if d.get("shade"):
            s.append(f'<w:shd w:val="clear" w:fill="{d["shade"]}"/>')
        return "<w:rPr>" + "".join(s) + "</w:rPr>"

    def run(t, d=None):
        d = d or {}
        # 闸门：只允许字符串进文档。这里以前是 str(t)，
        # 会把 list/dict 的 repr 原样写进正文，且不报错、很难发现。
        if not isinstance(t, str):
            raise TypeError(
                f"段落文本必须是 str，收到 {type(t).__name__}：{t!r:.80}\n"
                f"（多半是内容定义里多了个逗号，把字符串变成了元组）")
        parts = t.split("\n")
        body = []
        for i, seg in enumerate(parts):
            if i:
                body.append("<w:br/>")
            body.append(f'<w:t xml:space="preserve">{esc(seg)}</w:t>')
        return f"<w:r>{rpr(d)}{''.join(body)}</w:r>"

    # runs 已在函数开头归一化为 [(文本, 属性), ...]；元组原样保留，不能再包一层
    rs = []
    for item in runs:
        if isinstance(item, tuple):
            if len(item) != 2:
                raise ValueError(f"run 必须是 (文本, 属性) 二元组，收到长度 {len(item)}：{item!r:.60}")
            t, d = item
            rs.append(fld(t[4:]) if (isinstance(t, str) and t.startswith("FLD:")) else run(t, d))
        else:
            rs.append(run(item))

    ppr_xml = ("<w:pPr>" + "".join(ppr) + "</w:pPr>") if ppr else ""
    return f"<w:p>{ppr_xml}{''.join(rs)}</w:p>"


# 便捷封装
# 注意：序号由 04_生成报告.py 作为**文字**写进标题（"1 引言"），
# 所以这里统一用 no_num=True 顶掉模板样式自带的自动编号，避免"1 1 引言"。
def h1(t, **kw):
    kw.setdefault("no_num", True)
    return p(t, style="a0", before=343, after=343, **kw)


def h2(t, **kw):
    kw.setdefault("no_num", True)
    return p(t, style="a1", before=249, after=249, **kw)


def h3(t, **kw):
    kw.setdefault("no_num", True)
    return p(t, style="a2", before=156, after=156, **kw)


def body(t, **kw):
    """正文：首行缩进 2 字符（360 twips），1.15 倍行距。"""
    kw.setdefault("first_line", 360)
    kw.setdefault("line", 276)
    kw.setdefault("jc", "both")
    return p(t, **kw)


def bullet(t, level=0):
    """无序条目：用 "·" 手工打点，避免再引入一套模板外的编号定义。"""
    marks = ["●", "○", "◆"]
    return p(marks[min(level, 2)] + " " + t, left=360 + level * 300, first_line=0,
             line=276, jc="both")


def caption(t, kind="图"):
    """图题 / 表题。"""
    return p(t, jc="center", sz=17, before=60, after=120, line=240)


def table_caption(t):
    return p(t, jc="center", sz=19, ea="黑体", before=120, after=60, line=240, keep_next=True)


def code(lines):
    """等宽代码块：整段浅灰底，模拟 IDE 里的样子。"""
    out = []
    for i, ln in enumerate(lines):
        d = {"mono": True, "sz": 17, "shade": "F2F2F2"}
        out.append(p(ln if ln else " ", left=340, first_line=0, line=240,
                     before=(60 if i == 0 else 0), after=(60 if i == len(lines) - 1 else 0),
                     runs=[(ln if ln else " ", d)]))
    return "".join(out)


# ============================================================
# 表格
# ============================================================
def _cell(text, w, *, header=False, jc="center", sz=18, ea=None):
    rpr = ('<w:rPr><w:rFonts %s/><w:sz w:val="%d"/><w:szCs w:val="%d"/>%s</w:rPr>'
           % (CN % (ea or ("黑体" if header else "宋体")), sz, sz,
              "<w:b/><w:bCs/>" if header else ""))
    body_ = "".join(
        f'<w:r>{rpr}<w:t xml:space="preserve">{esc(seg)}</w:t></w:r>'
        for seg in str(text).split("\n")
    )
    shd = '<w:shd w:val="clear" w:fill="EDEFF3"/>' if header else ""
    return (f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}'
            f'<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p><w:pPr><w:jc w:val="{jc}"/><w:spacing w:line="240" w:lineRule="auto"/>'
            f'<w:ind w:firstLine="0"/></w:pPr>{body_}</w:p></w:tc>')


def table(rows, widths, sz=18):
    """
    rows[0] 作为表头。widths 为各列宽度（twips），总和 = MAX_W_TWIP。
    """
    borders = ('<w:tblBorders>'
               + "".join(f'<w:{s} w:val="single" w:sz="6" w:space="0" w:color="7F7F7F"/>'
                         for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
               + "</w:tblBorders>")
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"
    trs = []
    for i, r in enumerate(rows):
        cells = [_cell(c, widths[j], header=(i == 0), sz=sz) for j, c in enumerate(r)]
        trs.append(f'<w:tr>{"".join(cells)}</w:tr>')
    pr = (f'<w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/><w:jc w:val="center"/>'
          f'{borders}<w:tblLayout w:type="fixed"/></w:tblPr>')
    return f"<w:tbl>{pr}{grid}{''.join(trs)}</w:tbl>"


# ============================================================
# 图片
# ============================================================
class Media:
    """收集要写进 docx 的图片，分配 rId 与文件名。"""
    def __init__(self):
        self.items = []          # (rId, zip 内路径, 本地路径)

    def add(self, path, rid):
        ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
        name = f"media/image{len(self.items) + 1}.{ext}"
        self.items.append((rid, name, path))
        return name


def figure(media, path, rid, caption_text, max_w=MAX_W_TWIP, max_h=MAX_H_TWIP):
    """插入居中图片 + 图题。按原始宽高比缩放到版心内。"""
    with Image.open(path) as im:
        pw, ph = im.size
    w = max_w
    h = int(w * ph / pw)
    if h > max_h:
        h = max_h
        w = int(h * pw / ph)
    name = media.add(path, rid)
    cx, cy = w * 635, h * 635          # twips -> EMU (1 twip = 635 EMU)
    xml = (
        '<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="40"/>'
        '<w:ind w:firstLine="0"/></w:pPr><w:r><w:drawing>'
        '<wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{rid}" name="{name}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr>'
        f'<pic:cNvPr id="{rid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="rId{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
        f'<a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )
    return xml + caption(caption_text)


# ============================================================
# 读实验产出的数值，写进报告（不让报告里的数字和真实结果脱钩）
# ============================================================
def load_results():
    import csv
    out = {}
    flt = os.path.join(RES, "滤波指标.csv")
    if os.path.exists(flt):
        with open(flt, encoding="utf-8-sig") as f:
            out["filter"] = list(csv.DictReader(f))
    acc = os.path.join(RES, "精度对比.csv")
    if os.path.exists(acc):
        with open(acc, encoding="utf-8-sig") as f:
            out["acc"] = {r["实验"]: r for r in csv.DictReader(f)}
    lay = os.path.join(RES, "CNN逐层结构.csv")
    if os.path.exists(lay):
        with open(lay, encoding="utf-8-sig") as f:
            out["layers"] = list(csv.DictReader(f))
    hist = os.path.join(RES, "训练历史.json")
    if os.path.exists(hist):
        with open(hist, encoding="utf-8") as f:
            out["hist"] = json.load(f)
    return out
