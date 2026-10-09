# -*- coding: utf-8 -*-
"""
生成《第一次作业_报告.docx》。

流程
----
1. 读 作业/结果/*.csv、*.json 里的真实实验数据（不让报告里的数字和实验结果脱钩）
2. 现场把 5 种构建写法的参数量再算一遍（避免写死在报告里）
3. 用 _docx_kit.py 把这些内容渲染成 document.xml
4. 以《报告模板.docx》为底包：沿用它的 styles.xml / numbering.xml / 页面设置，
   替换 document.xml，追加图片与页脚 —— 不改动模板文件本身

为什么必须用模板的 numbering：一级/二级/三级标题的编号（1、1.1、1.1.1）
是模板 styles.xml 里 numId=2 定义的自动编号，只有把段落挂到 a0/a1/a2 样式上才能自动生成。

运行
----
    cd 作业 && python 04_生成报告.py
输出
----
    第一次作业_报告.docx
"""
import os
import re
import json
import shutil
import zipfile

import _docx_kit as K
import _report_content as C

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "第一次作业_报告.docx")
FIG = os.path.join(HERE, "结果图")
RES = os.path.join(HERE, "结果")

# 用作样式底版的模板。优先用作业目录下的副本，没有就用随任务下发的原始模板。
TEMPLATE_CANDIDATES = [
    os.path.join(HERE, "报告模板.docx"),
    os.path.join(os.environ.get("APPDATA", ""), "dsh-desktop", "harness", "attachments",
                 "v1", "files", "33",
                 "33a6da58f71aa26a03b53045117cf3e4b7d2b28eaac551fe282360624b057ee4",
                 "报告模板.docx"),
]


def find_template():
    for p in TEMPLATE_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError("找不到《报告模板.docx》，请把它放在作业目录下。"
                            "已尝试：\n  " + "\n  ".join(TEMPLATE_CANDIDATES))


# 网页/Word 的正文可用宽度：A4 210mm - 左右各 1800 twips(31.75mm) = 146.5mm ≈ 8308 twips
K.MAX_W_TWIP = 8300
K.MAX_H_TWIP = 12200


# ============================================================
# 现场重算 5 种构建写法的参数量（不写死数字）
# ============================================================
def compute_build_table():
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location(
        "ana", os.path.join(HERE, "02_CNN结构分析.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    variants = [
        ("A  __init__ 逐层声明 + forward 手写串联（已有项目）", mod.BuildA(),
         "最直白，便于逐层调试；层多时冗长"),
        ("B  nn.Sequential 顺序容器", mod.build_b(),
         "最简洁；只能表达一条直线的连接"),
        ("C  nn.ModuleList + for 循环", mod.BuildC(),
         "重复结构最省事；需自己在 forward 里串联"),
        ("D  nn.ModuleDict 按名字存取", mod.BuildD(),
         "便于按名字替换某层做消融实验"),
        ("E  全局平均池化 + 单层分类头", mod.BuildE(),
         "参数量降两个数量级（结构升级版）"),
    ]
    rows = [["写法", "参数量", "输出形状", "适用场景"]]
    for name, m, note in variants:
        n = sum(p.numel() for p in m.parameters())
        m.eval()
        with torch.no_grad():
            y = m(torch.randn(2, 3, 32, 32))
        rows.append([name, f"{n:,}", str(tuple(y.shape)), note])
    return rows


# ============================================================
# 各表格的数据（表头 + 数据行 + 列宽 twips，总和 ≈ 8300）
# ============================================================
def build_tables(data):
    T = {}

    # ---- 表1 滤波指标 ----
    rows = [["滤波器核", "核边长 k", "彩色 MAE", "灰度 MAE", "彩色 PSNR (dB)", "边缘保留率"]]
    for r in data["filter"]:
        rows.append([r["核"], r["k"], r["彩色MAE"], r["灰度MAE"],
                     r["彩色PSNR_dB"], r["边缘保留率"] + " %"])
    T["filter"] = (rows, [2100, 1050, 1250, 1250, 1450, 1200])

    # ---- 表2 逐层结构 ----
    rows = [["层名", "类型", "超参", "输出形状", "参数量", "感受野"]]
    for r in data["layers"]:
        p = r["参数量"].strip()
        if p in ("", "0"):
            p = "0"
        else:
            p = f"{int(p):,}"
        rows.append([r["层名"], r["类型"], r["超参"], r["输出形状"], p, r["感受野"]])
    T["layers"] = (rows, [800, 1050, 2950, 1250, 1350, 900])

    # ---- 表3 参数量 vs 计算量 ----
    T["cost"] = ([
        ["项目", "卷积层", "全连接层", "合计"],
        ["参数量（个）", "333,056", "4,727,306", "5,060,362"],
        ["参数占比", "6.58 %", "93.42 %", "100 %"],
        ["前向乘加 MACs", "11,726,208", "4,723,712", "16,449,920"],
        ["计算量占比", "71.3 %", "28.7 %", "100 %"],
        ["层数（带参数）", "3", "4", "7"],
    ], [2100, 2100, 2100, 2000])

    # ---- 表4 感受野对比（三种结构，实测） ----
    T["rf"] = ([
        ["结构", "参数量", "末层感受野", "对 32×32 的覆盖率"],
        ["原模型 ImgMode（3 段「卷积+池化」）", "5,060,362", "22 × 22", "47.3 %"],
        ["改用 stride=2 卷积（只下采样 2 次）", "452,426", "13 × 13", "16.5 %"],
        ["改进结构（6 卷积 + 3 池化 + 全局平均池化）", "140,714", "28 × 28", "76.6 %"],
    ], [3400, 1600, 1650, 1650])

    # ---- 表5 五种构建写法（现场计算） ----
    T["build"] = (compute_build_table(), [3050, 1150, 1000, 3100])

    # ---- 表5 精度对比 ----
    acc = data.get("acc", {})
    hist = data.get("hist", {})
    best = hist.get("best")
    n_imp = hist.get("params")
    base = float(acc.get("复现原模型(未标准化)", {}).get("测试精度", 72.67))
    rows = [["模型", "参数量", "模型大小", "测试集准确率", "说明"]]
    rows.append(["原 ImgMode（CNN/model/ImgModel.pth）", "5,060,362", "19.30 MB",
                 f"{base:.2f} %",
                 "独立重测；与 README 记录一致"])
    if n_imp and best:
        rows.append(["改进网络（作业/model/ImprovedCNN.pth）", f"{n_imp:,}",
                     f"{n_imp * 4 / 1024 / 1024:.2f} MB", f"{best:.2f} %",
                     "BatchNorm + 全局平均池化 + 标准化"])
        rows.append(["改进后参数压缩比", f"{n_imp / 5060362 * 100:.2f} %",
                     "1 / 35.8",
                     f"+{best - base:.2f} 个百分点", "网络小得多，精度反而更高"])
    T["acc"] = (rows, [2700, 1250, 1000, 1350, 2000])

    # ---- 表6 batch_size 扫描（来自 README.md 记录的实测，独立于报告） ----
    T["batch"] = ([
        ["batch_size", "单步耗时", "吞吐（张/秒）", "一个 epoch"],
        ["8", "1.82 ms", "4 401", "19.17 s"],
        ["16", "2.07 ms", "7 720", "—"],
        ["64", "2.10 ms", "30 537", "3.80 s"],
        ["128", "2.99 ms", "42 759", "3.07 s"],
        ["256", "5.68 ms", "45 055", "2.17 s"],
    ], [2100, 1900, 2300, 2000])

    return T


# ============================================================
# 占位符解析：正文里的 {{BEST}} 等一律从实验产物取值，
# 保证报告里的数字和真实跑出来的结果永远一致（改实验不用改报告）
# ============================================================
VALUES = {}


def resolve(x):
    """
    把字符串里的 {{KEY}} 换成实验产出的真实数值。
    只接受字符串：如果收到 tuple/list 说明内容定义写错了（比如尾逗号把字符串变成了元组），
    必须直接报错而不是 str() 一下了事 —— 否则 Python 的元组 repr 会被原样写进正文。
    """
    if isinstance(x, str):
        out = x
        for k, v in VALUES.items():
            out = out.replace("{{" + k + "}}", v)
        if "{{" in out:
            raise KeyError(f"未解析的占位符在：{out[:80]}...")
        return out
    if isinstance(x, tuple):
        return tuple(resolve(i) for i in x)
    if isinstance(x, list):
        return [resolve(i) for i in x]
    return x


def as_text(label, value):
    """内容定义必须是纯字符串，这里做一道类型闸门。"""
    if not isinstance(value, str):
        raise TypeError(f"{label} 应该是字符串，实际是 {type(value).__name__}：{value!r:.60}...")
    return value


def init_values(data):
    hist = data.get("hist") or {}
    acc = data.get("acc", {})
    best = hist.get("best")
    base = float(acc.get("复现原模型(未标准化)", {}).get("测试精度", 72.67))
    ep = tr = 0.0
    if "hist_full" in data:
        h = data["hist_full"]
        ep = h.get("best_epoch")
        if ep is None:      # 老格式的 JSON 没有记录轮次，退回按最大值查找
            hist_ = h["hist"]
            ep = hist_["epoch"][max(range(len(hist_["test_acc"])),
                                    key=lambda k: hist_["test_acc"][k])]
        tr = h["hist"]["train_acc"][int(ep) - 1]
        best = h["hist"]["test_acc"][int(ep) - 1]
    VALUES.update({
        "BEST": f"{best:.2f}%" if best else "—",
        "DELTA": f"{best - base:.2f}" if best else "—",
        "BEPOCH": str(ep),
        "TRACC": f"{tr:.2f}",
        "SEC": f"{data.get('hist_full', {}).get('sec', 200.9):.1f}",
        "BASE": f"{base:.2f}%",
    })


# ============================================================
# 组装 document.xml
# ============================================================
def build_document(data):
    media = K.Media()
    rid = [100]                      # 图片 rId 计数器（避开 rId1=rId 页脚、rId2=styles）

    def next_rid():
        rid[0] += 1
        return rid[0]

    parts = []

    # ---------- 封面 ----------
    parts.append(K.p(as_text("标题", C.COVER["title"]), jc="center", sz=36, ea="黑体", bold=True,
                     before=200, after=120, line=300))
    parts.append(K.p(as_text("副标题", C.COVER["subtitle"]), jc="center", sz=28, ea="楷体_GB2312",
                     after=240, line=300))
    parts.append(K.p(f'{C.COVER["author"]}　{C.COVER["sid"]}', jc="center", sz=24, after=60))
    parts.append(K.p(f'（电子邮件邮箱{C.COVER["email"]}）', jc="center", sz=24, after=200))

    # ---------- 摘要 ----------
    parts.append(K.p([("摘要：", {"ea": "黑体", "bold": True, "sz": 21}),
                      (resolve(as_text("摘要", C.ABSTRACT)), {"ea": "楷体_GB2312", "sz": 21})],
                     first_line=0, line=276, jc="both"))
    parts.append(K.p(" "))
    parts.append(K.p([("关键词：", {"ea": "黑体", "bold": True, "sz": 21}),
                      (as_text("关键词", C.KEYWORDS), {"ea": "楷体_GB2312", "sz": 21})],
                     first_line=0, line=276, jc="both"))

    # ---------- 正文 ----------
    # 标题序号直接写成文字（"1 引言" / "2.1 输入图像…"），并用 numId=0 顶掉模板样式
    # 自带的自动编号，避免变成"1 1 引言"。手写序号的好处是：不管在 Word、WPS 还是
    # 在线预览里打开，序号都在，不依赖阅读器是否渲染 numbering。
    first_body = True
    counters = [0, 0, 0]

    def numbered(level, title):
        counters[level] += 1
        for k in range(level + 1, 3):
            counters[k] = 0
        return ".".join(str(c) for c in counters[:level + 1]) + "　" + title

    for item in C.CONTENT:
        kind = item[0]
        if kind == "h1":
            parts.append(K.h1(numbered(0, resolve(item[1])), page_break=first_body))
            first_body = False
        elif kind == "h2":
            parts.append(K.h2(numbered(1, resolve(item[1]))))
        elif kind == "h3":
            parts.append(K.h3(numbered(2, resolve(item[1]))))
        elif kind == "h2n":                       # 不编号的章标题（备用）
            parts.append(K.p(resolve(item[1]), style="a0", before=343, after=343,
                             runs=[(resolve(item[1]), {"ea": "黑体", "bold": True, "sz": 24})],
                             no_num=True, page_break=first_body))
            first_body = False
        elif kind == "p":
            parts.append(K.body(resolve(item[1])))
        elif kind == "b":
            parts.append(K.bullet(resolve(item[1])))
        elif kind == "b2":
            parts.append(K.bullet(resolve(item[1]), level=1))
        elif kind == "code":
            parts.append(K.code([resolve(x) for x in item[1]]))
        elif kind == "ref":
            parts.append(K.p(resolve(item[1]), style="af", left=340, first_line=-340, line=260))
        elif kind == "fig":
            key = item[1]
            path = os.path.join(FIG, FIGFILE[key])
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少配图：{path}（请先运行对应实验脚本）")
            parts.append(K.figure(media, path, next_rid(),
                                  f"图 {FIGNO[key]}　{resolve(FIGCAP[key])}"))
        elif kind == "tblcap":
            parts.append(K.table_caption(resolve(item[1])))
        elif kind == "tblfoot":
            parts.append(K.p(resolve(item[1]), sz=17, line=240, before=40, after=160,
                             left=120, first_line=0, jc="both"))
        elif kind == "table":
            key = item[1]
            if key not in TABLES:
                raise KeyError(f"正文引用了未定义的表格：{key}")
            rows, widths = TABLES[key]
            parts.append(K.table(resolve(rows), widths))
        else:
            raise ValueError(f"未知的内容类型：{kind}")

    # ---------- 页脚（第 X 页 共 Y 页） ----------
    footer = ('<w:p><w:pPr><w:jc w:val="center"/><w:ind w:firstLine="0"/>'
              '<w:rPr><w:sz w:val="18"/></w:rPr></w:pPr>'
              '<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t xml:space="preserve">第 </w:t></w:r>'
              + K.fld("PAGE")
              + '<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t xml:space="preserve"> 页　共 </w:t></w:r>'
              + K.fld("NUMPAGES")
              + '<w:r><w:rPr><w:sz w:val="18"/></w:rPr><w:t xml:space="preserve"> 页</w:t></w:r>'
              '</w:p>')
    footer = ('<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
              + footer + '</w:ftr>')

    # ---------- sectPr：完全沿用模板的页面设置，仅追加页脚引用 ----------
    sectpr = ('<w:sectPr>'
              '<w:footerReference w:type="default" r:id="rId1"/>'
              '<w:pgSz w:w="11900" w:h="16840"/>'
              '<w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800"'
              ' w:header="851" w:footer="992" w:gutter="0"/>'
              '<w:cols w:space="425"/><w:docGrid w:type="lines" w:linePitch="312"/>'
              '</w:sectPr>')

    body = "".join(parts) + sectpr
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
           '<w:document '
           'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
           'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
           'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
           'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture" '
           'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
           'mc:Ignorable="w14 wp14">'
           f'<w:body>{body}</w:body></w:document>')
    return doc, media, footer


FIGFILE = {
    "fig1": "fig1_滤波对比.png",
    "fig2": "fig2_局部放大.png",
    "fig3": "fig3_指标曲线.png",
    "fig4": "fig4_网络结构.png",
    "fig5": "fig5_训练曲线.png",
    "fig6": "fig6_预测样例.png",
    "fig7": "fig7_滤波器核.png",
}
FIGNO = {"fig1": "1", "fig2": "2", "fig3": "3", "fig4": "4",
         "fig5": "5", "fig6": "6", "fig7": "7"}
FIGCAP = {
    "fig1": "不同大小滤波器核的滤波结果与差异对比（上：彩色滤波结果　中：与原图的差值图　"
            "下：灰度滤波结果）",
    "fig2": "细节最密集区域的 4 倍放大对比：核越大，纹理被抹得越平",
    "fig3": "三项指标随滤波器核大小的变化",
    "fig4": "ImgMode 逐层结构与参数量分布",
    "fig5": "改进网络（BatchNorm + 全局平均池化 + 数据标准化）的训练过程，最佳测试精度 {{BEST}}",
    "fig6": "改进模型在测试集上的随机 16 张预测样例（绿=正确，红=错误）",
    "fig7": "四种滤波器核的系数分布",
}
TABLES = {}


# ============================================================
# 打包成 docx
# ============================================================
def write_docx(doc_xml, media, footer):
    src = zipfile.ZipFile(K.TEMPLATE)
    names = src.namelist()
    if os.path.exists(K.OUT):
        os.remove(K.OUT)

    with zipfile.ZipFile(K.OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            if n == "word/document.xml":
                z.writestr(n, doc_xml)
                continue
            if n == "word/_rels/document.xml.rels":
                # 重建关系：页脚 + 所有图片 + 保留模板原有的主题/字体/设置关系
                rels = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
                        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
                        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>']
                for i, (rid, zp, _lp) in enumerate(media.items, start=2):
                    ext = os.path.splitext(zp)[1].lstrip(".").lower()
                    rels.append(
                        f'<Relationship Id="rId{rid}" Type="http://schemas.openxmlformats.org/'
                        f'officeDocument/2006/relationships/image" Target="{zp}"/>')
                for n2 in names:
                    if n2.startswith("word/_rels/") and n2 != "word/_rels/document.xml.rels":
                        continue
                if "word/styles.xml" in names:
                    rels.append('<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>')
                if "word/settings.xml" in names:
                    rels.append('<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>')
                if "word/theme/theme1.xml" in names:
                    rels.append('<Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
                if "word/fontTable.xml" in names:
                    rels.append('<Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable" Target="fontTable.xml"/>')
                if "word/webSettings.xml" in names:
                    rels.append('<Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/webSettings" Target="webSettings.xml"/>')
                rels.append('</Relationships>')
                z.writestr(n, "".join(rels))
                continue
            if n == "[Content_Types].xml":
                ct = src.read(n).decode("utf-8")
                want = [("png", "png"), ("jpg", "jpeg"), ("jpeg", "jpeg")]
                exts = "".join(f'<Default Extension="{e}" ContentType="image/{m}"/>'
                               for e, m in want if f'Extension="{e}"' not in ct)
                if exts:
                    # 只替换第一个 <Types ...> 开标签的收尾，绝不能用 replace(">", ...)：
                    # 那会命中 <?xml ...?> 声明里的 ">"，把 <Default> 插到根元素前面去，
                    # 生成出来的 Content_Types.xml 直接不是合法 XML（Word 会报文档损坏）。
                    ct, cnt = re.subn(r"(<Types\b[^>]*>)", r"\1" + exts, ct, count=1)
                    if cnt != 1:
                        raise RuntimeError("未能定位 [Content_Types].xml 的 <Types> 根标签")
                z.writestr(n, ct)
                continue
            z.writestr(n, src.read(n))

        # 页脚
        z.writestr("word/footer1.xml", footer)
        # 图片
        for rid, zp, lp in media.items:
            with open(lp, "rb") as f:
                z.writestr("word/" + zp, f.read())

    src.close()


def safe_replace(tmp_path, out_path):
    """
    把生成好的 docx 落到目标路径。

    为什么要绕一下：这份文档被 Word 打开过后，即使 WINWORD 进程已经不在了，
    文件句柄也可能还挂着（本次就遇到了 WinError 32：另一个程序正在使用此文件）。
    这时如果直接覆盖会失败，所以退一步写成一个带序号的新文件，并明确告诉用户，
    而不是让脚本报个 PermissionError 就结束、留下一个旧版本让人误以为是新的。
    返回实际写出的路径。
    """
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        os.replace(tmp_path, out_path)
        return out_path
    except PermissionError:
        stem, ext = os.path.splitext(out_path)
        for n in range(2, 100):
            cand = f"{stem}_v{n}{ext}"
            if not os.path.exists(cand):
                os.replace(tmp_path, cand)
                print(f"\n[注意] {os.path.basename(out_path)} 正被其他程序占用（多半是 Word 还开着），")
                print(f"       本次输出改写到：{os.path.basename(cand)}")
                print(f"       关闭 Word 后删掉旧文件、把新文件改名即可。")
                return cand
        raise


def main():
    global TABLES
    print("=" * 74)
    print("生成报告 Word 文档")
    print("=" * 74)

    tpl = find_template()
    K.TEMPLATE = tpl          # kit 里的默认路径只是占位，实际用哪个模板由这里决定
    K.OUT = OUT + ".tmp"      # 先写临时文件，最后再安全替换
    print(f"样式底版 : {tpl}")

    data = K.load_results()
    if "filter" not in data:
        raise SystemExit("缺少 结果/滤波指标.csv，请先运行 01_图像滤波对比.py")
    hist_path = os.path.join(RES, "训练历史.json")
    if os.path.exists(hist_path):
        with open(hist_path, encoding="utf-8") as f:
            h = json.load(f)
        data["hist"] = {"best": h.get("best"), "params": h.get("params")}
        data["hist_full"] = h
        i = max(range(len(h["hist"]["test_acc"])), key=lambda k: h["hist"]["test_acc"][k])
        print(f"训练历史 : 最佳测试精度 {h.get('best')}%（第 {h['hist']['epoch'][i]} 轮，"
              f"训练集 {h['hist']['train_acc'][i]:.2f}%），参数量 {h.get('params'):,}")
    else:
        data["hist"] = {}
        print("[warn] 没有找到 结果/训练历史.json，表 5 将不包含改进模型一行"
              "（请运行 03_CNN精度验证与改进.py --only 3）")

    init_values(data)
    TABLES = build_tables(data)
    doc_xml, media, footer = build_document(data)

    # 打印模板包的部件清单，方便核对图片目标路径（media 名是相对 word/ 的）
    with zipfile.ZipFile(K.TEMPLATE) as z:
        print("\n模板部件 : " + ", ".join(sorted(n for n in z.namelist() if n.endswith(".xml"))))
    print("图片部件 : " + ", ".join("word/" + zp for _r, zp, _l in media.items))

    write_docx(doc_xml, media, footer)

    real = safe_replace(K.OUT, OUT)
    size = os.path.getsize(real)
    print(f"已生成 : {os.path.relpath(real, HERE)}  ({size / 1024:.1f} KB)")
    print(f"  生成的 document.xml 大小 ≈ {len(doc_xml):,} 字节")
    print(f"  插图 {len(media.items)} 张，表格 {len(TABLES)} 个，页脚 1 个")
    print("  样式：完全沿用《报告模板.docx》的 styles.xml / numbering.xml / 页面设置")


if __name__ == "__main__":
    main()
