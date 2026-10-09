import matplotlib.font_manager as fm
names = sorted({f.name for f in fm.fontManager.ttflist})
want = [n for n in names if any(k in n for k in ("SimHei","Hei","Song","Kai","YaHei","Times","DejaVu","Arial"))]
with open(r"D:\ML_Project\Deep Learn\作业\_fonts.txt","w",encoding="utf-8") as fh:
    fh.write("\n".join(want))
print("count", len(names))
