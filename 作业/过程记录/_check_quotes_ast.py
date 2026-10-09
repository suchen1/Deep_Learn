# -*- coding: utf-8 -*-
"""
用 AST 检查：正文内容里到底还有没有 ASCII 直引号。
这才是正确的检查方式——直接看**逻辑字符串**，
而不是在源码行上做正则（那会被 Python 的相邻字符串拼接骗到）。
"""
import ast
import re

P = r"D:\ML_Project\Deep Learn\作业\_report_content.py"
tree = ast.parse(open(P, encoding="utf-8").read())

code_ranges = []
for node in ast.walk(tree):
    if (isinstance(node, ast.Tuple) and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant) and node.elts[0].value == "code"
            and isinstance(node.elts[1], ast.List)):
        code_ranges.append((node.lineno, node.end_lineno))

def in_code(ln):
    return any(a <= ln <= b for a, b in code_ranges)

bad = []
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if in_code(node.lineno):
            continue
        if '"' in node.value:
            bad.append((node.lineno, node.value))

print(f"正文（非代码块）里含 ASCII 直引号的字符串: {len(bad)} 个")
for ln, v in bad:
    i = v.find('"')
    print(f"  行 {ln}: ...{v[max(0,i-30):i+40]}...")

# 统计中文弯引号用了多少
n_curly = sum(v.count("\u201c") for node in ast.walk(tree)
              if isinstance(node, ast.Constant) and isinstance(node.value, str)
              and not in_code(node.lineno)
              for v in [node.value])
print(f"\n正文里中文弯引号 “ 共 {n_curly} 个")
