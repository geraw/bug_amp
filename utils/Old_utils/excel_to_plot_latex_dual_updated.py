#!/usr/bin/env python
"""
excel_to_plot_latex_dual_updated.py
-----------------------------------
• Accepts one Excel workbook *or* a directory.
• For each workbook writes:
      <base>_5th_best_cases.png
      <base>_10th_best_vectors.png
• Builds ONE LaTeX file in the same folder.

Column map (0-based):
    x-axis   : 0   (A)
    Ens-5th  : 3   (D)
    Ens-10th : 4   (E)
    BF-5th   : 12  (M)
    BF-10th  : 13  (N)

Any “no-data” cell (e.g. '#DIV/0!') is converted to 0 .0.
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ────────── configuration ───────────────────────────────────────────
FIFTH_COLS  = [3, 12]           # Ens-5th, BF-5th
TENTH_COLS  = [4, 13]           # Ens-10th, BF-10th
METHODS     = ["Ens", "BF"]
MARKERS     = ['o', 'x']
COLORS      = ['blue', 'orange']
TITLE_5TH   = "5th-Best Test Cases"
TITLE_10TH  = "10th-Best Test Vectors"

# ────────── helpers ────────────────────────────────────────────────
def excel_list(p):
    if os.path.isdir(p):
        return [os.path.join(p, f) for f in sorted(os.listdir(p))
                if f.lower().endswith(('.xls', '.xlsx'))]
    return [p] if p.lower().endswith(('.xls', '.xlsx')) else []

BAD_TOKENS = ['DIV/0!', '#DIV/0!', 'N/A', '#N/A', 'NaN', 'nan', None, '']

def to_float_array(val_or_ser):
    ser = (val_or_ser if isinstance(val_or_ser, pd.Series)
           else pd.Series(val_or_ser if isinstance(val_or_ser, (list, tuple, np.ndarray))
                          else [val_or_ser]))
    ser = ser.replace(BAD_TOKENS, '0')
    ser = pd.to_numeric(ser, errors='coerce').fillna(0.0)
    return ser.astype(float).to_numpy()

def load_excel(path, cols):
    df = pd.read_excel(path, header=None)
    lab = df.iloc[:, 1].astype(str).str.strip().str.upper()
    avg = lab.eq("AVRG")
    std = lab.eq("STD")

    x   = to_float_array(df.loc[avg, 0])
    y, e = [], []
    for c in cols:
        y.append(to_float_array(df.loc[avg, c]))
        e.append(to_float_array(df.loc[std, c]))
    return x, y, e

def caption_token(path):
    parts = os.path.splitext(os.path.basename(path))[0].split('_')
    return parts[1] if len(parts) > 1 else parts[0]

# ────────── PNG plot ───────────────────────────────────────────────
def plot_png(x, y, err, out_png, title):
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(METHODS):
        plt.errorbar(x, y[i], yerr=err[i],
                     fmt=f'-{MARKERS[i]}', color=COLORS[i],
                     label=m, capsize=3)
    plt.ylim(0, 1); plt.grid(True)
    ticks = list(range(0, int(max(x)) + 501, 500))
    plt.xticks(ticks, rotation=45)
    plt.xlabel("number of test cases"); plt.ylabel("Probability")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

# ────────── TikZ / pgfplots block ─────────────────────────────────
def tikz_block(x, y, err, title, caption):
    pgfmark = {'o': '*', 'x': 'x'}
    ticks = list(range(0, int(max(x)) + 501, 500))

    lines = [
        r"\pgfplotsset{compat=1.3,width=0.8\columnwidth}",
        r"\begin{figure}[H]\centering\vspace{3ex}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[width=0.8\columnwidth,height=7cm,",
        r"  ymin=0,ymax=1,",
        f"  xtick={{{','.join(map(str, ticks))}}},",
        f"  xticklabels={{{','.join(map(str, ticks))}}},",
        r"  xlabel={number of test cases},ylabel={Probability},",
        rf"  title={{{title}}},grid=major,",
        r"  legend style={at={(1.05,1)},anchor=north west},",
        r"  error bars/y dir=both,error bars/y explicit]",
    ]

    for i, m in enumerate(METHODS):
        lines.append(
            rf"\addplot+[mark={pgfmark[MARKERS[i]]},color={COLORS[i]},"
            r"error bars/.cd,y dir=both,y explicit] coordinates {")
        for xi, yi, ei in zip(x, y[i], err[i]):
            yi = 0.0 if np.isnan(yi) else min(max(float(yi), 0.0), 1.0)
            ei = 0.0 if np.isnan(ei) else max(float(ei), 0.0)
            lines.append(f"({int(xi)},{yi}) +- (0,{ei})")
        lines.append("};")
        lines.append(rf"\addlegendentry{{{m}}}")
    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{{caption}}}",
        rf"\label{{fig:{caption.replace(' ','_')}}}",
        r"\end{figure}",
    ]
    return "\n".join(lines)

# ────────── main ───────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex_dual_updated.py <xlsx | directory>")
        sys.exit(1)

    target = sys.argv[1]
    sheets = excel_list(target)
    if not sheets:
        print("No Excel files found."); sys.exit(1)

    out_dir = target if os.path.isdir(target) else os.path.dirname(target) or "."
    tex_blocks = []

    for xl in sheets:
        token = caption_token(xl)

        # 5-th best
        x5, y5, e5 = load_excel(xl, FIFTH_COLS)
        png5 = os.path.join(out_dir,
                os.path.splitext(os.path.basename(xl))[0] + "_5th_best_cases.png")
        plot_png(x5, y5, e5, png5, TITLE_5TH)
        tex_blocks.append(tikz_block(x5, y5, e5, TITLE_5TH, f"{token} (5th)"))

        # 10-th best
        x10, y10, e10 = load_excel(xl, TENTH_COLS)
        png10 = os.path.join(out_dir,
                 os.path.splitext(os.path.basename(xl))[0] + "_10th_best_vectors.png")
        plot_png(x10, y10, e10, png10, TITLE_10TH)
        tex_blocks.append(tikz_block(x10, y10, e10, TITLE_10TH, f"{token} (10th)"))

        tex_blocks.append("%----------------------------------------\n")
        print("Generated:", png5, "and", png10)

    tex_name = (os.path.splitext(os.path.basename(sheets[0]))[0] + "_dual.tex"
                if len(sheets) == 1
                else os.path.splitext(os.path.basename(__file__))[0] + ".tex")
    tex_path = os.path.join(out_dir, tex_name)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_blocks))
    print("LaTeX master written:", tex_path)

if __name__ == "__main__":
    main()
