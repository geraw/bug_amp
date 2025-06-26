#!/usr/bin/env python
"""
excel_to_plot_latex_dual_updated.py
-----------------------------------
• One Excel file OR all *.xls[x] in a folder.
• Two PNGs per workbook: 5th-best & 10th-best.
• ONE LaTeX file containing ALL figures.
   – Single workbook  →  <base>_dual.tex
   – Directory        →  excel_to_plot_latex_dual_updated.tex
Columns (0-index):
   x-axis = A(0)
   Ens    = C(2)   (5th) / D(3)   (10th)
   BF     = M(12)  (5th) / N(13)  (10th)
"""

import os, sys, glob
import pandas as pd
import matplotlib.pyplot as plt

# ------------- configuration ----------------------------------------
FIFTH_COLS   = [2, 12]   # Ens, BF (C, M)
TENTH_COLS   = [3, 13]   # Ens, BF (D, N)
METHODS      = ["Ens", "BF"]
MARKERS      = ['o', 'x']
COLORS       = ['blue', 'orange']
TITLE_5TH    = "5th-Best Test Cases"
TITLE_10TH   = "10th-Best Test Vectors"

# --------------------------------------------------------------------
def list_excels(path: str):
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.endswith(('.xls', '.xlsx'))]
    return [path] if path.endswith(('.xls', '.xlsx')) else []

def load(path: str, cols):
    df   = pd.read_excel(path, header=None)
    avg  = df[1] == "AVRG"
    std  = df[1] == "STD"
    x    = df.loc[avg, 0].astype(float).to_numpy()
    y,e  = [], []
    for c in cols:
        y.append(df.loc[avg, c].astype(float).to_numpy())
        e.append(df.loc[std, c].astype(float).to_numpy())
    return x, y, e

def caption_token(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split('_')
    return parts[1] if len(parts) > 1 else base

# -------------------- PNG plotting ----------------------------------
def save_png(x, y, e, out_png, title):
    plt.figure(figsize=(10,6))
    for i, m in enumerate(METHODS):
        plt.errorbar(x, y[i], yerr=e[i], marker=MARKERS[i],
                     color=COLORS[i], label=m, capsize=3)
    plt.ylim(0,1); plt.grid(True)
    ticks = list(range(0, int(max(x))+501, 500))
    plt.xticks(ticks, rotation=45)
    plt.xlabel("number of test cases"); plt.ylabel("Probability")
    plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

# -------------- TikZ/PGF figure block -------------------------------
def tikz_block(x, y, e, title, caption):
    pgf = {'o':'*', 'x':'x'}
    ticks = list(range(0, int(max(x))+501, 500))
    lines = [
        r"\pgfplotsset{compat=1.3,width=0.8\columnwidth}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\vspace{3ex}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"  width=0.8\columnwidth,",
        r"  height=7cm,",
        r"  ymin=0, ymax=1,",
        f"  xtick={{{','.join(map(str,ticks))}}},",
        f"  xticklabels={{{','.join(map(str,ticks))}}},",
        r"  xlabel={number of test cases},",
        r"  ylabel={Probability},",
        r"  title={" + title + "},",
        r"  grid=major,",
        r"  legend style={at={(1.05,1)},anchor=north west},",
        r"  error bars/y dir=both,",
        r"  error bars/y explicit]",
    ]
    for i, m in enumerate(METHODS):
        lines.append(
            rf"\addplot+[mark={pgf[MARKERS[i]]},color={COLORS[i]},"
            r"error bars/.cd,y dir=both,y explicit] coordinates {")
        for xi, yi, ei in zip(x, y[i], e[i]):
            yi = max(0, min(1, yi)); ei = min(ei, 1-yi)
            lines.append(f"({int(xi)}, {yi}) +- (0, {ei})")
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

# --------------------------- main -----------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex_dual_updated.py <xlsx | directory>")
        sys.exit(1)

    target = sys.argv[1]
    sheets = list_excels(target)
    if not sheets:
        print("No Excel files found."); sys.exit(1)

    out_dir = target if os.path.isdir(target) else os.path.dirname(target) or "."
    master_blocks = []

    for xl in sheets:
        token = caption_token(xl)

        # 5th best
        x5, y5, e5 = load(xl, FIFTH_COLS)
        png5 = os.path.join(out_dir, os.path.splitext(os.path.basename(xl))[0] + "_5th_best_cases.png")
        save_png(x5, y5, e5, png5, TITLE_5TH)
        master_blocks.append(tikz_block(x5, y5, e5, TITLE_5TH, f"{token} (5th)"))

        # 10th best
        x10, y10, e10 = load(xl, TENTH_COLS)
        png10 = os.path.join(out_dir, os.path.splitext(os.path.basename(xl))[0] + "_10th_best_vectors.png")
        save_png(x10, y10, e10, png10, TITLE_10TH)
        master_blocks.append(tikz_block(x10, y10, e10, TITLE_10TH, f"{token} (10th)"))

        # divider line
        master_blocks.append("%----------------------------------------\n")

        print("PNGs:", png5, "and", png10)

    # choose master tex filename
    if len(sheets) == 1:
        stem = os.path.splitext(os.path.basename(sheets[0]))[0] + "_dual.tex"
    else:
        stem = os.path.splitext(os.path.basename(__file__))[0] + ".tex"
    master_path = os.path.join(out_dir, stem)

    with open(master_path, "w", encoding="utf-8") as f:
        f.write("\n".join(master_blocks))

    print("LaTeX master written:", master_path)

if __name__ == "__main__":
    main()
