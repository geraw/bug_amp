#!/usr/bin/env python
"""
excel_to_plot_latex_best_4methods.py
------------------------------------
• One Excel file OR all *.xls[x] in a directory.
• Produces <base>_best.png   (linear x-axis)
• Produces one LaTeX file in the SAME folder:
      • single Excel  → <base>_best.tex
      • multiple      → excel_to_plot_latex_best_4methods.tex
Columns: x-axis = A(0); Ens=C(2), BF=L(11), SA=O(14), GA=P(15)
"""

import os, sys
import pandas as pd
import matplotlib.pyplot as plt

# ----- fixed config --------------------------------------------------
COLS    = [2, 11, 14, 15]              # Ens, BF, SA, GA
METHODS = ["Ens", "BF", "SA", "GA"]
MARKERS = ['o', 'x', 's', '^']
COLORS  = ['blue', 'orange', 'green', 'red']
TITLE   = "Best Results — Ens, BF, SA, GA"

# --------------------------------------------------------------------
def excel_files(path: str):
    """Return list of .xls/.xlsx paths (single file or all in dir)."""
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.endswith(('.xls', '.xlsx'))]
    return [path] if path.endswith(('.xls', '.xlsx')) else []

def load_excel(path: str):
    """Extract x, y-series, e-series from workbook."""
    df  = pd.read_excel(path, header=None)
    avg = df[1] == "AVRG"
    std = df[1] == "STD"
    x   = df.loc[avg, 0].astype(float).to_numpy()          # column A
    y,e = [], []
    for c in COLS:
        y.append(df.loc[avg, c].astype(float).to_numpy())
        e.append(df.loc[std, c].astype(float).to_numpy())
    return x, y, e

def caption_from_name(path: str) -> str:
    base  = os.path.splitext(os.path.basename(path))[0]
    parts = base.split('_')
    return parts[1] if len(parts) >= 2 else base

# -------------------- PNG plot --------------------------------------
def save_png(x, y, e, out_png):
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(METHODS):
        plt.errorbar(x, y[i], yerr=e[i],
                     marker=MARKERS[i], color=COLORS[i],
                     label=m, capsize=3)
    plt.ylim(0, 1)
    plt.grid(True)
    xticks = list(range(0, int(max(x)) + 501, 500))
    plt.xticks(xticks, rotation=45)
    plt.xlabel("number of test cases")
    plt.ylabel("Probability")
    plt.title(TITLE)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ------------------ PGFPlots figure block ---------------------------
def tikz_block(x, y, e, caption):
    pgf = {'o': '*', 'x': 'x', 's': 'square*', '^': 'triangle*'}
    xticks = list(range(0, int(max(x)) + 501, 500))
    xtick_line = "xtick={" + ",".join(map(str, xticks)) + "},"
    xtlbl_line = "xticklabels={" + ",".join(map(str, xticks)) + "},"

    block = [
        r"\pgfplotsset{compat=1.3,width=0.8\columnwidth}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\vspace{3ex}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.8\columnwidth,",
        r"    height=7cm,",
        r"    ymin=0, ymax=1,",
        xtick_line,
        xtlbl_line,
        r"    xlabel={number of test cases},",
        r"    ylabel={Probability},",
        r"    grid=major,",
        r"    legend style={at={(1.05,1)},anchor=north west},",
        r"    error bars/y dir=both,",
        r"    error bars/y explicit]",
    ]
    for i, m in enumerate(METHODS):
        block.append(
            rf"\addplot+[mark={pgf[MARKERS[i]]},color={COLORS[i]},"
            r"error bars/.cd,y dir=both,y explicit] coordinates {")
        for xi, yi, ei in zip(x, y[i], e[i]):
            yi = max(0, min(1, yi))
            ei = min(ei, 1 - yi)
            block.append(f"({int(xi)}, {yi}) +- (0, {ei})")
        block.append("};")
        block.append(rf"\addlegendentry{{{m}}}")
    block.extend([
        r"\end{axis}",
        r"\end{tikzpicture}",
        rf"\caption{{{caption}}}",
        rf"\label{{fig:{caption.replace(' ', '_')}}}",
        r"\end{figure}",
        ""  # blank line for spacing
    ])
    return "\n".join(block)

# ---------------------- Main processing -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex_best_4methods.py <xlsx or dir>")
        sys.exit(1)

    input_arg = sys.argv[1]
    sheets    = excel_files(input_arg)
    if not sheets:
        print("No Excel files found.")
        sys.exit(1)

    # Output directory = same folder that holds the sheets
    out_dir = input_arg if os.path.isdir(input_arg) else os.path.dirname(input_arg) or "."
    all_figures = []

    for sheet in sheets:
        x, y, e = load_excel(sheet)

        base_no_ext = os.path.splitext(os.path.basename(sheet))[0]
        png_name    = base_no_ext + "_best.png"
        png_path    = os.path.join(out_dir, png_name)
        save_png(x, y, e, png_path)
        print("PNG:", png_path)

        caption = caption_from_name(sheet)
        all_figures.append(tikz_block(x, y, e, caption))

    # Master LaTeX filename logic
    if len(sheets) == 1:
        master_name = base_no_ext + "_best.tex"
    else:
        script_stem = os.path.splitext(os.path.basename(__file__))[0]
        master_name = script_stem + ".tex"

    master_path = os.path.join(out_dir, master_name)
    with open(master_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_figures))
    print("LaTeX:", master_path)

# --------------------------------------------------------------------
if __name__ == "__main__":
    main()
