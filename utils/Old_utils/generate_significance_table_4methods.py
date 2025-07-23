
"""
generate_significance_table_4methods.py

Creates:
    • LaTeX (.tex) directional significance matrix
    • PNG preview (optional via --png)

Method set (4):
    Ens_best   – column C
    BF_best    – column L   (Brute Force)
    SA_best    – column O   (Simulated Annealing)
    GA_best    – column P   (Genetic Algorithm)

Statistical test:
    One‑sided Wilcoxon signed‑rank (paired)
    α = 0.05

Colour logic:
    🟩 Green  – column is significantly better than row (p < α)
    🟥 Red    – row is significantly better than column (p < α)
    ⚪ Gray   – not significant

Shade intensity:
    Darker shade → smaller p‑value (stronger evidence)

Usage
-----
    python generate_significance_table_4methods.py results.xlsx --png
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ALPHA = 0.05
COL_LETTERS = ["C", "L", "O", "P"]  # Ens, BF, SA, GA
LABELS = ["Ens_best", "BF_best", "SA_best", "GA_best"]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_columns(excel_path, col_letters):
    df = pd.read_excel(excel_path, header=None)
    avrg_rows = df[1] == "AVRG"
    data = []
    for letter in col_letters:
        idx = ord(letter.upper()) - ord("A")
        data.append(df.loc[avrg_rows, idx].astype(float).to_numpy())
    return data


def wilcoxon_directional(method_data):
    n = len(method_data)
    matrix = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = "diag"
            else:
                diff = method_data[j] - method_data[i]
                p_col_better = wilcoxon(diff, alternative="less").pvalue   # column better
                p_row_better = wilcoxon(diff, alternative="greater").pvalue  # row better

                if p_col_better < p_row_better:
                    direction = "green" if p_col_better < ALPHA else "gray"
                    p_val = p_col_better
                else:
                    direction = "red" if p_row_better < ALPHA else "gray"
                    p_val = p_row_better

                matrix[i][j] = (direction, f"{p_val:.3f}", p_val)
    return matrix


def p_to_intensity(p, alpha=ALPHA, min_pct=30, max_pct=80):
    if p >= alpha:
        return min_pct
    frac = (alpha - p) / alpha
    return int(min_pct + frac * (max_pct - min_pct))


# ----------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------
HEADER = r"""
\documentclass{article}
\usepackage{colortbl}
\usepackage{tikz}
\usepackage{adjustbox}
\usepackage{array}
\begin{document}
\begin{table}[h!]
\centering
\caption{Directional Wilcoxon significance matrix ($p<0.05$)}
\begin{adjustbox}{max width=\textwidth}
\renewcommand{\arraystretch}{1.5}
"""
FOOTER = r"""\end{adjustbox}
\end{table}
\end{document}
"""

def matrix_to_latex(matrix, labels, tex_path):
    n = len(labels)
    latex = HEADER
    latex += r"\begin{tabular}{c|" + "c"*n + "}\n"
    latex += " & " + " & ".join(labels) + r" \\\hline" + "\n"

    for i, row in enumerate(matrix):
        line = labels[i]
        for j, cell in enumerate(row):
            if cell == "diag":
                line += " & "
            else:
                color, ptxt, pval = cell
                if color == "gray":
                    tikz = rf"\tikz[baseline=(char.base)]{{\node[fill=lightgray,rounded corners=0.5mm,inner sep=5pt] (char) {{{ptxt}}};}}"
                else:
                    base = "green" if color == "green" else "red"
                    pct = p_to_intensity(pval)
                    tikz = rf"\tikz[baseline=(char.base)]{{\node[fill={base}!{pct}!white,rounded corners=0.5mm,inner sep=5pt] (char) {{{ptxt}}};}}"
                line += f" & {tikz}"
        latex += line + r" \\" + "\n"

    latex += r"\end{tabular}" + FOOTER
    with open(tex_path, "w") as f:
        f.write(latex)


# ----------------------------------------------------------------------
# PNG
# ----------------------------------------------------------------------
def matrix_to_png(matrix, labels, out_path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(4 + n, 4 + n))
    ax.axis("off")

    cell = 1.0
    for i in range(n):
        for j in range(n):
            x, y = j*cell, (n-1-i)*cell
            if matrix[i][j] == "diag":
                color = (0.95, 0.95, 0.95, 1)
                txt = ""
            else:
                clr, txt, pval = matrix[i][j]
                if clr == "gray":
                    color = (0.9, 0.9, 0.9, 1)
                else:
                    base = np.array([0.8, 1.0, 0.8]) if clr == "green" else np.array([1.0, 0.8, 0.8])
                    intensity = (ALPHA - min(pval, ALPHA)) / ALPHA  # 0..1
                    # mix toward base (darker)
                    color = base - intensity * 0.4
                    color = np.clip(color, 0, 1)

                ax.add_patch(plt.Rectangle((x, y), cell, cell, facecolor=color, edgecolor='black'))
            ax.text(x+cell/2, y+cell/2, txt, ha='center', va='center', fontsize=10)

    for idx, lbl in enumerate(labels):
        ax.text(idx*cell+cell/2, n*cell+0.1, lbl, ha='center', va='bottom', fontsize=11, rotation=45)
        ax.text(-0.1, (n-1-idx)*cell+cell/2, lbl, ha='right', va='center', fontsize=11)

    ax.set_xlim(0, n*cell)
    ax.set_ylim(0, n*cell)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel", help="Excel file with results")
    parser.add_argument("--png", action="store_true", help="also save PNG preview")
    args = parser.parse_args()

    data = load_columns(args.excel, COL_LETTERS)
    matrix = wilcoxon_directional(data)

    base = os.path.splitext(args.excel)[0]
    tex_path = base + "_sig4.tex"
    matrix_to_latex(matrix, LABELS, tex_path)
    print("LaTeX saved:", tex_path)

    if args.png:
        png_path = base + "_sig4.png"
        matrix_to_png(matrix, LABELS, png_path)
        print("PNG saved:", png_path)


if __name__ == "__main__":
    main()
