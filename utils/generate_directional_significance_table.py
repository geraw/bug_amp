
"""
generate_directional_significance_table.py

Creates BOTH:
    1.  A LaTeX table (TikZ nodes) matching the style in the sample screenshot.
    2.  A PNG preview of the significance matrix (coloured cells with p‑values).

Key features
------------
* Methods compared:  Ens_best, BF_best, SA_best, GA_best  (columns C, F, I, L).
* One‑sided Wilcoxon signed‑rank test, directional:
      – If the **column** is statistically better (p < 0.05)  → lightgreen cell.
      – If the **row**    is statistically better (p < 0.05)  → lightred cell.
      – Otherwise                                            → lightgray cell.
* P‑values shown with three‑decimal precision.
* Vertical bar after the label column, extra row height, caption identical to sample.
* Optional PNG preview (no LaTeX compile required).

Usage
-----
    python generate_directional_significance_table.py results.xlsx [--png]

The script will output:
    results_directional_sig.tex
    results_directional_sig.png   (only if --png is given)
"""

import sys
import os
import argparse
import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_columns(excel_path, col_letters):
    df_raw = pd.read_excel(excel_path, header=None)
    method_data = []
    for col_letter in col_letters:
        col_idx = ord(col_letter.upper()) - ord('A')
        avrg_rows = df_raw[1] == 'AVRG'
        data = df_raw.loc[avrg_rows, col_idx].astype(float).to_numpy()
        method_data.append(data)
    return method_data


def wilcoxon_directional(method_data, alpha=0.05):
    """Return matrix with ('green'|'red'|'gray', p‑value str)."""
    n = len(method_data)
    matrix = [['' for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 'diag'
            else:
                diff = method_data[j] - method_data[i]
                p_col_better = wilcoxon(diff, alternative='less').pvalue
                p_row_better = wilcoxon(diff, alternative='greater').pvalue

                if p_col_better < p_row_better:
                    color = 'green' if p_col_better < alpha else 'gray'
                    pval  = p_col_better
                else:
                    color = 'red' if p_row_better < alpha else 'gray'
                    pval  = p_row_better

                matrix[i][j] = (color, f"{pval:.3f}")
    return matrix


# ------------------------------------------------------------
# LaTeX generation
# ------------------------------------------------------------
LATEX_TEMPLATE_HEADER = r"""
\documentclass{article}
\usepackage{colortbl}
\usepackage{tikz}
\usepackage{adjustbox}
\usepackage{array}
\definecolor{lightgreen}{RGB}{200,255,200}
\definecolor{lightred}{RGB}{255,200,200}
\definecolor{lightgray}{RGB}{230,230,230}
\begin{document}
\begin{table}[h!]
\centering
\caption{Directional Wilcoxon significance matrix ($p < 0.05$ indicates significance)}
\begin{adjustbox}{max width=\textwidth}
\renewcommand{\arraystretch}{1.5}
"""

LATEX_TEMPLATE_FOOTER = r"""
\end{adjustbox}
\end{table}
\end{document}
"""

def color_to_tex(color_name):
    return {
        "green": "lightgreen",
        "red":   "lightred",
        "gray":  "lightgray"
    }[color_name]

def matrix_to_latex(matrix, labels):
    n = len(labels)
    latex = LATEX_TEMPLATE_HEADER
    latex += r"\begin{tabular}{c|" + "c" * n + "}\n"
    latex += " & " + " & ".join(labels) + r" \\\hline" + "\n"

    for i, row in enumerate(matrix):
        line = labels[i]
        for j, cell in enumerate(row):
            if cell == 'diag':
                line += " & "
            else:
                color, pval = cell
                tcolor = color_to_tex(color)
                tikz_box = rf"\tikz[baseline=(char.base)]{{\node[fill={tcolor},rounded corners=0.5mm,inner sep=5pt, font=\normalsize] (char) {{{pval}}};}}"
                line += f" & {tikz_box}"
        latex += line + r" \\" + "\n"

    latex += r"\end{tabular}" + LATEX_TEMPLATE_FOOTER
    return latex


# ------------------------------------------------------------
# PNG preview with Matplotlib
# ------------------------------------------------------------
def matrix_to_png(matrix, labels, out_path):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(4 + n, 4 + n))
    ax.set_axis_off()

    cell_size = 1.0
    for i in range(n):
        for j in range(n):
            x, y = j * cell_size, (n - 1 - i) * cell_size
            if matrix[i][j] == 'diag':
                color = (0.9, 0.9, 0.9, 0.3)
                txt = ''
            else:
                col, txt = matrix[i][j]
                if col == 'green':
                    color = (0.80, 1.00, 0.80, 1)
                elif col == 'red':
                    color = (1.00, 0.80, 0.80, 1)
                else:
                    color = (0.9, 0.9, 0.9, 1)

            ax.add_patch(plt.Rectangle((x, y), cell_size, cell_size,
                                       facecolor=color, edgecolor='black'))
            ax.text(x + cell_size/2, y + cell_size/2, txt,
                    ha='center', va='center', fontsize=10)

    # draw labels
    for idx, name in enumerate(labels):
        ax.text(idx * cell_size + cell_size/2, n * cell_size + 0.1,
                name, ha='center', va='bottom', rotation=45, fontsize=11)
        ax.text(-0.1, (n - 1 - idx) * cell_size + cell_size/2,
                name, ha='right', va='center', fontsize=11)

    ax.set_xlim(0, n * cell_size)
    ax.set_ylim(0, n * cell_size)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel", help="Excel file with results")
    parser.add_argument("--png", action="store_true", help="Also output a PNG preview")
    args = parser.parse_args()

    col_letters = ['C', 'F', 'I', 'L']   # Ens, BF, SA, GA
    labels      = ["Ens_best", "BF_best", "SA_best", "GA_best"]

    data = load_columns(args.excel, col_letters)
    matrix = wilcoxon_directional(data)

    base = os.path.splitext(args.excel)[0]
    tex_path = base + "_directional_sig.tex"
    with open(tex_path, "w") as f:
        f.write(matrix_to_latex(matrix, labels))
    print("LaTeX saved:", tex_path)

    if args.png:
        png_path = base + "_directional_sig.png"
        matrix_to_png(matrix, labels, png_path)
        print("PNG saved:", png_path)


if __name__ == "__main__":
    main()
