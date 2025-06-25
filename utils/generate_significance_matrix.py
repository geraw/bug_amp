
"""
generate_significance_matrix.py

Compares four optimisation / search methods (Ens_best, BF_best, SA_best, GA_best)
pair‑wise using one‑sided Wilcoxon signed‑rank tests, shows *who wins* and how
strong the evidence is.

Outputs
-------
1.  <excel_basename>_sig.tex  – LaTeX (TikZ/PGFPlots) coloured matrix
2.  <excel_basename>_sig.png  – PNG preview (same colours)

Colour scheme
-------------
* Green  – column method significantly BETTER (lower score) than row   (p < α)
* Red    – row    method significantly BETTER than column              (p < α)
* Grey   – no significance (p ≥ α)
Shade intensity encodes p‑value: darker ⇒ stronger evidence.

Usage
-----
    python generate_significance_matrix.py results.xlsx

Assumptions
-----------
* Excel sheet #1
* Row identifier column A:
    - 'AVRG'  rows contain mean metric values
    - 'STD'   rows contain std‑dev values  (not used here)
* Method columns:
    C → Ens_best   (formerly Ans_best / Ensemble)
    L → BF_best
    O → SA_best
    P → GA_best
* Lower metric values are *better* (e.g. failure probability).

You can adjust `COL_LETTERS` and `METHOD_LABELS` below if your file differs.
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
ALPHA = 0.05
COL_LETTERS = ['C', 'L', 'O', 'P']
METHOD_LABELS = ['Ens_best', 'BF_best', 'SA_best', 'GA_best']


def load_method_arrays(excel_path, col_letters):
    """Return list of numpy arrays (one per method) with paired observations."""
    df_raw = pd.read_excel(excel_path, header=None)
    avrg_rows = df_raw[1] == 'AVRG'
    method_arrays = []
    for col_letter in col_letters:
        col_idx = ord(col_letter.upper()) - ord('A')
        series = df_raw.loc[avrg_rows, col_idx].astype(float).to_numpy()
        method_arrays.append(series)
    return method_arrays


def compute_matrix(method_arrays):
    """Compute p‑values and directions."""
    n = len(method_arrays)
    pvals = np.ones((n, n))
    direction = np.zeros((n, n), dtype=int)  # +1 col better, -1 row better
    for i, j in itertools.combinations(range(n), 2):
        row = method_arrays[i]
        col = method_arrays[j]
        # two one‑sided Wilcoxon tests
        stat_right, p_right = wilcoxon(col - row, alternative='less')     # col better
        stat_left,  p_left  = wilcoxon(col - row, alternative='greater')  # row better
        if p_right < p_left:
            pvals[i, j] = p_right
            pvals[j, i] = p_right
            direction[i, j] = +1
            direction[j, i] = -1
        else:
            pvals[i, j] = p_left
            pvals[j, i] = p_left
            direction[i, j] = -1
            direction[j, i] = +1
    return pvals, direction


def rgb_with_intensity(base_color, intensity):
    """Return RGBA tuple scaled by intensity (0..1)."""
    r, g, b = base_color
    return (r, g, b, intensity)


def plot_matrix(pvals, direction, labels, png_path, alpha=ALPHA):
    n = len(labels)
    cell_size = 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    # base colours
    green_base = (0.0, 0.8, 0.0)
    red_base   = (0.8, 0.0, 0.0)
    grey_base  = (0.7, 0.7, 0.7)

    for i in range(n):
        for j in range(n):
            x, y = j * cell_size, (n - 1 - i) * cell_size
            if i == j:
                color = (*grey_base, 0.4)
                txt = '—'
            else:
                p = pvals[i, j]
                txt = f"{p:.3f}"
                if p < alpha:
                    if direction[i, j] == +1:
                        intensity = min(0.85, 0.25 + (alpha - p) / alpha * 0.6)
                        color = rgb_with_intensity(green_base, intensity)
                    else:
                        intensity = min(0.85, 0.25 + (alpha - p) / alpha * 0.6)
                        color = rgb_with_intensity(red_base, intensity)
                else:
                    color = (*grey_base, 0.3)

            ax.add_patch(
                plt.Rectangle((x, y), cell_size, cell_size,
                              facecolor=color, edgecolor='black'))
            ax.text(x + cell_size/2, y + cell_size/2, txt,
                    ha='center', va='center', fontsize=9)

    # Draw labels
    for idx, name in enumerate(labels):
        ax.text(idx * cell_size + cell_size/2, n * cell_size + 0.1,
                name, ha='center', va='bottom', fontsize=10, rotation=45)
        ax.text(-0.1, (n - 1 - idx) * cell_size + cell_size/2,
                name, ha='right', va='center', fontsize=10)

    plt.xlim(0, n * cell_size)
    plt.ylim(0, n * cell_size)
    plt.title("Pair‑wise significance (Wilcoxon one‑sided)",
              pad=50, fontsize=12)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()


def generate_latex(pvals, direction, labels, tex_path, alpha=ALPHA):
    """Write LaTeX table with coloured rectangles."""
    latex = r"""\documentclass{standalone}
\usepackage{tikz}
\usepackage{array}
\usepackage{booktabs}
\definecolor{darkgreen}{RGB}{0,150,0}
\definecolor{lightgreen}{RGB}{200,255,200}
\definecolor{darkred}{RGB}{180,0,0}
\definecolor{lightred}{RGB}{255,200,200}
\definecolor{lightgrey}{RGB}{230,230,230}
\begin{document}
\begin{tabular}{c|""" + "c" * len(labels) + "}\n"

    # header row
    latex += " & " + " & ".join(labels) + r" \\\midrule" + "\n"

    n = len(labels)
    for i in range(n):
        row_code = labels[i]
        for j in range(n):
            if i == j:
                cell = r"\cellcolor{lightgrey}--"
            else:
                p = pvals[i, j]
                dir_ = direction[i, j]
                if p < alpha:
                    # choose colour based on winner
                    if dir_ == +1:
                        shade = 'darkgreen' if p < 0.01 else 'lightgreen'
                    else:
                        shade = 'darkred' if p < 0.01 else 'lightred'
                    cell = rf"\cellcolor{{{shade}}}{p:.3f}"
                else:
                    cell = rf"\cellcolor{{lightgrey}}{p:.3f}"
            row_code += " & " + cell
        latex += row_code + r" \\" + "\n"

    latex += r"""\bottomrule
\end{tabular}
\end{document}
"""
    with open(tex_path, 'w') as f:
        f.write(latex)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_significance_matrix.py path/to/results.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    if not os.path.isfile(excel_path):
        print("Error: File not found.")
        sys.exit(1)

    base = os.path.splitext(excel_path)[0]
    out_png = base + "_sig.png"
    out_tex = base + "_sig.tex"

    method_arrays = load_method_arrays(excel_path, COL_LETTERS)
    if len(set(len(arr) for arr in method_arrays)) != 1:
        print("Error: All method arrays must have the same length (paired observations).")
        sys.exit(1)

    pvals, direction = compute_matrix(method_arrays)
    plot_matrix(pvals, direction, METHOD_LABELS, out_png)
    generate_latex(pvals, direction, METHOD_LABELS, out_tex)

    print("Generated:")
    print(" -", out_png)
    print(" -", out_tex)


if __name__ == "__main__":
    main()
