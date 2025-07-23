
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import sys
import os

# Constants
COL_LETTERS = ['C', 'L', 'O', 'P']  # Ens, BF, SA, GA
METHOD_LABELS = ["Ensemble", "Brute Force", "Simulated Annealing", "Genetic Algorithm"]
ALPHA = 0.05


def letter_to_index(c):
    return ord(c.upper()) - ord('A')


def extract_data(filepath):
    df = pd.read_excel(filepath, header=None)
    avrg_rows = df[1] == 'AVRG'

    x = df.loc[avrg_rows, 0].astype(float).to_numpy()
    data = []
    for col_letter in COL_LETTERS:
        col_index = letter_to_index(col_letter)
        values = df.loc[avrg_rows, col_index].astype(float).to_numpy()
        data.append(values)
    return METHOD_LABELS, data


def format_cell(p, alpha=ALPHA):
    if np.isnan(p):
        return " "
    color = "lightgreen" if p < alpha else "lightred"
    return rf"\tikz[baseline=(char.base)]{{\node[fill={color},rounded corners=0.5mm,inner sep=4pt, font=\normalsize] (char) {{{p:.3f}}};}}"


def generate_latex_matrix(labels, data, title="Statistical significance matrix ($p < 0.05$ indicates significance)"):
    n = len(labels)
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{title}}}")
    lines.append(r"\begin{adjustbox}{max width=\textwidth}")
    lines.append(r"\renewcommand{\arraystretch}{1.5}")
    lines.append(r"\begin{tabular}{c|" + "c" * n + "}")
    lines.append(" & " + " & ".join(labels) + r" \\" + r"\hline")

    for i in range(n):
        row = [labels[i]]
        for j in range(n):
            if i == j:
                row.append(" ")
            else:
                stat, p = wilcoxon(data[i], data[j], alternative='less')  # test if col (j) < row (i)
                cell = format_cell(p)
                row.append(cell)
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_significance_matrix.py <results.xlsx>")
        return

    filepath = sys.argv[1]
    labels, data = extract_data(filepath)
    tex_code = generate_latex_matrix(labels, data)
    out_file = os.path.splitext(filepath)[0] + "_significance_table.tex"
    with open(out_file, "w") as f:
        f.write(tex_code)
    print(f"LaTeX table saved to {out_file}")


if __name__ == "__main__":
    main()
