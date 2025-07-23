import sys
import os
import pandas as pd
import scipy.stats as stats

def load_columns(excel_path, col_letters):
    df_raw = pd.read_excel(excel_path, header=None)
    method_data = []
    for col_letter in col_letters:
        col_idx = ord(col_letter.upper()) - ord('A')
        avrg_rows = df_raw[1] == 'AVRG'
        data = df_raw.loc[avrg_rows, col_idx].astype(float).to_numpy()
        method_data.append(data)
    return method_data

def significance_matrix(method_data, alpha=0.05):
    n = len(method_data)
    sig_matrix = [['' for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                sig_matrix[i][j] = 'diag'
                continue
            t_stat, p_val = stats.ttest_rel(method_data[i], method_data[j])
            sig_matrix[i][j] = 'green' if p_val < alpha else 'red'
    return sig_matrix

def generate_latex_table(matrix, labels, output_path):
    color_defs = r"""
\usepackage{colortbl}
\definecolor{lightgreen}{RGB}{200,255,200}
\definecolor{lightred}{RGB}{255,200,200}
"""
    latex = r"""\documentclass{article}
\usepackage{tikz}
""" + color_defs + r"""
\usepackage{adjustbox}
\begin{document}
\begin{table}[h!]
\centering
\caption{Statistical significance matrix ($p < 0.05$ indicates significance)}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{c|""" + "c" * len(labels) + "}\n"

    # Header
    latex += " & " + " & ".join(labels) + r" \\\hline" + "\n"

    for i, row in enumerate(matrix):
        line = labels[i]
        for j, cell in enumerate(row):
            if cell == 'diag':
                line += " & "
            else:
                color = "lightgreen" if cell == 'green' else "lightred"
                line += f" & \\cellcolor{{{color}}}\\tikz\\draw[fill={color}, draw=none] (0,0) rectangle (0.3,0.3);"
        latex += line + r" \\" + "\n"

    latex += r"""\end{tabular}
\end{adjustbox}
\end{table}
\end{document}
"""
    with open(output_path, 'w') as f:
        f.write(latex)

def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_significance_table.py path/to/table.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]
    base_name = os.path.splitext(excel_path)[0]
    output_tex = base_name + "_sig.tex"

    col_letters = ['C', 'F', 'I', 'L', 'O', 'P']
    method_labels = [
        "Assemble classification", "Random Forest", "MLP Classifier",
        "Brute Force", "Simulated Annealing", "Genetic Algorithm"
    ]

    method_data = load_columns(excel_path, col_letters)
    matrix = significance_matrix(method_data)
    generate_latex_table(matrix, method_labels, output_tex)

    print("LaTeX file generated:", output_tex)

if __name__ == "__main__":
    main()
