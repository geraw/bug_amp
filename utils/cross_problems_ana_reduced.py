
"""
cross_problems_ana.py  – updated (Ens, BF, SA, GA)

Changes applied:
    1. Dropped 'Classifier' and 'MLP' series entirely.
    2. Renamed 'Ans' → 'Ens' (ensemble).
    3. Each method now has a unique marker (o, x, s, ^) in both PNG and LaTeX outputs.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_graph(directory, case_size):
    # Updated target methods
    target_methods = ["Ens", "BF", "SA", "GA"]
    method_avg = {method: [] for method in target_methods}
    method_std = {method: [] for method in target_methods}
    problem_labels = []

    # Get all Excel files in the directory
    problem_files = [(f.split("_")[1], os.path.join(directory, f))
                     for f in sorted(os.listdir(directory)) if f.endswith(".xlsx")]

    # Read data from each file
    for name, path in problem_files:
        df = pd.read_excel(path, sheet_name="גיליון1")
        match_indices = df.index[df.iloc[:, 0] == case_size].tolist()

        if not match_indices:
            for method in target_methods:
                method_avg[method].append(np.nan)
                method_std[method].append(0.0)
            continue

        i = match_indices[0]
        avg_row = df.iloc[i]
        std_row = df.iloc[i + 1]
        problem_labels.append(name)

        for method in target_methods:
            matching_cols = [col for col in df.columns if method in col and "_best" in col]
            if matching_cols:
                col = matching_cols[0]
                try:
                    avg = float(avg_row[col])
                    std = float(std_row[col])
                except:
                    avg, std = np.nan, 0.0
            else:
                avg, std = np.nan, 0.0
            method_avg[method].append(avg)
            method_std[method].append(std)

    # Plot PNG
    x_vals = np.arange(len(problem_labels))
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = cm.tab10(np.linspace(0, 1, len(target_methods)))
    markers = ['o', 'x', 's', '^']  # Unique markers

    for i, method in enumerate(target_methods):
        ax.errorbar(
            x_vals, method_avg[method],
            yerr=method_std[method],
            fmt=f'-{markers[i]}',
            color=colors[i],
            label=method,
            capsize=5
        )

    ax.set_xticks(x_vals)
    ax.set_xticklabels(problem_labels, rotation=45)
    ax.set_ylabel("Probability (Average)")
    ax.set_title(f"All Problems – {case_size} Test Cases")
    ax.legend(title="Methods", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"method_performance_{case_size}.png"), dpi=300)
    plt.close()

    # Generate LaTeX
    latex_code = r"""\documentclass{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=14cm, height=7cm,
    xlabel={Problem},
    ylabel={Probability (Average)},
    xtick=data,
    xticklabels={""" + ",".join(problem_labels) + r"""},
    xticklabel style={rotate=45, anchor=east},
    legend style={at={(1.05,1)}, anchor=north west},
    grid=both,
    error bars/y dir=both,
    error bars/y explicit
]
"""

    latex_markers = ['*', 'x', 'square*', 'triangle*']  # pgfplots marker styles

    for idx, method in enumerate(target_methods):
        coords = [f"({i},{method_avg[method][i]}) +- (0,{method_std[method][i]})"
                  for i in range(len(problem_labels))]
        latex_code += (
            f"\\addplot+[mark={latex_markers[idx]}, error bars/.cd, y dir=both, y explicit] "
            f"coordinates {{ {' '.join(coords)} }};\n"
        )
        latex_code += f"\\addlegendentry{{{method.replace('_', '\\_')}}}\n"

    latex_code += r"""\end{axis}
\end{tikzpicture}
\end{document}"""

    with open(os.path.join(directory, f"method_performance_{case_size}.tex"), "w") as f:
        f.write(latex_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="Directory with Excel files")
    parser.add_argument("case_size", type=int, help="Number of test cases (e.g., 500, 1100, 3900)")
    args = parser.parse_args()
    generate_graph(args.directory, args.case_size)
