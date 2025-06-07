import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_method_graphs(directory, num_test_cases, output_png='method_performance.png', output_tex='method_performance.tex'):
    method_results = {}
    problem_order = []

    # Scan all Excel files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            problem_name = filename.split("_")[1]  # Extract nickname
            problem_order.append(problem_name)
            full_path = os.path.join(directory, filename)
            df = pd.read_excel(full_path, sheet_name="גיליון1")

            for i in range(0, min(len(df), num_test_cases * 2), 2):
                if i + 1 >= len(df):
                    continue
                row_avg = df.iloc[i]
                row_std = df.iloc[i + 1]
                for col in df.columns[2:]:
                    method = col
                    if method not in method_results:
                        method_results[method] = {"problem": [], "avg": [], "std": []}
                    method_results[method]["problem"].append(problem_name)
                    method_results[method]["avg"].append(row_avg[col])
                    method_results[method]["std"].append(row_std[col])

    # Sort problems for smooth graph
    problem_order = sorted(set(problem_order))
    sorted_methods = sorted(method_results.keys())

    # Plot PNG
    fig, ax = plt.subplots(figsize=(14, 6))
    x = list(range(len(problem_order)))

    for method in sorted_methods:
        data = method_results[method]
        problem_to_val = dict(zip(data["problem"], zip(data["avg"], data["std"])))
        if not all(p in problem_to_val for p in problem_order):
            continue
        y_avg = [problem_to_val[p][0] for p in problem_order]
        y_std = [problem_to_val[p][1] for p in problem_order]
        ax.errorbar(x, y_avg, yerr=y_std, label=method, marker='o', linestyle='-', capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(problem_order, rotation=45)
    ax.set_ylabel("Probability (Average)")
    ax.set_title("Performance Across Problems with Error Bars")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    output_png_path = os.path.join(directory, output_png)
    fig.savefig(output_png_path)

    # Generate LaTeX (pgfplots)
    latex_code = r"""\documentclass{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepackage{pgfplotstable}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=14cm, height=7cm,
    xlabel={Problem},
    ylabel={Probability (Average)},
    xtick=data,
    xticklabels={""" + ",".join(problem_order) + r"""},
    xticklabel style={rotate=45, anchor=east},
    legend style={at={(1.05,1)}, anchor=north west},
    grid=both,
    error bars/.cd, y dir=both, y explicit,
    cycle list name=color list
]
"""

    for method in sorted_methods:
        data = method_results[method]
        problem_to_val = dict(zip(data["problem"], zip(data["avg"], data["std"])))
        if not all(p in problem_to_val for p in problem_order):
            continue
        coords = " ".join(
            f"({i},{problem_to_val[p][0]}) +- (0,{problem_to_val[p][1]})"
            for i, p in enumerate(problem_order)
        )
        latex_code += f"\\addplot+[error bars/.cd, y dir=both, y explicit] coordinates {{{coords}}};\n"
        latex_code += f"\\addlegendentry{{{method.replace('_', r'\\_')}}}\n"

    latex_code += r"""\end{axis}
\end{tikzpicture}
\end{document}
"""

    output_tex_path = os.path.join(directory, output_tex)
    with open(output_tex_path, "w") as f:
        f.write(latex_code)

# Example usage (uncomment to run):
# generate_method_graphs("/path/to/directory", num_test_cases=3)
