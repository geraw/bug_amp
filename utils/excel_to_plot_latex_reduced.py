
"""
excel_to_plot_latex.py  (updated)

Changes:
1. Removes "Classifier_best" and "MLP_best" series – only Ensemble (ENS), BF, SA, GA remain.
2. Renames "Ans_best" → "Ens_best" (Ensemble).
3. Each series line now has both a unique color (auto) and a unique marker symbol (o, x, s, ^).
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_data(excel_path):
    """Load x‑axis, y‑values, and error bars from the Excel sheet."""
    xls = pd.ExcelFile(excel_path)
    df_raw = xls.parse(xls.sheet_names[0], header=None)

    # Identify rows
    avrg_rows = df_raw[1] == 'AVRG'
    std_rows  = df_raw[1] == 'STD'

    x = df_raw.loc[avrg_rows, 0].astype(float).to_numpy()

    # Columns: (value_col, std_col)
    series_indices = [
        (2, 2),   # Ens_best   (formerly Ans_best)
        (11, 11), # BF_best
        (14, 14), # SA_best
        (15, 15)  # GA_best
    ]
    series_labels  = ["Ens_best", "BF_best", "SA_best", "GA_best"]

    ys, yerrs = [], []
    for val_col, std_col in series_indices:
        ys.append(df_raw.loc[avrg_rows, val_col].astype(float).to_numpy())
        yerrs.append(df_raw.loc[std_rows, std_col].astype(float).to_numpy())

    return x, ys, yerrs, series_labels


def plot_graph(x, ys, yerrs, series_labels, png_path):
    """Generate a PNG with distinct markers for every series."""
    markers = ['o', 'x', 's', '^']  # circle, x, square, triangle
    plt.figure(figsize=(10, 6))

    for i, label in enumerate(series_labels):
        marker = markers[i % len(markers)]
        # Clip values to [0,1]
        yi = [min(1.0, max(0.0, v)) for v in ys[i]]

        # Clip error bars to stay within [0,1]
        ei = []
        for y_val, err in zip(yi, yerrs[i]):
            lower = max(0.0, y_val - err)
            upper = min(1.0, y_val + err)
            ei.append(max(upper - y_val, y_val - lower))

        plt.errorbar(
            x, yi, yerr=ei, label=label,
            marker=marker, linestyle='-', capsize=3
        )

    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()


def latex_escape(text):
    specials = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    for char, escape in specials.items():
        text = text.replace(char, escape)
    return text


def generate_latex(x, ys, yerrs, series_labels, tex_path):
    """Generate a standalone LaTeX pgfplots document with distinct markers."""
    latex_markers = ['*', 'x', 'square*', 'triangle*']  # markers for pgfplots

    header = r"""\documentclass{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
    width=14cm,
    height=9cm,
    ymin=0, ymax=1,
    xlabel={Number of test cases},
    ylabel={Probability},
    title={Comparison of Series with Error Bars},
    grid=major,
    legend style={at={(1.05,1)}, anchor=north west},
    error bars/y dir=both,
    error bars/y explicit
]
"""

    body = ""
    for i, label in enumerate(series_labels):
        y  = ys[i]
        yerr = yerrs[i]
        escaped_label = latex_escape(label)
        pgf_marker = latex_markers[i % len(latex_markers)]

        body += f"% Plot for {escaped_label}\n"
        body += f"\\addplot+[mark={pgf_marker}, error bars/.cd, y dir=both, y explicit] coordinates {{\n"
        for xi, yi, ei in zip(x, y, yerr):
            yi_clipped = min(1.0, max(0.0, yi))
            lower = max(0.0, yi_clipped - ei)
            upper = min(1.0, yi_clipped + ei)
            ei_clip = max(upper - yi_clipped, yi_clipped - lower)
            body += f"({xi}, {yi_clipped}) +- (0, {ei_clip})\n"
        body += "};\n"
        body += f"\\addlegendentry{{{escaped_label}}}\n\n"

    footer = r"""\end{axis}
\end{tikzpicture}
\end{document}
"""

    with open(tex_path, "w") as f:
        f.write(header + body + footer)


def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex_reduced.py path/to/result_table2.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]

    base_name = os.path.splitext(excel_path)[0]
    png_path = base_name + ".png"
    tex_path = base_name + ".tex"

    x, ys, yerrs, labels = load_data(excel_path)
    plot_graph(x, ys, yerrs, labels, png_path)
    generate_latex(x, ys, yerrs, labels, tex_path)

    print("Files generated:")
    print(" -", png_path)
    print(" -", tex_path)


if __name__ == "__main__":
    main()
