
"""
Script: excel_to_plot_latex.py
Purpose: Read an Excel table (structured as described), produce:
         1. A PNG graph with error bars.
         2. A standalone LaTeX file (pgfplots) that reproduces the graph.
Usage:
    python excel_to_plot_latex.py path/to/result_table2.xlsx
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def load_data(excel_path):
    xls = pd.ExcelFile(excel_path)
    df_raw = xls.parse(xls.sheet_names[0], header=None)
    avrg_rows = df_raw[1] == 'AVRG'
    std_rows  = df_raw[1] == 'STD'

    x = df_raw.loc[avrg_rows, 0].astype(float).to_numpy()

    series_indices = [(2, 2), (5, 5), (8, 8), (11, 11), (14, 14), (15, 15)]
    series_labels  = ["Ans_best", "Classifier_best", "MLP_best",
                      "BF_best", "SA_best", "GA_best"]

    ys, yerrs = [], []
    for val_col, std_col in series_indices:
        ys.append(df_raw.loc[avrg_rows, val_col].astype(float).to_numpy())
        yerrs.append(df_raw.loc[std_rows, std_col].astype(float).to_numpy())

    return x, ys, yerrs, series_labels


def plot_graph(x, ys, yerrs, series_labels, png_path):
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(series_labels):
        plt.errorbar(x, ys[i], yerr=yerrs[i], label=label, fmt='-o', capsize=3)
        yi_clipped = [min(1.0, max(0.0, v)) for v in ys[i]]
        ei_clipped = []
        for y_val, err in zip(ys[i], yerrs[i]):
            lower = max(0.0, y_val - err)
            upper = min(1.0, y_val + err)
            ei_clipped.append(upper - y_val if upper - y_val >= y_val - lower else y_val - lower)
        plt.errorbar(x, yi_clipped, yerr=ei_clipped, label=label, fmt='-o', capsize=3)


    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

def latex_escape(text):
    specials = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\\\": r"\textbackslash{}",
    }
    for char, escape in specials.items():
        text = text.replace(char, escape)
    return text


def generate_latex(x, ys, yerrs, series_labels, tex_path):
    header = r"""\documentclass{standalone}
    \usepackage{pgfplots}
    \pgfplotsset{compat=1.18}
    \begin{document}
    \begin{tikzpicture}
    \begin{axis}[
        width=14cm,
        height=9cm,
        ymin=0, ymax=1,
        xlabel={number of test cases},
        ylabel={Probability},
        title={Comparison of Series with Error Bars},
        grid=major,
        legend style={at={(1.05,1)}, anchor=north west},
        error bars/y dir=both,
        error bars/y explicit,
        every axis plot/.append style={mark=*},
        cycle list name=color list
    ]
    """

    body = ""
    for i, label in enumerate(series_labels):
        y = ys[i]
        yerr = yerrs[i]
        escaped_label = latex_escape(label)  # <--- Escape ONCE here

        body += f"% Plot for {escaped_label}\n"  # Optional comment
        body += "\\addplot+[error bars/.cd, y dir=both, y explicit] coordinates {\n"
        for xi, yi, ei in zip(x, y, yerr):
            yi = min(1.0, max(0.0, yi))
            lower = max(0.0, yi - ei)
            upper = min(1.0, yi + ei)
            ei_clipped = upper - yi if upper - yi >= yi - lower else yi - lower

            ei_clipped = min(ei, 1.0 - yi) if yi + ei > 1 else ei  # limit error bar to stay within [0,1]
            body += f"({xi}, {yi}) +- (0, {ei_clipped})\n"
        body += "};\n"
        body += f"\\addlegendentry{{{escaped_label}}}\n\n"

    
    
    
    # for i, label in enumerate(series_labels):
    #     body += "\\addplot+[error bars/.cd, y dir=both, y explicit] coordinates {\n"
    #     for xi, yi, ei in zip(x, ys[i], yerrs[i]):
    #         # body += f"({{xi}}, {{yi}}) +- (0, {{ei}})\n".format(xi=xi, yi=yi, ei=ei)
    #         yi = min(1.0, max(0.0, yi))
    #         lower = max(0.0, yi - ei)
    #         upper = min(1.0, yi + ei)
    #         ei_clipped = upper - yi if upper - yi >= yi - lower else yi - lower
    #         body += f"({xi}, {yi}) +- (0, {ei_clipped})\n"

    #     body += "};\n"          
    #     # body += f"\\addlegendentry{{{{{label}}}}}\n"
    #     escaped_label = latex_escape(label)
    #     body += f"\\addlegendentry{{{escaped_label}}}\n"


    footer = r"""\end{axis}
    \end{tikzpicture}
    \end{document}
    """

    with open(tex_path, "w") as f:
        f.write(header + body + footer)


def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex.py path/to/result_table2.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]

    # Remove the file extension from excel_path
    base_name = os.path.splitext(excel_path)[0]
    # Create paths with new extensions
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
