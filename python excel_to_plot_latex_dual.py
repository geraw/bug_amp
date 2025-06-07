import sys
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(excel_path, indices, labels):
    xls = pd.ExcelFile(excel_path)
    df_raw = xls.parse(xls.sheet_names[0], header=None)
    avrg_rows = df_raw[1] == 'AVRG'
    std_rows  = df_raw[1] == 'STD'

    x = df_raw.loc[avrg_rows, 0].astype(float).to_numpy()

    ys, yerrs = [], []
    for val_col, std_col in indices:
        ys.append(df_raw.loc[avrg_rows, val_col].astype(float).to_numpy())
        yerrs.append(df_raw.loc[std_rows, std_col].astype(float).to_numpy())

    return x, ys, yerrs, labels


def plot_graph(x, ys, yerrs, series_labels, png_path, title):
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(series_labels):
        plt.errorbar(x, ys[i], yerr=yerrs[i], label=label, fmt='-o', capsize=3)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel("number of test cases")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

def latex_escape(text):
    return text.replace("_", r"\_")


def generate_latex(x, ys, yerrs, series_labels, tex_path, title):
    header = rf"""\documentclass{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{axis}}[
    width=14cm,
    height=9cm,
    ymin=0, ymax=1,
    xlabel={{number of test cases}},
    ylabel={{Probability}},
    title={{{title}}},
    grid=major,
    legend style={{at={{(1.05,1)}}, anchor=north west}},
    error bars/y dir=both,
    error bars/y explicit,
    every axis plot/.append style={{mark=*}},
    cycle list name=color list
]
"""

    body = ""
    for i, label in enumerate(series_labels):
        y = ys[i]
        yerr = yerrs[i]
        escaped_label = latex_escape(label)
        body += f"% Plot for {escaped_label}\n"
        body += "\\addplot+[error bars/.cd, y dir=both, y explicit] coordinates {\n"
        for xi, yi, ei in zip(x, y, yerr):
            yi = min(1.0, max(0.0, yi))
            lower = max(0.0, yi - ei)
            upper = min(1.0, yi + ei)
            ei_clipped = upper - yi if upper - yi >= yi - lower else yi - lower
            ei_clipped = min(ei, 1.0 - yi) if yi + ei > 1 else ei
            body += f"({xi}, {yi}) +- (0, {ei_clipped})\n"
        body += "};\n"
        body += f"\\addlegendentry{{{escaped_label}}}\n\n"

    footer = r"""\end{axis}
\end{tikzpicture}
\end{document}
"""

    with open(tex_path, "w") as f:
        f.write(header + body + footer)

def process_dataset(excel_path, indices, labels, suffix, title):
    base_name = os.path.splitext(excel_path)[0] + f"_{suffix}"
    png_path = base_name + ".png"
    tex_path = base_name + ".tex"

    x, ys, yerrs, labels = load_data(excel_path, indices, labels)
    plot_graph(x, ys, yerrs, labels, png_path, title)
    generate_latex(x, ys, yerrs, labels, tex_path, title)
    print(f"Generated: {png_path}, {tex_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex_dual.py path/to/result_table2.xlsx")
        sys.exit(1)

    excel_path = sys.argv[1]

    fifth_best_indices = [(3, 3), (6, 6), (9, 9), (12, 12)]
    tenth_best_indices = [(4, 4), (7, 7), (10, 10), (13, 13)]
    series_labels = ["Ans_5^th", "Classifier_5^th", "MLP_5^th", "BF_5^th"]

    process_dataset(
        excel_path,
        fifth_best_indices,
        series_labels,
        suffix="5th_best_cases",
        title="5th Best Test Cases"
    )
    series_labels_10 = ["Ans_10^th", "Classifier_10^th", "MLP_10^th", "BF_10^th"]
    process_dataset(
        excel_path,
        tenth_best_indices,
        series_labels_10,
        suffix="10^th_best_vectors",
        title="10^th Best Test Vectors"
    )

if __name__ == "__main__":
    main()
