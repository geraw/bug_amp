import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def collect_data(directory, test_cases):
    target_methods = ["Ens", "BF", "SA", "GA"]
    col_indices = [2, 11, 14, 15]
    data = {}
    problem_names = []
    excel_files = [f for f in sorted(os.listdir(directory)) if f.endswith(".xlsx")]
    for filename in excel_files:
        try:
            problem_name = filename.split("_")[1].split(".")[0]
        except IndexError:
            problem_name = filename.split(".")[0]
        problem_names.append(problem_name)
        data[problem_name] = {}
        df = pd.read_excel(os.path.join(directory, filename), sheet_name="Results")
        for method, col in zip(target_methods, col_indices):
            data[problem_name][method] = {}
            for case_size in test_cases:
                match_indices = df.index[df.iloc[:, 0] == case_size].tolist()
                if not match_indices:
                    data[problem_name][method][case_size] = np.nan
                    continue
                i = match_indices[0]
                try:
                    avg_row = df.iloc[i]
                    avg = float(avg_row.iloc[col])
                except:
                    avg = np.nan
                data[problem_name][method][case_size] = avg
    return data, problem_names

def generate_latex(data, problem_names, test_cases, output_dir):
    x_coords = []
    xticks = []
    xticklabels = []
    for i, pname in enumerate(problem_names):
        for m in ["Ens", "BF", "SA", "GA"]:
            x_coords.append(f"{pname}-{m}")
        if i < len(problem_names) - 1:
            x_coords.append(f"spacer{i+1}")
        xticks.append(f"{pname}-BF")
        xticklabels.append(pname)

    latex_code = """\\documentclass[landscape]{article}
\\usepackage[margin=0.5cm]{geometry}
\\usepackage{pgfplots}
\\pgfplotsset{compat=1.18}

\\begin{document}

\\begin{figure}
\\centering
\\begin{tikzpicture}
\\begin{axis}[
    ybar,
    bar width=5pt,
    width=25cm,
    height=15cm,
    symbolic x coords={""" + ", ".join(x_coords) + """},
    xtick={""" + ", ".join(xticks) + """},
    xticklabels={""" + ", ".join(xticklabels) + """},
    x tick label style={rotate=45, anchor=east},
    xlabel={Problems},
    ylabel={Probability},
    title={Method Performance Across Problems and Test Cases},
    legend style={at={(0.5,-0.25)}, anchor=north, legend columns=4},
    ymin=0,
    ymax=1,
    grid=major
]
"""

    method_colors = {
        "Ens": "blue",
        "BF": "red",
        "SA": "green",
        "GA": "orange"
    }
    intensities = ["50", "30", "20", "10"]

    for case_idx, case_size in enumerate(test_cases):
        for method in ["Ens", "BF", "SA", "GA"]:
            coords = []
            for problem in problem_names:
                avg = data[problem][method][case_size]
                if not np.isnan(avg):
                    coords.append(f"({problem}-{method}, {avg:.3f})")
            if coords:
                shift = case_idx * 3
                color = method_colors[method]
                intensity = intensities[case_idx]
                latex_code += f"""
\\addplot+[
    bar shift={shift}pt,
    draw=white,
    fill=black!{intensity}!{color}
] coordinates {{
    {' '.join(coords)}
}};
"""

    legend_labels = ", ".join([f"{m}-{tc}" for tc in test_cases for m in ["Ens", "BF", "SA", "GA"]])
    latex_code += f"""\\legend{{{legend_labels}}}

\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Method performance across problems with different test-case sizes.}}
\\end{{figure}}

\\end{{document}}
"""

    with open(os.path.join(output_dir, "shifted_bars_with_spacers.tex"), "w", encoding="utf-8") as f:
        f.write(latex_code)

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX bar chart with spacers and centered labels.")
    parser.add_argument("directory", type=str, help="Directory with Excel files")
    parser.add_argument("test_cases", type=int, nargs='+', help="Test case sizes (up to 4)")
    args = parser.parse_args()
    test_cases = args.test_cases[:4]
    data, problem_names = collect_data(args.directory, test_cases)
    generate_latex(data, problem_names, test_cases, args.directory)
    print(f"✅ LaTeX file saved in {args.directory}/shifted_bars_with_spacers.tex")

if __name__ == "__main__":
    main()
