import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Accept directory as command-line argument
if len(sys.argv) < 2:
    print("Usage: python avg_curve_by_testcase.py <directory>")
    sys.exit(1)

input_dir = sys.argv[1]

# Methods and associated plotting styles (colors are already set as requested)
methods = {
    "Ans_best": {"label": "Ens", "color": "blue", "marker": "o"},
    "BF_best": {"label": "BF", "color": "red", "marker": "x"},
    "SA_best": {"label": "SA", "color": "green", "marker": "s"},
    "GA_best": {"label": "GA", "color": "orange", "marker": "^"},
}

# Storage for average probabilities by method and test case
aggregated = {method: {} for method in methods}

# Process each Excel file in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".xlsx"):
        file_path = os.path.join(input_dir, filename)
        df = pd.read_excel(file_path)
        for col in df.columns:
            for method in methods:
                # Check if the column name ends with the specific method suffix
                if col.endswith(f"_1k_{method}"):
                    try:
                        # Drop NaN values and convert to list
                        values = df[col].dropna().tolist()
                        for i, val in enumerate(values):
                            # Aggregate values by test case index for each method
                            aggregated[method].setdefault(i, []).append(val)
                    except Exception as e:
                        print(f"Error processing column {col} in {filename}: {e}")

# Calculate mean for each test case and method
x_values_set = set()
for data_dict in aggregated.values():
    x_values_set.update(data_dict.keys())
x_values = sorted(list(x_values_set))

averaged_results = {}
for method, data in aggregated.items():
    averaged_results[method] = []
    for x_idx in x_values:
        if x_idx in data:
            # Calculate average for the current test case index
            avg_prob = sum(data[x_idx]) / len(data[x_idx])
            averaged_results[method].append(((x_idx + 1) * 100, avg_prob))
        else:
            # If no data for this test case index, append (x_value, 0.0)
            averaged_results[method].append(((x_idx + 1) * 100, 0.0))

# --- PNG Plotting (Existing functionality) ---
plt.figure(figsize=(10, 6))
for method, values in averaged_results.items():
    x_vals = [x for x, _ in values]
    y_vals = [y for _, y in values]
    plt.plot(x_vals, y_vals, label=methods[method]["label"], color=methods[method]["color"],
             marker=methods[method]["marker"])

plt.xlabel("Number of test cases")
plt.ylabel("Probability")
plt.title("Aggregated Average Probability per Method Across All Problems")
plt.grid(True)
plt.legend()
plt.ylim(0, 1)

png_output_path = os.path.join(input_dir, "avg_prob_curve_colored.png")
plt.savefig(png_output_path)
plt.close()
print(f"📄 PNG plot saved to: {png_output_path}")


# --- LaTeX Plot Generation Function ---
def generate_latex_plot(averaged_results, methods, caption_text):
    """
    Generates a LaTeX tikzpicture plot within a figure environment.
    The output is designed to be inline (not a full standalone document).
    """
    latex_code = []

    # Figure environment start
    latex_code.append(r"\begin{figure}[htbp]") # [htbp] for here, top, bottom, page
    latex_code.append(r"    \centering")
    latex_code.append(r"    \begin{tikzpicture}")
    latex_code.append(r"        \begin{axis}[")
    latex_code.append(r"            xlabel={Number of test cases},")
    latex_code.append(r"            ylabel={Probability},")
    latex_code.append(r"            title={Aggregated Average Probability per Method Across All Problems},")
    latex_code.append(r"            xmin=0, xmax={auto},") # xmax will be determined automatically
    latex_code.append(r"            ymin=0, ymax=1,")
    latex_code.append(r"            grid=both, % Add grid lines")
    latex_code.append(r"            legend pos=south east,")
    latex_code.append(r"            legend style={draw=none, fill=none},") # No box around legend
    latex_code.append(r"            mark size=2pt, % Adjust marker size")
    latex_code.append(r"            line width=1pt % Adjust line width")
    latex_code.append(r"        ]")

    # Add plots for each method
    for method_key, values in averaged_results.items():
        # Ensure the label is properly escaped for LaTeX
        escaped_label = methods[method_key]["label"].replace("_", "\\_")
        color = methods[method_key]["color"]
        marker = methods[method_key]["marker"]

        # Map matplotlib markers to pgfplots markers
        pgf_marker = ""
        if marker == 'o': pgf_marker = "circle"
        elif marker == 'x': pgf_marker = "cross"
        elif marker == 's': pgf_marker = "square"
        elif marker == '^': pgf_marker = "triangle*" # triangle* is filled triangle
        # Add more mappings if needed

        latex_code.append(f"            \\addplot[color={color}, mark={pgf_marker}] coordinates {{")
        for x, y in values:
            # Clip y values to be within [0, 1] for display purposes in LaTeX
            y_clipped = min(1.0, max(0.0, y))
            latex_code.append(f"                ({x}, {y_clipped:.4f})")
        latex_code.append(r"            };")
        latex_code.append(f"            \\addlegendentry{{{escaped_label}}}")

    latex_code.append(r"        \end{axis}")
    latex_code.append(r"    \end{tikzpicture}")

    # Add caption
    latex_code.append(f"    \\caption{{{caption_text}}}")
    latex_code.append(r"    \label{fig:aggregated_performance_curve}") # Unique label for easy referencing

    # Figure environment end
    latex_code.append(r"\end{figure}")

    return "\n".join(latex_code)

# Define the caption text based on the "graph_explanation" immersive
latex_caption_text = (
    "This graph, \"Aggregated Performance Across Multiple Files (No Error Bars),\" "
    "provides a bird's-eye view of Ensemble (Ens), Brute Force (BF), Simulated Annealing (SA), "
    "and Genetic Algorithm (GA) performance. It aggregates average metrics from multiple Excel files, "
    "showing method convergence and relative efficiency as test cases increase. "
    "The x-axis denotes the number of test cases, and the y-axis represents probability."
)

# Generate and save the LaTeX output
latex_output = generate_latex_plot(averaged_results, methods, latex_caption_text)
latex_output_path = os.path.join(input_dir, "avg_prob_curve_inline.tex")
with open(latex_output_path, "w") as f:
    f.write(latex_output)
print(f"📄 Inline LaTeX plot saved to: {latex_output_path}")

