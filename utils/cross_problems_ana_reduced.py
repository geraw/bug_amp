"""
cross_problems_ana_fixedcols.py

Fixed version: Ens=C (2), BF=L (11), SA=O (14), GA=P (15)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

def generate_graphs(directory, case_sizes):
    """
    Generates grouped bar graphs (PNG and LaTeX) from Excel files.

    Args:
        directory (str): The path to the directory containing the Excel result files.
        case_sizes (list): A list of 4 test case numbers to plot,
                           e.g., [500, 1100, 3900, 5000].
    """
    target_methods = ["Ens", "BF", "SA", "GA"]
    col_indices = [2, 11, 14, 15]  # fixed column positions for each method

    # Gather Excel files, sorting them by problem name
    problem_files = [(f.split("_")[1], os.path.join(directory, f))
                     for f in sorted(os.listdir(directory)) if f.endswith(".xlsx")]

    # Dictionary to store data for all case sizes
    all_case_data = {}

    # First pass: Read all data for all specified case sizes
    for case_size in case_sizes:
        method_avg = {m: [] for m in target_methods}
        method_std = {m: [] for m in target_methods}
        problem_labels_for_case = [] # Labels specific to this case_size's data

        for name, path in problem_files:
            try:
                df = pd.read_excel(path, sheet_name="גיליון1")
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue # Skip to the next file

            # Find the row corresponding to the current case_size
            match_indices = df.index[df.iloc[:, 0] == case_size].tolist()

            if not match_indices:
                # If case_size is not found for this problem, append NaN for averages
                # and 0.0 for std dev to maintain list length for consistent plotting.
                for method in target_methods:
                    method_avg[method].append(np.nan)
                    method_std[method].append(0.0)
                # Add problem name even if no data, to ensure it appears on x-axis if needed
                problem_labels_for_case.append(name)
                continue

            i = match_indices[0] # Get the first matching index
            avg_row = df.iloc[i]
            std_row = df.iloc[i + 1] # Standard deviation is assumed to be in the next row
            problem_labels_for_case.append(name)

            for method, col in zip(target_methods, col_indices):
                try:
                    # Convert values to float, handle potential errors
                    avg = float(avg_row.iloc[col])
                    std = float(std_row.iloc[col])
                except (ValueError, IndexError):
                    # If conversion fails or column is out of bounds, use NaN
                    avg, std = np.nan, 0.0
                method_avg[method].append(avg)
                method_std[method].append(std)

        # Store the collected data for the current case_size
        all_case_data[case_size] = {
            "avg": method_avg,
            "std": method_std,
            "labels": problem_labels_for_case
        }

    # Collect all unique problem labels across all case sizes for consistent X-axis
    all_problem_labels_set = set()
    for case_size in case_sizes:
        if case_size in all_case_data:
            all_problem_labels_set.update(all_case_data[case_size]["labels"])
    all_problem_labels = sorted(list(all_problem_labels_set))

    # --- Plotting PNG (Matplotlib) ---
    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharey=True)
    axes = axes.flatten() # Flatten the 2x2 array of axes for easy iteration

    colors = cm.tab10(np.linspace(0, 1, len(target_methods))) # Colors for the bars
    bar_width = 0.2 # Width of each individual bar
    # X-axis positions for the groups of bars
    x = np.arange(len(all_problem_labels))

    for idx, case_size in enumerate(case_sizes):
        ax = axes[idx] # Get the current subplot axis
        current_data = all_case_data.get(case_size, {"avg": {}, "std": {}, "labels": []})
        current_avg = current_data["avg"]
        current_std = current_data["std"]
        current_problem_labels = current_data["labels"]

        # Map problem labels to their corresponding index in the overall `all_problem_labels`
        problem_label_to_index = {label: i for i, label in enumerate(all_problem_labels)}

        for i, method in enumerate(target_methods):
            # Create full arrays for current method's avg and std, filling with NaN
            # where a problem doesn't have data for this specific case_size.
            plot_avg = np.full(len(all_problem_labels), np.nan)
            plot_std = np.full(len(all_problem_labels), 0.0)

            for p_idx, problem_name in enumerate(current_problem_labels):
                if problem_name in problem_label_to_index:
                    target_idx = problem_label_to_index[problem_name]
                    # Ensure we access data only if it exists for this problem/method
                    if method in current_avg and p_idx < len(current_avg[method]):
                        plot_avg[target_idx] = current_avg[method][p_idx]
                        plot_std[target_idx] = current_std[method][p_idx]

            # Filter out NaN values before plotting to avoid gaps or incorrect bar placements
            valid_indices = ~np.isnan(plot_avg)
            ax.bar(
                x[valid_indices] + i * bar_width, # Offset bars for grouping
                plot_avg[valid_indices],
                yerr=plot_std[valid_indices],
                width=bar_width,
                color=colors[i],
                label=method,
                capsize=5 # Size of the error bar caps
            )

        # Set X-ticks to be in the middle of the grouped bars
        ax.set_xticks(x + (len(target_methods) - 1) * bar_width / 2)
        ax.set_xticklabels(all_problem_labels, rotation=45, ha='right') # Rotate labels for readability
        ax.set_title(f"{case_size} Test Cases")
        ax.set_ylabel("Probability (Average)")
        ax.grid(axis='y', linestyle='--', alpha=0.7) # Add horizontal grid lines

    # Overall title for the figure
    fig.suptitle("Performance of Methods Across Problems for Different Test Case Sizes", fontsize=18, y=0.98)
    # Create a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Methods", bbox_to_anchor=(1.02, 0.95), loc='upper left', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 0.98, 0.93]) # Adjust layout to make space for suptitle and legend
    plt.savefig(os.path.join(directory, f"method_performance_all_cases.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Generate LaTeX Plot (PGFPlots) ---
    # The LaTeX output now only generates the figure environment, not a full document.
    latex_code = r"""
% Define a custom style for error bars on bars in PGFPlots
% This block should ideally be in your main document's preamble if used globally
\pgfplotsset{
    error bars/ybar error/.style={
        /pgfplots/error bars/y dir=both,
        /pgfplots/error bars/y explicit,
        /pgfplots/error bars/error mark options={
            rotate=90, % Ensure error bar caps are horizontal
            xshift=0pt % Adjust as needed
        }
    }
}

\begin{figure*}[!ht] % Use figure* for a wide figure spanning two columns
    \centering
    % Loop through each case size to create a subfigure
"""
    # Define colors for LaTeX bars (can be adjusted)
    latex_bar_fills = ['blue!30', 'red!30', 'green!30', 'orange!30']
    num_methods = len(target_methods)
    bar_width_pt = 4 # PGFPlots bar width in pt (adjust as needed)
    group_width_pt = num_methods * bar_width_pt
    # Calculate the starting offset for the first bar in a group
    start_offset_pt = -(group_width_pt / 2) + (bar_width_pt / 2)

    for idx_case, case_size in enumerate(case_sizes):
        current_data = all_case_data.get(case_size, {"avg": {}, "std": {}, "labels": []})
        current_avg = current_data["avg"]
        current_std = current_data["std"]
        current_problem_labels = current_data["labels"]

        # Start a new subfigure for each case_size
        latex_code += f"""
    \\begin{{subfigure}}[b]{{0.48\\textwidth}} % Adjust width as needed for 2x2 layout
        \\centering
        \\begin{{tikzpicture}}
        \\begin{{axis}}[
            ybar, % Bar plot type
            bar width={bar_width_pt}pt, % Width of individual bars
            width=\\textwidth, height=6cm, % Size of the axis environment
            ymin=0, % Start y-axis from 0
            xlabel={{Problem}},
            ylabel={{Probability (Average)}},
            xtick=data, % Use data points for x-ticks
            xticklabel style={{rotate=45, anchor=east}}, % Rotate x-labels
            enlargelimits=0.15, % Enlarge limits to give space around bars
            symbolic x coords={{{",".join(all_problem_labels)}}}, % Use problem names as symbolic x-coordinates
            % Legend style for the subfigure (will be hidden for all but the first)
            legend style={{at={{(0.5,-0.30)}}, anchor=north,legend columns=-1}},
            grid=both, % Show grid lines
        ]
"""
        # Add legend entries only for the first subplot
        if idx_case == 0:
            for method in target_methods:
                latex_code += f"            \\addlegendentry{{{method}}}\n"

        for i, method in enumerate(target_methods):
            coords = []
            for problem_name in all_problem_labels:
                # Find the actual data for the current problem and method
                try:
                    # Check if this problem has data for the current case_size
                    data_idx = current_problem_labels.index(problem_name)
                    avg = current_avg[method][data_idx]
                    std = current_std[method][data_idx]
                    if not np.isnan(avg): # Only add coordinates if average is not NaN
                        # Format for PGFPlots: (symbolic_x_coord, value) +- (0, error)
                        coords.append(f"({problem_name},{avg}) +-(0,{std})")
                except ValueError:
                    # Problem not found for this case_size's data, skip plotting bar for it
                    continue

            # Plot bars with error bars, using xshift to group them
            latex_code += (
                f"            \\addplot[draw=black, fill={latex_bar_fills[i]}, error bars/ybar error, "
                f"xshift={start_offset_pt + i * bar_width_pt}pt] coordinates {{ {' '.join(coords)} }};\n"
            )

        latex_code += f"""
        \\end{{axis}}
        \\end{{tikzpicture}}
        \\caption{{{case_size} Test Cases}} % Caption for the individual subfigure
        \\label{{fig:case_{case_size}}}
    \\end{{subfigure}}
"""
        # Add horizontal space between subfigures in the same row
        if idx_case % 2 == 0 and idx_case < len(case_sizes) - 1:
            latex_code += r"    \hfill"
        # Add vertical space between rows of subfigures
        if idx_case == 1: # After the second subplot (first row)
            latex_code += r"""
    \par\vspace{\baselineskip} % Add some vertical space between rows
"""

    latex_code += r"""
    \caption{Performance of different optimization methods across various problem instances for different test case sizes. Each subplot shows the average probability of success with standard deviation for Ensemble (Ens), Brute Force (BF), Simulated Annealing (SA), and Genetic Algorithm (GA) methods.}
    \label{fig:all_method_performance}
\end{figure*}
"""

    # Save the LaTeX code to a .tex file
    with open(os.path.join(directory, f"method_performance_all_cases.tex"), "w", encoding="utf-8") as f:
        f.write(latex_code)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grouped bar graphs (PNG and LaTeX) from Excel files."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Directory with Excel files (e.g., /path/to/your/data)"
    )
    parser.add_argument(
        "case_sizes",
        type=int,
        nargs=4, # Expect exactly 4 test case numbers
        help="Four space-separated test case numbers (e.g., 500 1100 3900 5000)"
    )
    args = parser.parse_args()

    # Sort case sizes to ensure consistent plotting order across runs
    args.case_sizes.sort()

    generate_graphs(args.directory, args.case_sizes)
