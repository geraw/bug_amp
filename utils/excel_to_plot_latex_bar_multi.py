"""
multi_excel_bar_chart.py

Creates bar charts from up to 6 Excel files with the structure:
- X-axis: Test case numbers
- For each test case: 4 methods (Ens, BF, SA, GA) grouped together
- For each method: 3-6 problems (Excel files) as bars with different color tones
- First Excel file appears last (rightmost) in each method group

USAGE:
    python multi_excel_bar_chart.py file1.xlsx file2.xlsx file3.xlsx ... [up to 6 files]
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path

def load_excel_data(excel_path):
    """Load data from a single Excel file"""
    try:
        df = pd.read_excel(excel_path, sheet_name=0, header=None)
        
        # Find AVRG and STD rows
        avrg_mask = df.iloc[:, 1] == 'AVRG'
        std_mask = df.iloc[:, 1] == 'STD'
        
        # Extract test case numbers
        test_cases = df.loc[avrg_mask, 0].astype(float).values
        
        # Extract data for each method (columns: Ens=2, BF=11, SA=14, GA=15)
        method_cols = {'Ens': 2, 'BF': 11, 'SA': 14, 'GA': 15}
        
        data = {}
        for method, col in method_cols.items():
            avg_vals = df.loc[avrg_mask, col].astype(float).values
            std_vals = df.loc[std_mask, col].astype(float).values
            data[method] = list(zip(avg_vals, std_vals))
        
        return test_cases, data
    except Exception as e:
        print(f"Error loading {excel_path}: {e}")
        return None, None

def collect_all_data(excel_files):
    """Collect data from all Excel files"""
    all_data = {}  # {test_case: {method: [(avg1, std1), (avg2, std2), ...]}}
    problem_names = []
    
    for i, excel_file in enumerate(excel_files):
        test_cases, file_data = load_excel_data(excel_file)
        
        if test_cases is None:
            continue
            
        # Extract problem name from filename
        problem_name = Path(excel_file).stem.split('_')[1] if '_' in Path(excel_file).stem else f"Problem{i+1}"
        problem_names.append(problem_name)
        
        # Organize data by test case and method
        for j, test_case in enumerate(test_cases):
            if test_case not in all_data:
                all_data[test_case] = {'Ens': [], 'BF': [], 'SA': [], 'GA': []}
            
            for method in ['Ens', 'BF', 'SA', 'GA']:
                if j < len(file_data[method]):
                    all_data[test_case][method].append(file_data[method][j])
    
    return all_data, problem_names

def generate_png(all_data, problem_names, output_path):
    """Generate PNG chart"""
    test_cases = sorted(all_data.keys())
    methods = ['Ens', 'BF', 'SA', 'GA']
    n_problems = len(problem_names)
    
    # Base colors for methods
    method_colors = {
        'Ens': '#1f77b4',  # blue
        'BF': '#ff7f0e',   # orange
        'SA': '#2ca02c',   # green
        'GA': '#d62728'    # red
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Bar configuration
    bar_width = 0.15
    method_spacing = 0.8
    problem_spacing = 0.05
    
    # Calculate positions
    test_case_positions = {}
    for i, test_case in enumerate(test_cases):
        test_case_positions[test_case] = i * (len(methods) * method_spacing + 1.0)
    
    # Plot bars
    for test_case in test_cases:
        base_x = test_case_positions[test_case]
        
        for method_idx, method in enumerate(methods):
            method_base_x = base_x + method_idx * method_spacing
            
            # Get data for this method and test case
            method_data = all_data[test_case][method]
            
            # Reverse order so first file is rightmost
            method_data = method_data[::-1]
            
            for prob_idx, (avg, std) in enumerate(method_data):
                # Calculate position
                x_pos = method_base_x + prob_idx * problem_spacing
                
                # Calculate color intensity (darker for later files, lighter for earlier)
                intensity = 0.3 + (0.7 * prob_idx / max(1, n_problems - 1))
                color = to_rgba(method_colors[method], alpha=intensity)
                
                # Plot bar
                ax.bar(x_pos, avg, bar_width, 
                      color=color, 
                      edgecolor='white', 
                      linewidth=0.5,
                      yerr=std if std > 0 else None,
                      capsize=2)
    
    # Customize plot
    ax.set_xlabel('Test Cases', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Method Performance Across Test Cases and Problems', fontsize=14)
    
    # Set x-axis labels
    x_labels = [str(int(tc)) for tc in test_cases]
    x_label_positions = [test_case_positions[tc] + (len(methods) - 1) * method_spacing / 2 
                        for tc in test_cases]
    ax.set_xticks(x_label_positions)
    ax.set_xticklabels(x_labels)
    
    # Add method labels
    for i, method in enumerate(methods):
        for j, test_case in enumerate(test_cases):
            x_pos = test_case_positions[test_case] + i * method_spacing + (n_problems - 1) * problem_spacing / 2
            ax.text(x_pos, -0.05, method, ha='center', va='top', fontsize=10, 
                   transform=ax.get_xaxis_transform())
    
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # Create legend
    legend_elements = []
    
    # Method legend
    for method in methods:
        legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                           facecolor=method_colors[method], 
                                           alpha=0.7, 
                                           label=method))
    
    # Problem legend (reversed order to match display)
    for i, problem in enumerate(reversed(problem_names)):
        intensity = 0.3 + (0.7 * i / max(1, n_problems - 1))
        legend_elements.append(plt.Rectangle((0,0), 1, 1, 
                                           facecolor='gray', 
                                           alpha=intensity, 
                                           label=problem))
    
    ax.legend(handles=legend_elements, 
              loc='upper left', 
              bbox_to_anchor=(1.02, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex(all_data, problem_names, output_path):
    """Generate LaTeX code"""
    test_cases = sorted(all_data.keys())
    methods = ['Ens', 'BF', 'SA', 'GA']
    n_problems = len(problem_names)
    
    # Create coordinate system
    coords = []
    for test_case in test_cases:
        for method in methods:
            for i in range(n_problems):
                coords.append(f"{int(test_case)}-{method}-{i}")
    
    latex_code = r"""\documentclass[landscape]{article}
\usepackage[margin=0.5cm]{geometry}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}

\begin{document}

\begin{figure}
\centering
\begin{tikzpicture}
\begin{axis}[
    ybar,
    bar width=3pt,
    width=25cm,
    height=15cm,
    symbolic x coords={""" + ", ".join(coords) + r"""},
    xtick={""" + ", ".join([f"{int(tc)}-Ens-0" for tc in test_cases]) + r"""},
    xticklabels={""" + ", ".join([str(int(tc)) for tc in test_cases]) + r"""},
    xlabel={Test Cases},
    ylabel={Probability},
    title={Method Performance Across Test Cases and Problems},
    legend style={at={(1.02,1)}, anchor=north west},
    ymin=0,
    ymax=1,
    grid=major,
    every axis x label/.style={at={(axis description cs:0.5,-0.1)}, anchor=north},
    every axis y label/.style={at={(axis description cs:-0.05,0.5)}, anchor=south},
]

"""
    
    # Method colors for LaTeX
    method_colors = {
        'Ens': 'blue',
        'BF': 'orange', 
        'SA': 'green',
        'GA': 'red'
    }
    
    # Generate plots for each problem (reversed order)
    for prob_idx in range(n_problems):
        actual_prob_idx = n_problems - 1 - prob_idx  # Reverse order
        problem_name = problem_names[actual_prob_idx]
        
        for method in methods:
            plot_coords = []
            
            for test_case in test_cases:
                method_data = all_data[test_case][method]
                if prob_idx < len(method_data):
                    avg, std = method_data[prob_idx]
                    plot_coords.append(f"({int(test_case)}-{method}-{prob_idx}, {avg:.3f})")
            
            if plot_coords:
                # Calculate color intensity
                intensity = 10 + (80 * prob_idx / max(1, n_problems - 1))
                color = method_colors[method]
                
                latex_code += f"""
% {problem_name} - {method}
\\addplot+[
    bar shift={prob_idx * 4}pt,
    draw=white,
    fill=black!{int(intensity)}!{color}
] coordinates {{
    {' '.join(plot_coords)}
}};
"""
    
    # Add legend entries
    latex_code += r"""
% Legend entries
\legend{"""
    
    legend_entries = []
    for method in methods:
        legend_entries.append(f"{method}")
    
    # Add problem entries (reversed to match display order)
    for problem in reversed(problem_names):
        legend_entries.append(f"{problem}")
    
    latex_code += ", ".join(legend_entries)
    latex_code += r"""}

\end{axis}
\end{tikzpicture}
\caption{Method performance across test cases with different problems. Each method group shows bars for different problems, with the first input file appearing rightmost.}
\end{figure}

\end{document}"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

def main():
    if len(sys.argv) < 2:
        print("Usage: python multi_excel_bar_chart.py file1.xlsx file2.xlsx file3.xlsx ... [up to 6 files]")
        sys.exit(1)
    
    # Get Excel files (up to 6)
    excel_files = sys.argv[1:7]  # Limit to 6 files
    
    if len(excel_files) > 6:
        print("Warning: Only first 6 files will be processed")
        excel_files = excel_files[:6]
    
    # Verify files exist
    valid_files = []
    for file in excel_files:
        if os.path.exists(file):
            valid_files.append(file)
        else:
            print(f"Warning: File {file} not found, skipping")
    
    if not valid_files:
        print("Error: No valid Excel files found")
        sys.exit(1)
    
    # Collect data from all files
    print(f"Processing {len(valid_files)} Excel files...")
    all_data, problem_names = collect_all_data(valid_files)
    
    if not all_data:
        print("Error: No data found in Excel files")
        sys.exit(1)
    
    # Generate output filenames
    output_dir = os.path.dirname(valid_files[0]) if valid_files else "."
    png_path = os.path.join(output_dir, "multi_excel_bar_chart.png")
    tex_path = os.path.join(output_dir, "multi_excel_bar_chart.tex")
    
    # Generate outputs
    print("Generating PNG chart...")
    generate_png(all_data, problem_names, png_path)
    
    print("Generating LaTeX file...")
    generate_latex(all_data, problem_names, tex_path)
    
    # Print summary
    test_cases = sorted(all_data.keys())
    print(f"\nChart generated successfully!")
    print(f"Problems processed: {len(problem_names)} ({', '.join(problem_names)})")
    print(f"Test cases: {len(test_cases)} ({', '.join(map(str, map(int, test_cases)))})")
    print(f"Methods: 4 (Ens, BF, SA, GA)")
    print(f"\nFiles created:")
    print(f"  - {png_path}")
    print(f"  - {tex_path}")
    print(f"\nNote: First Excel file appears rightmost in each method group")


if __name__ == "__main__":
    main()