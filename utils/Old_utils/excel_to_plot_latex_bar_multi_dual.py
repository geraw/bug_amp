# Modified to combine MLP and BF for 5th and 10th best on one graph

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

METHODS = ['MLP', 'BF']
PERCENTILES = ['5th', '10th'] # New: Define percentiles
COL_PATTERNS = { # Updated: Combine patterns for 5th and 10th
    'MLP_5th': '_MLP_5th', 'BF_5th': '_BF_5th',
    'MLP_10th': '_MLP_10th', 'BF_10th': '_BF_10th'
}
# Updated base colours for differentiation
BASE_COLOURS = {
    'MLP_5th': '#6a3d9a',    # Purple for 5th MLP
    'BF_5th': '#ff7f0e',     # Orange for 5th BF
    'MLP_10th': '#33a02c',   # Green for 10th MLP
    'BF_10th': '#e31a1c'     # Red for 10th BF
}
BAR_W = 0.2
PROB_SHIFT_PX = 0.02
PROB_SHIFT_PT = 0.8
Y_MAX = 1.05

def find_columns(header_row, patterns):
    # Updated to find columns based on combined patterns
    return {p: [i for i, h in enumerate(header_row) if isinstance(h, str) and pat in h][0]
            for p, pat in patterns.items()}

def load_data(file_path, patterns):
    df = pd.read_excel(file_path, header=None)
    avg_mask = df.iloc[:, 1] == 'AVRG'
    tcs = df.loc[avg_mask, 0].astype(int).values
    header = df.iloc[0]
    col_idx = find_columns(header, patterns)
    # Load values for all combined patterns
    values = {p: df.loc[avg_mask, col_idx[p]].astype(float).values for p in patterns.keys()}
    return tcs, values

def aggregate(files, patterns):
    problems, all_data = [], {}
    for idx, f in enumerate(files):
        tcs, vals = load_data(f, patterns)
        pname = Path(f).stem.split('_')[1] if '_' in Path(f).stem else f"Prob{idx+1}"
        problems.append(pname)
        for j, t in enumerate(tcs):
            all_data.setdefault(t, {p: [] for p in patterns.keys()}) # Initialize for all patterns
            for p in patterns.keys():
                all_data[t][p].append(vals[p][j])
    return sorted(all_data), all_data, problems

def plot_png(filename, tcs, data, problems, title):
    n_prob = len(problems)
    fig, ax = plt.subplots(figsize=(22, 12))  # Increased figure size
    
    # Combined modes for plotting
    plot_modes = [f"{m}_{p}" for p in PERCENTILES for m in METHODS]
    n_modes = len(plot_modes)

    x0 = np.arange(len(tcs)) * (n_modes + 1) # Adjust x0 for all modes

    legend_elements = []
    legend_labels = []

    for mode_idx, mode in enumerate(plot_modes):
        for ti, t in enumerate(tcs):
            base_x = x0[ti] + mode_idx * BAR_W # Shift by BAR_W for each mode
            for pi, p in enumerate(problems):
                shift = (pi - (n_prob - 1) / 2) * PROB_SHIFT_PX
                alpha = 0.3 + 0.7 * (pi / max(1, n_prob - 1))
                color = to_rgba(BASE_COLOURS[mode], alpha=alpha)
                
                ax.bar(base_x + shift, data[t][mode][pi], BAR_W, color=color, edgecolor='none')

                if ti == 0:
                    legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color))
                    legend_labels.append(f"{mode}-{p}")

    # Center the x-axis labels
    ax.set_xticks(x0 + (n_modes * BAR_W) / 2 - BAR_W/2)
    ax.set_xticklabels([str(t) for t in tcs])
    ax.set_xlabel("Test Cases")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, Y_MAX)
    ax.set_title(title)
    ax.grid(axis='y', alpha=.3)

    ax.margins(x=0.05)

    ax.legend(legend_elements, legend_labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=min(4, len(legend_labels)), frameon=True)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼  PNG  → {filename}")

def generate_tex(file_path, tcs, data, problems):
    n_prob = len(problems)
    # Adjusted col_map for LaTeX colors
    tex_col_map = {
        'MLP_5th': 'blue!80',    # Darker blue for 5th MLP
        'BF_5th': 'orange!80',     # Darker orange for 5th BF
        'MLP_10th': 'green!80',   # Darker green for 10th MLP
        'BF_10th': 'red!80'     # Darker red for 10th BF
    }

    # Combined modes for plotting
    plot_modes = [f"{m}_{p}" for p in PERCENTILES for m in METHODS]
    
    sym_coords = [f"{t}-{mode}" for t in tcs for mode in plot_modes]
    tick_coords = []
    for t in tcs:
        # Center the tick label for each test case
        tick_coords.append(f"{t}-{plot_modes[len(plot_modes)//2]}")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n\n% --- Combined Graph ---\n")
        f.write("\n".join([
            r"\begin{figure}[H]",
            r"\centering",
            r"\rotatebox{270}{",
            r"\begin{tikzpicture}",
            r"\begin{axis}[",
            r"ybar,",
            r"bar width=4pt,",
            r"width=24cm,", # Increased width further for combined graph
            r"height=12cm,", # Increased height further
            f"symbolic x coords={{" + ",".join(sym_coords) + r"}},",
            f"xtick={{" + ",".join(tick_coords) + r"}},",
            f"xticklabels={{" + ",".join(str(t) for t in tcs) + r"}},",
            r"x tick label style={rotate=45,anchor=east},",
            r"xlabel={Test Cases},",
            r"ylabel={Probability},",
            r"legend style={at={(0.5,-0.25)},anchor=north,legend columns=4},", # More columns for legend
            r"ymin=0,ymax=" + str(Y_MAX) + r",",
            r"enlargelimits=0.05,",
            r"scale only axis=true,",
            r"grid=major]"
        ]))

        legend_entries = []
        for pi, p in enumerate(problems):
            shift_pt = (pi - (n_prob-1)/2)*PROB_SHIFT_PT
            intensity = int(90 - 70*pi/max(1,n_prob-1)) # Intensity calculation for shading

            for mode_idx, mode in enumerate(plot_modes):
                # Calculate bar shift for each mode
                mode_shift_pt = (mode_idx - (len(plot_modes) - 1) / 2) * (BAR_W * 10) # Adjust based on BAR_W for visual separation
                
                pts = " ".join(f"({t}-{mode},{data[t][mode][pi]:.3f})" for t in tcs)
                f.write(
                    rf"""
\addplot+[
  bar shift={shift_pt + mode_shift_pt:.2f}pt,
  draw=none,
  fill={tex_col_map[mode]}!{intensity}!white
] coordinates {{
  {pts}
}};"""
                )
                legend_entries.append(f"{mode}-{p}")

        legend_items = ", ".join(legend_entries)
        f.write(rf"""
\legend{{{legend_items}}}
\end{{axis}}
\end{{tikzpicture}}
}}
\caption{{Combined 5th and 10th Best Performance Across Problems}}
\label{{fig:combined_performance}}
\end{{figure}}
""")
    print(f"📄 LaTeX → {file_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py file1.xlsx file2.xlsx ...")
        sys.exit(1)
    files = sys.argv[1:]
    out_dir = Path(files[0]).parent

    # Aggregate all data (5th and 10th for MLP and BF)
    tcs, all_data, probs = aggregate(files, COL_PATTERNS)

    # Plot the combined PNG graph
    plot_png(out_dir / "combined_performance.png", tcs, all_data, probs, "Combined 5th and 10th Best: MLP vs BF")
    
    # Generate the combined LaTeX graph
    generate_tex(out_dir / "combined_performance.tex", tcs, all_data, probs)

if __name__ == "__main__":
    main()