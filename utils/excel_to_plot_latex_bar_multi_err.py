#!/usr/bin/env python3
"""
excel_to_plot_latex_bar_multi.py  –  OVERLAID BAR VERSION
---------------------------------------------------------
* Reads up to 8 Excel files (each = one problem) of identical structure.
* Produces
    multi_excel_bar_chart.png
    multi_excel_bar_chart.tex   (PGFPlots, MDPI-friendly)
* Bars for different *problems* are virtually at the same x-position;
  each problem is nudged by ±Δ to appear "in front of" the others.

USAGE
-----
python excel_to_plot_latex_bar_multi.py file1.xlsx file2.xlsx …

The order of Excel files defines the colour shade:
    first = lightest, last = darkest.
"""

from pathlib import Path
import sys, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
METHODS        = ['Ens', 'BF', 'SA', 'GA']
COL_IDX        = {'Ens':2, 'BF':11, 'SA':14, 'GA':15}
BASE_COLOURS   = {'Ens':"#73b41f",'BF':'#ff7f0e','SA':'#2ca02c','GA':'#d62728'}
BAR_W          = 0.18
PROB_SHIFT_PX  = 0.02    # ← tiny lateral offset per problem (PNG)
PROB_SHIFT_PT  = 0.8     # ← same offset in TikZ (≈0.02*40pt)
Y_MAX          = 1.05

# ---------------------------------------------------------------------------
def load_workbook(path: Path):
    """Return (test_case_array, {method:[(avg,std), …]})"""
    df = pd.read_excel(path, header=None, sheet_name=0)
    avg_mask = df.iloc[:,1] == "AVRG"
    std_mask = df.iloc[:,1] == "STD"
    tcs      = df.loc[avg_mask, 0].astype(int).values
    data     = {m:[] for m in METHODS}
    for m,c in COL_IDX.items():
        data[m] = list(zip(df.loc[avg_mask,c].astype(float).values,
                           df.loc[std_mask,c].astype(float).values))
    return tcs, data

def aggregate(files):
    """files -> (sorted_test_cases, data_by_tc, problem_names)"""
    problems, all_data = [], {}
    for idx,f in enumerate(files):
        tc, d = load_workbook(f)
        pname  = Path(f).stem.split('_')[1] if '_' in Path(f).stem else f"Prob{idx+1}"
        problems.append(pname)
        for j,t in enumerate(tc):
            all_data.setdefault(t,{m:[] for m in METHODS})
            for m in METHODS:
                all_data[t][m].append(d[m][j])
    return sorted(all_data), all_data, problems

# ---------------------------------------------------------------------------
# PNG
# ---------------------------------------------------------------------------
def plot_png(tcs, data, probs, out_png):
    n_prob = len(probs)
    fig,ax = plt.subplots(figsize=(18,10))  # Increased figure size for better margins
    base_x = np.arange(len(tcs))* (len(METHODS)+1)

    # Create legend elements
    legend_elements = []
    legend_labels = []

    for mi,m in enumerate(METHODS):
        for ti,t in enumerate(tcs):
            x0 = base_x[ti] + mi
            for pi,p in enumerate(probs):
                shift = (pi - (n_prob-1)/2)*PROB_SHIFT_PX
                avg, std = data[t][m][pi]
                # Changed: first file is darkest (alpha=0.3), last file is lightest (alpha=1.0)
                alpha = 0.3 + 0.7*(pi/(max(1,n_prob-1)))
                color = to_rgba(BASE_COLOURS[m], alpha=alpha)
                ax.bar(x0+shift, avg, BAR_W, color=color,
                       yerr=std, capsize=3,
                       error_kw={'ecolor': 'black', 'alpha': 0.7, 'linewidth': 1},
                       edgecolor='black')
                # Add to legend (only once per method-problem combination)
                if ti == 0:  # Only add to legend for first test case
                    legend_elements.append(plt.Rectangle((0,0),1,1,fc=color))
                    legend_labels.append(f"{m}-{p}")

    # Center the x-axis labels on the middle of the 4 methods
    ax.set_xticks(base_x + (len(METHODS)-1)/2)
    ax.set_xticklabels([str(t) for t in tcs])
    ax.set_xlabel("Test Cases")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, Y_MAX)
    ax.set_title("Over-laid Method Performance Across Problems")
    ax.grid(axis='y', alpha=.3)
    
    # Add margins
    ax.margins(x=0.05)  # 5% margin on left and right
    
    # Place legend below the graph
    ax.legend(legend_elements, legend_labels, 
              loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=min(4, len(legend_labels)), frameon=True)
    
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼  PNG  → {out_png}")

# ---------------------------------------------------------------------------
# PGFPlots / TikZ
# ---------------------------------------------------------------------------
def plot_tex(tcs, data, probs, out_tex):
    n_prob = len(probs)
    sym_coords = [f"{t}-{m}" for t in tcs for m in METHODS]
    
    # Create custom tick positions that will center labels between all 4 methods
    # We'll use a trick: place ticks at positions that are between SA and GA (middle of the 4)
    tick_coords = []
    for t in tcs:
        tick_coords.append(f"{t}-SA")  # SA is roughly in the middle of Ens,BF,SA,GA

    tex  = r"""\begin{figure}[H]
    \centering
    \begin{tikzpicture}
    \begin{axis}[
  ybar,
  bar width=4pt,
  width=20cm,
  height=10cm,
  symbolic x coords={""" + ",".join(sym_coords) + r"""},
  xtick={""" + ",".join(tick_coords) + r"""},
  xticklabels={""" + ",".join(str(t) for t in tcs) + r"""},
  x tick label style={rotate=45,anchor=east},
  xlabel={Test Cases},
  ylabel={Probability},
  legend style={at={(0.5,-0.25)},anchor=north,legend columns=4},
  ymin=0,ymax=""" + str(Y_MAX) + r""",
  enlargelimits=0.05,
  scale only axis=true, 
  grid=major]
"""

    # addplots
    col_map = {'Ens':'blue','BF':'orange','SA':'green','GA':'red'}
    legend_entries = []
    
    for pi,p in enumerate(probs):
        shift_pt = (pi - (n_prob-1)/2)*PROB_SHIFT_PT
        # Changed: first file is darkest (90), last file is lightest (20)
        intensity = int(90 - 70*pi/max(1,n_prob-1))
        for m in METHODS:
            # Create coordinates with error values
            coords_with_errors = []
            for t in tcs:
                avg_val = data[t][m][pi][0]
                std_val = data[t][m][pi][1]
                coords_with_errors.append(f"({t}-{m},{avg_val:.3f}) +- (0,{std_val:.3f})")
            
            coords_str = " ".join(coords_with_errors)
            tex += rf"""
\addplot+[
  bar shift={shift_pt:.2f}pt,
  draw=none,
  fill=black!{intensity}!{col_map[m]},
  error bars/.cd,
  y dir=both,
  y explicit
] coordinates {{
  {coords_str}
}};"""
            legend_entries.append(f"{m}-{p}")

    # Create legend with method-problem combinations
    legend_items = ", ".join(legend_entries)
    tex += rf"""
\legend{{{legend_items}}}
\end{{axis}}
\end{{tikzpicture}}
}}
\caption{{Overlaid Bar Chart of Method Performance Across Problems}}    
\label{{fig:multi_excel_bar_chart}}
\end{{figure}}
"""
    out_tex.write_text(tex, encoding='utf8')
    print(f"📄 LaTeX → {out_tex}")

# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex_bar_multi.py file1.xlsx file2.xlsx …")
        sys.exit(1)

    excel_files = [f for f in sys.argv[1:] if Path(f).suffix == '.xlsx']
    if not excel_files:
        print("No .xlsx files recognised.")
        sys.exit(1)

    tcs, data, probs = aggregate(excel_files)
    out_dir = Path(excel_files[0]).parent
    plot_png(tcs, data, probs, out_dir / "multi_excel_bar_chart_err.png")
    plot_tex(tcs, data, probs, out_dir / "multi_excel_bar_chart_err.tex")
    print("✅ Done.")
    print(f"Problems processed (first=darkest, last=lightest): {probs}")

if __name__ == "__main__":
    main()