#!/usr/bin/env python3
"""
excel_to_plot_latex_bar_multi.py  (fixed)

* One PNG + one LaTeX PGFPlots file
* For every <test-case, method> group all problem-bars are shown
  “in front of each other” with a tiny horizontal shift.
* Colour tone reflects the order in which Excel files were given
  (first = lightest, last = darkest).

USAGE:
    python excel_to_plot_latex_bar_multi.py file1.xlsx file2.xlsx …  (max 6)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from pathlib import Path


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
METHODS        = ['Ens', 'BF', 'SA', 'GA']
METHOD_COLS    = {'Ens': 2, 'BF': 11, 'SA': 14, 'GA': 15}
METHOD_COLORS  = {'Ens': '#1f77b4', 'BF': '#ff7f0e',
                  'SA': '#2ca02c', 'GA': '#d62728'}
BAR_WIDTH      = 0.15
BAR_SHIFT_PX   = 0.05     # ⬅ very small x-shift (PNG)
BAR_SHIFT_PT   = 1.2      # ⬅ very small bar shift in TikZ (≈ BAR_SHIFT_PX*25pt)
Y_MAX          = 1.05


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_excel(path):
    """Return (test_case_array, {method: [(avg,std),…]}) for one workbook."""
    df = pd.read_excel(path, sheet_name=0, header=None)
    avrg_mask = df.iloc[:, 1] == 'AVRG'
    std_mask  = df.iloc[:, 1] == 'STD'

    test_cases = df.loc[avrg_mask, 0].astype(float).values
    data = {m: [] for m in METHODS}
    for m, col in METHOD_COLS.items():
        avg_vals = df.loc[avrg_mask, col].astype(float).values
        std_vals = df.loc[std_mask,  col].astype(float).values
        data[m] = list(zip(avg_vals, std_vals))
    return test_cases, data


def collect_all(files):
    """Aggregate per-file data into {tc: {method:[(avg,std),…]}}"""
    all_data, names = {}, []
    for idx, f in enumerate(files):
        tc, d = load_excel(f)
        pname = Path(f).stem.split('_')[1] if '_' in Path(f).stem else f"Problem{idx+1}"
        names.append(pname)
        for j, t in enumerate(tc):
            all_data.setdefault(t, {m: [] for m in METHODS})
            for m in METHODS:
                all_data[t][m].append(d[m][j])
    return all_data, names


# --------------------------------------------------------------------------
# PNG
# --------------------------------------------------------------------------
def plot_png(all_data, pnames, out_png):
    test_cases = sorted(all_data)
    n_probs    = len(pnames)

    fig, ax = plt.subplots(figsize=(16, 8))
    base_x = np.arange(len(test_cases)) * (len(METHODS) + 1)

    for mi, method in enumerate(METHODS):
        for pi, pname in enumerate(pnames):
            for ti, tc in enumerate(test_cases):
                avg, std = all_data[tc][method][pi]
                x = base_x[ti] + mi + BAR_SHIFT_PX*pi
                alpha = 0.3 + 0.7 * (pi / max(1, n_probs-1))
                color = to_rgba(METHOD_COLORS[method], alpha=alpha)
                ax.bar(x, avg, BAR_WIDTH, color=color,
                       yerr=std if std > 0 else None, capsize=2,
                       edgecolor='white', linewidth=0.4)

    # X-ticks = centred under method group
    ax.set_xticks(base_x + (len(METHODS)-1)/2)
    ax.set_xticklabels([str(int(t)) for t in test_cases])
    ax.set_xlabel('Test Cases')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, Y_MAX)
    ax.set_title('Method Performance Across Test Cases and Problems')
    ax.grid(axis='y', alpha=0.3)

    # Legend
    handles = [plt.Rectangle((0,0),1,1,facecolor=METHOD_COLORS[m], label=m)
               for m in METHODS]
    for pi, pname in enumerate(reversed(pnames)):   # darkest last at top
        alpha = 0.3 + 0.7 * ( (n_probs-1-pi) / max(1,n_probs-1))
        handles.append(plt.Rectangle((0,0),1,1,facecolor='gray', alpha=alpha,
                                     label=pname))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"🖼  PNG -> {out_png}")


# --------------------------------------------------------------------------
# LaTeX (PGFPlots)
# --------------------------------------------------------------------------
def plot_tex(all_data, pnames, out_tex):
    test_cases = sorted(all_data)
    n_probs    = len(pnames)
    coords     = [f"{int(tc)}-{m}-{i}"
                  for tc in test_cases for m in METHODS for i in range(n_probs)]

    # Header
    tex = r"""\documentclass[landscape]{article}
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
  xtick={""" + ", ".join(f"{int(tc)}-Ens-0" for tc in test_cases) + r"""},
  xticklabels={""" + ", ".join(str(int(tc)) for tc in test_cases) + r"""},
  x tick label style={rotate=45, anchor=east},
  xlabel={Test Cases},
  ylabel={Probability},
  ymin=0, ymax=""" + str(Y_MAX) + r""",
  legend style={at={(1.02,1)}, anchor=north west},
  grid=major]
"""

    color_lut = {'Ens':'blue','BF':'orange','SA':'green','GA':'red'}
    for pi in range(n_probs):
        actual_idx = n_probs - 1 - pi     # darkest first in legend
        pname      = pnames[actual_idx]
        for m in METHODS:
            pts = []
            for tc in test_cases:
                avg, _ = all_data[tc][m][actual_idx]
                pts.append(f"({int(tc)}-{m}-{pi}, {avg:.3f})")
            intensity = 10 + (80*pi/max(1,n_probs-1))
            tex += rf"""
% {pname} – {m}
\addplot+[
  bar shift={BAR_SHIFT_PT*pi}pt,
  draw=white,
  fill=black!{int(intensity)}!{color_lut[m]}
] coordinates {{ {' '.join(pts)} }};"""

    # Legend (methods then problems)
    legend = ', '.join(METHODS + list(reversed(pnames)))
    tex += rf"""
\legend{{{legend}}}
\end{{axis}}
\end{{tikzpicture}}
\caption{{Method performance across test cases with different problems.}}
\end{{figure}}
\end{{document}}"""

    with open(out_tex, 'w', encoding='utf8') as f:
        f.write(tex)
    print(f"📄 LaTeX -> {out_tex}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python excel_to_plot_latex_bar_multi.py file1.xlsx …")
        sys.exit(1)
    files = sys.argv[1:7]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("No valid Excel files.")
        sys.exit(1)

    data, pnames = collect_all(files)
    out_dir      = Path(files[0]).parent
    png_path     = out_dir / "multi_excel_bar_chart.png"
    tex_path     = out_dir / "multi_excel_bar_chart.tex"

    plot_png(data, pnames, png_path)
    plot_tex(data, pnames, tex_path)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
