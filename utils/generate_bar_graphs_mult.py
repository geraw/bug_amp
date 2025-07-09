#!/usr/bin/env python3
"""
generate_bar_graphs_mult.py

Creates one combined bar chart (PNG + LaTeX) from Excel files.
X-axis: Problems → Methods (side-by-side)
Each method: bars for multiple test cases (stacked front-to-back)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

# ── CONFIG ───────────────────────────────────────────
METHODS = ["Ens", "BF", "SA", "GA"]
COL_IDX = {"Ens": 2, "BF": 11, "SA": 14, "GA": 15}
HEX     = {"Ens": "#1f77b4", "BF": "#ff7f0e", "SA": "#2ca02c", "GA": "#d62728"}
TIKZCLR = {"Ens": "blue",    "BF": "red",  "SA": "green",   "GA": "orange"}
BAR_W   = 0.18
SHIFT   = 0.02
Y_MAX   = 1.05

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ── Helpers ──────────────────────────────────────────
def load_workbook(path: Path):
    df       = pd.read_excel(path, header=None, sheet_name=0)
    avg_mask = df.iloc[:, 1] == "AVRG"
    tcs      = df.loc[avg_mask, 0].astype(int).values
    data     = {m: df.loc[avg_mask, COL_IDX[m]].astype(float).values for m in METHODS}
    return tcs, data

def gather(directory: Path, test_cases: list[int]):
    problems, d_all = [], {}
    for f in sorted(directory.glob("*.xlsx")):
        pname = f.stem.split("_")[1] if "_" in f.stem else f.stem
        problems.append(pname)
        tcs, vals = load_workbook(f)
        for tc in test_cases:
            d_all.setdefault(tc, {m: [] for m in METHODS})
        for m in METHODS:
            for tc in test_cases:
                idx_tc = np.where(tcs == tc)[0]
                val    = vals[m][idx_tc[0]] if len(idx_tc) else np.nan
                d_all[tc][m].append(val)
    return problems, d_all

# ── PNG Output ───────────────────────────────────────
def png_chart(probs, data, tcs, out_png: Path):
    fig, ax = plt.subplots(figsize=(18, 8))
    base_x  = np.arange(len(probs)) * (len(METHODS) + 1)
    sorted_tcs = sorted(tcs)

    for mi, m in enumerate(METHODS):
        pos = base_x + mi
        for ci, tc in enumerate(sorted_tcs):
            shift  = (ci - (len(sorted_tcs) - 1) / 2) * SHIFT
            alpha  = 0.3 + 0.7 * ci / max(1, len(sorted_tcs) - 1)
            colour = to_rgba(HEX[m], alpha)
            ax.bar(pos + shift, data[tc][m], BAR_W,
                   color=colour, edgecolor="white", linewidth=0.4,
                   label=f"{m}-{tc}" if mi == 0 else "")

    ax.set_xticks(base_x + (len(METHODS) - 1) / 2)
    ax.set_xticklabels(probs, rotation=45, ha="right")
    ax.set_xlabel("Problems")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, Y_MAX)
    ax.set_title("Combined Bar Chart")
    ax.grid(axis="y", linestyle="--", alpha=.3)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), ncol=6, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✔ PNG saved to:", out_png)

# ── LaTeX PGFPlots Output ───────────────────────────
def tex_chart(probs, data, tcs, out_tex: Path):
    sorted_tcs = sorted(tcs)
    sym = [f"{p}-{m}" for p in probs for m in METHODS]
    tex = [
        r"\begin{figure}[H]",
        r"\centering",
        r"\rotatebox{270}{%",
        r"\begin{tikzpicture}",
        r"\begin{axis}[ybar, bar width=4pt,",
        rf"  symbolic x coords={{ {','.join(sym)} }},",
        rf"  xtick={{ {','.join(f'{p}-BF' for p in probs)} }},",
        rf"  xticklabels={{ {','.join(probs)} }},",
        r"  x tick label style={rotate=45, anchor=east},",
        r"  xlabel={Problems}, ylabel={Probability},",
        r"  legend style={at={(0.5,-0.25)},anchor=north,legend columns=6},",
        r"  ymin=0, ymax={Y_MAX},",
        r"  enlargelimits=0.05,",
        r"  scale only axis=true,",
        rf"  grid=major]"
    ]

    for ci, tc in enumerate(sorted_tcs):
        shift_pt  = (ci - (len(sorted_tcs) - 1) / 2) * 0.8
        intensity = int(30 + 70 * ci / max(1, len(sorted_tcs) - 1))
        for m in METHODS:
            pts = " ".join(
                f"({p}-{m},{data[tc][m][probs.index(p)]:.3f})"
                for p in probs
            )
            tex += [
                rf"\addplot+[bar shift={shift_pt:.2f}pt,",
                rf"  fill=black!{intensity}!{TIKZCLR[m]}, draw=white]",
                rf"  coordinates {{ {pts} }};"
            ]

    tex += [
        r"\legend{" + ", ".join(f"{m}-{tc}" for tc in sorted_tcs for m in METHODS) + r"}",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"}",
        r"\caption{Combined Bar Chart for Multiple Test Cases}",
        r"\label{fig:combined_bar_chart}",  # Adjust label as needed    
        r"\end{figure}"
    ]
    out_tex.write_text("\n".join(tex), encoding="utf8")
    print("✔ LaTeX saved to:", out_tex)

# ── Main ────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print("Usage: python generate_bar_graphs_mult.py <dir> <tc1> [tc2 ... up to 6]")
        sys.exit(1)

    root = Path(sys.argv[1])
    tcs  = list(map(int, sys.argv[2:8]))  # up to 6
    probs, data = gather(root, tcs)

    png_chart(probs, data, tcs, root / "combined_bar_conv.png")
    tex_chart(probs, data, tcs, root / "combined_bar_conv.tex")

if __name__ == "__main__":
    main()
