#!/usr/bin/env python3
"""
cross_problems_group.py            • 2025-07-02

Create a 2×2 grouped-bar figure (PNG *and* stand-alone PGFPlots LaTeX)
comparing four search methods across several problems and four budgets.

USAGE
    python cross_problems_group.py <results_dir> 500 1100 2100 3900
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ───────────────────────────── configuration ──────────────────────────────
METHODS           = ["Ens", "BF", "SA", "GA"]
COLUMN_FOR_METHOD = dict(zip(METHODS, [2, 11, 14, 15]))     # 0-based
BAR_COLOR         = dict(zip(METHODS, cm.tab10(range(len(METHODS)))))
HATCH             = {"Ens": "", "BF": "//", "SA": "xx", "GA": "--"}

LATEX_TEMPLATE = r"""\documentclass{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{pgfplots.groupplots}
\begin{document}
%<*body>
{body}
%</body>
\end{document}"""

# ───────────────────────────── helpers ────────────────────────────────────
def collect_xlsx(dir_: Path) -> List[Path]:
    return sorted(p for p in dir_.iterdir() if p.suffix.lower() == ".xlsx")


def read_sheet(path: Path, sizes: List[int]) -> Dict[int, Dict[str, tuple[float, float]]]:
    """Return {size: {method: (avg, std)}} for *one* Excel file."""
    df = pd.read_excel(path, header=None)
    avrg_rows = df[1] == "AVRG"
    std_rows  = df[1] == "STD"

    out: Dict[int, Dict[str, tuple[float, float]]] = {}
    for s in sizes:
        try:
            idx = df.index[(df.iloc[:, 0] == s) & avrg_rows][0]
        except IndexError:
            continue                                # budget not present
        avg_row, std_row = df.iloc[idx], df.iloc[idx + 1]
        out[s] = {m: (float(avg_row.iloc[COLUMN_FOR_METHOD[m]]),
                      float(std_row.iloc[COLUMN_FOR_METHOD[m]]))
                  for m in METHODS}
    return out


def build_df(root: Path, sizes: List[int]) -> pd.DataFrame:
    """Long-form dataframe: problem • size • method • avg • std"""
    recs = []
    for xl in collect_xlsx(root):
        try:
            problem = xl.stem.split("_")[1]         # “…_<ProblemName>_…”
        except IndexError:
            problem = xl.stem
        for size, mdict in read_sheet(xl, sizes).items():
            for m, (avg, std) in mdict.items():
                recs.append(dict(problem=problem, size=size, method=m,
                                 avg=avg, std=std))
    return pd.DataFrame(recs)


# ───────────────────────────── plotting (PNG) ─────────────────────────────
def draw_png(df: pd.DataFrame, sizes: List[int], png_out: Path) -> None:
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9))
    axes = axes.flatten()

    width_total = .8
    width_bar   = width_total / len(METHODS)

    for ax, size in zip(axes, sizes):
        sub = df[df["size"] == size].copy()
        sub.sort_values("problem", inplace=True)
        probs = sub["problem"].unique()
        x0    = np.arange(len(probs))

        for j, m in enumerate(METHODS):
            y   = sub[sub["method"] == m]["avg"].values
            err = sub[sub["method"] == m]["std"].values
            ax.bar(x0 + j*width_bar, y, width_bar,
                   color=BAR_COLOR[m], hatch=HATCH[m],
                   yerr=err, capsize=2,
                   label=m if size == sizes[0] else None)

        ax.set_xticks(x0 + width_total/2 - width_bar/2)
        ax.set_xticklabels(probs, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{size:,} test cases")
        ax.grid(axis="y", ls="--", alpha=.3)

    for extra in axes[len(sizes):]:
        extra.remove()

    fig.suptitle("Bug-trigger probability per problem", fontsize=12)
    fig.legend(loc="upper center", ncol=len(METHODS))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(png_out, dpi=300)
    plt.close(fig)


# ───────────────────────────── plotting (LaTeX) ───────────────────────────
def esc(s: str) -> str:
    """Minimal TeX escaping."""
    for a, b in [('\\', r'\textbackslash '), ('_', r'\_'), ('&', r'\&'),
                 ('%', r'\%'), ('#', r'\#'), ('{', r'\{'), ('}', r'\}')]:
        s = s.replace(a, b)
    return s


def build_tex(df: pd.DataFrame, sizes: List[int], tex_out: Path) -> None:
    lines: List[str] = []
    lines += [r"\begin{tikzpicture}",
              r"\begin{groupplot}[group style={group size=2 by 2,"
              r" horizontal sep=1.8cm, vertical sep=1.6cm},",
              r"ymin=0,ymax=1, ybar, bar width=.20cm,",
              r"enlarge x limits=0.13,",
              r"xlabel={Problem}, ylabel={Probability},",
              r"error bars/y dir=both, error bars/y explicit,",
              r"xticklabel style={rotate=45, anchor=east, font=\footnotesize},",
              r"legend style={at={(0.5,-0.15)}, anchor=north, font=\footnotesize},",
              r"legend columns=4]"]

    # single legend – we'll emit \addlegendentry only on first subplot
    for k, size in enumerate(sizes):
        sub = df[df["size"] == size].copy()
        sub.sort_values("problem", inplace=True)
        probs = [esc(p) for p in sub["problem"].unique()]

        lines.append(rf"\nextgroupplot[title={{\small {size:,} tests}},"
                     r"xticklabels={{" + ",".join(probs) + r"}}, xtick=data]")

        for m in METHODS:
            colour  = BAR_COLOR[m]
            hatch   = HATCH[m] or "solid"                  ### NEW ###
            coords  = []
            for i, p in enumerate(sub["problem"].unique()):
                row = sub[(sub["problem"] == p) & (sub["method"] == m)]
                if row.empty or np.isnan(row["avg"].values[0]):    # pragma: no cover
                    continue
                y, err = row["avg"].values[0], row["std"].values[0]
                coords.append(f"({i},{y}) +- (0,{err})")
            if not coords:
                continue
            lines.append(rf"\addplot+[draw opacity=0, "
                         rf"fill={m.lower()}!60, pattern={hatch}] "
                         r"coordinates {" + " ".join(coords) + "};")
            if k == 0:
                lines.append(rf"\addlegendentry{{{m}}}")

    lines += [r"\end{groupplot}", r"\end{tikzpicture}"]

    tex_out.write_text(LATEX_TEMPLATE.replace("{body}", "\n".join(lines)),
                       encoding="utf-8")


# ───────────────────────────── main ────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Grouped bar-chart PNG + LaTeX")
    ap.add_argument("results_dir", type=str)
    ap.add_argument("sizes", nargs=4, type=int, metavar="SIZE",
                    help="exactly four budgets (they map to the 4 subplots)")
    args = ap.parse_args()

    root  = Path(args.results_dir).resolve()
    if not root.is_dir():
        sys.exit("[ERR] results_dir not found")

    sizes = list(args.sizes)
    df    = build_df(root, sizes)
    if df.empty:
        sys.exit("[ERR] No matching data – check directory & budgets")

    draw_png(df, sizes, root / "group_plots.png")
    build_tex(df, sizes, root / "group_plots.tex")
    print("✓ PNG  ->", root / "group_plots.png")
    print("✓ LaTeX->", root / "group_plots.tex")


if __name__ == "__main__":
    main()
