#!/usr/bin/env python
"""
cross_method_pattern.py
-----------------------
Draw a “pattern’’ plot for ONE method (Ens, BF, SA, or GA) across *all*
problems at several test-budget steps.

Example:
    python cross_method_pattern.py ./results BF 100 500 700

Outputs:
    BF_pattern_100_500_700.png
    BF_pattern_100_500_700.tex
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# ------------ column map ------------------------------------------------
COL_MAP = {"Ens": 2, "BF": 11, "SA": 14, "GA": 15}
BAD     = ['DIV/0!', '#DIV/0!', 'N/A', '#N/A', 'NaN', 'nan', None, '']

def to_float(v):
    if str(v).strip() in BAD: return 0.0
    try:
        return float(str(v).replace(',', '.').replace('%', '').strip())
    except ValueError:
        return 0.0

# ------------ gather data ----------------------------------------------
def collect(directory, method, budgets):
    problems, Y, E = [], {}, {}
    for b in budgets: Y[b], E[b] = [], []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".xlsx"): continue
        prob = fname.split("_")[1]
        problems.append(prob)
        df = pd.read_excel(os.path.join(directory, fname),
                           sheet_name="גיליון1", header=None)
        col = COL_MAP[method]
        for b in budgets:
            rows = df.index[df.iloc[:,0] == b].tolist()
            if rows:
                r = rows[0]
                y  = to_float(df.iat[r,   col])  # AVRG
                ey = to_float(df.iat[r+1, col])  # STD
            else:
                y, ey = np.nan, 0.0
            Y[b].append(y);  E[b].append(ey)
    return problems, Y, E

# ------------ PNG plot --------------------------------------------------
def plot_png(problems, Y, E, method, budgets, out_png):
    x = np.arange(len(problems))
    cmap    = cm.get_cmap('tab10', len(budgets))
    markers = ['o','x','s','^','v','*','P','D']
    plt.figure(figsize=(14,6))
    for i,b in enumerate(budgets):
        plt.errorbar(x, Y[b], yerr=E[b],
                     fmt=f'-{markers[i]}', color=cmap(i),
                     label=f'{b} tests', capsize=4)
    plt.xticks(x, problems, rotation=45)
    plt.ylabel("Probability"); plt.grid(True)
    plt.title(f"{method} across problems")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

# ------------ LaTeX (pgfplots) -----------------------------------------
def plot_tex(problems, Y, E, method, budgets, out_tex):
    pgfmark = ['*','x','square*','triangle*','diamond*','+','pentagon*','|']
    colors  = ['blue','orange','green','red','purple','brown','teal','gray']
    ticks   = ",".join(map(str, range(len(problems))))
    labels  = ",".join(problems)

    lines = [r"\begin{tikzpicture}",
             r"\begin{axis}[width=\linewidth,height=7cm,",
             r"  ymin=0,ymax=1,",
             rf" xtick={{{ticks}}},",
             rf" xticklabels={{{labels}}},",
             r" xlabel={Problem},ylabel={Probability},",
             r" xticklabel style={rotate=45, anchor=east},",
             r" legend style={at={(1.05,1)}, anchor=north west},",
             r" error bars/y dir=both, error bars/y explicit]"]

    for i,b in enumerate(budgets):
        coords = " ".join(
            f"({k},{0 if np.isnan(y) else y}) +- (0,{E[b][k]})"
            for k,y in enumerate(Y[b])
        )
        lines.append(
            rf"\addplot+[mark={pgfmark[i]},color={colors[i]},"
            r"error bars/.cd, y dir=both, y explicit] "
            rf"coordinates {{ {coords} }};")
        lines.append(rf"\addlegendentry{{{b}\ tests}}")

    lines += [r"\end{axis}", r"\end{tikzpicture}"]
    with open(out_tex,"w",encoding="utf-8") as f: f.write("\n".join(lines))

# ------------ main ------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python cross_method_pattern.py <folder> <method> <budget1> [budget2] ...")
        sys.exit(1)

    folder  = sys.argv[1]
    method  = sys.argv[2]
    budgets = list(map(int, sys.argv[3:]))
    if method not in COL_MAP:
        print("Method must be one of:", ", ".join(COL_MAP)); sys.exit(1)

    probs, Y, E = collect(folder, method, budgets)
    if not probs:
        print("No Excel files found."); sys.exit(1)

    tag = "_".join(map(str, budgets))
    png = os.path.join(folder, f"{method}_pattern_{tag}.png")
    tex = os.path.join(folder, f"{method}_pattern_{tag}.tex")

    plot_png(probs, Y, E, method, budgets, png)
    plot_tex(probs, Y, E, method, budgets, tex)

    print("Wrote:", png, "and", tex)
