#!/usr/bin/env python3
"""
generate_sig_table_best.py

Creates LaTeX table comparing 'best' scores among four methods
(Ans, BF, SA, GA) using one-sided Wilcoxon signed-rank tests (A > B).

Usage:
    python generate_sig_table_best.py <directory_with_excels>
"""

import sys, re
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

METHODS = ["Ans", "BF", "SA", "GA"]
PAIRS = [(a, b) for i, a in enumerate(METHODS) for b in METHODS[i+1:]]
ALPHA = 0.05

def tikz_box(p, color):
    if color == "blank":
        return ""
    label = "<0.001" if p < 0.001 else f"{p:.3f}"
    return (
        r"\tikz[baseline]{\node[draw="
        + color +
        ", fill=" + color + "!25, inner sep=2pt, rounded corners=1pt]"
        "{" + label + "};}"
    )

def collect_best(df, method):
    cols = [c for c in df.columns if isinstance(c, str) and c.endswith(f"{method}_best")]
    if not cols:
        return np.array([])
    data = df[cols].dropna().values.flatten()
    return data[~np.isnan(data)]

def process_file(filepath):
    df = pd.read_excel(filepath)
    vecs = {m: collect_best(df, m) for m in METHODS}
    if any(len(v) == 0 for v in vecs.values()):
        return None

    problem = re.sub(r"^results_|_.*$", "", filepath.stem)
    row = [problem]

    for a, b in PAIRS:
        x, y = vecs[a], vecs[b]
        if len(x) != len(y) or len(x) < 10:
            row.append("")
            continue
        try:
            p = wilcoxon(x, y, alternative='greater').pvalue
        except Exception:
            row.append("")
            continue
        if p <= ALPHA and np.mean(x) > np.mean(y):
            row.append(tikz_box(p, "green"))
        elif 0.05 < p < 0.95:
            row.append(tikz_box(p, "gray"))
        else:
            # row.append("")
            row.append(tikz_box(p, "red"))

    return row

def main(directory: str):
    d = Path(directory)
    headers = ["Problem"] + [f"{a}$\\leftrightarrow${b}" for a, b in PAIRS]
    rows = []

    for f in sorted(d.glob("results_*.xlsx")):
        row = process_file(f)
        if row:
            rows.append(row)
            
    tex = [
        r"\begin{table}[ht]",
        r"\small",
        r"\setlength{\tabcolsep}{1pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\centering",
        r"\begin{tabular}{|l|" + "c|" * (len(headers)-1) + r"}",
        r"\hline",
        " & ".join(headers) + r" \\ \hline"
    ]
    for r in rows:
        tex.append(" & ".join(r) + r" \\ \hline")
    tex += [
        r"\end{tabular}",
        r"\caption{Wilcoxon one-sided signed-rank test results on \textbf{best} scores. "
        r"Each cell shows the $p$-value for the hypothesis that the left method "
        r"performs better than the right (e.g., Ans$\rightarrow$BF). "
        r"\textcolor{green}{Green boxes} indicate significant results ($p\le0.05$), "
        r"\textcolor{gray}{Gray boxes} indicate no significance ($0.05<p<0.95$), "
        r"and \textcolor{red}{Red boxes} indicate evidence in the opposite direction.}",
        r"\label{tab:wilcoxon-best}",
        r"\end{table}"
    ]
    output = d / "comparison_table_best.tex"
    output.write_text("\n".join(tex), encoding="utf8")
    print("✅ LaTeX table saved to:", output)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_sig_table_best.py <directory>")
        sys.exit(1)
    main(sys.argv[1])
