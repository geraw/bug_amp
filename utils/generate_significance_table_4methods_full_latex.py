#!/usr/bin/env python
"""
generate_significance_table_4methods_full_latex.py
--------------------------------------------------
• Reads ONE Excel file or every *.xls[x] in a directory.
• Creates one PNG per workbook  <base>_sig4.png   (unless --nopng).
• Creates ONE master .tex file in the SAME folder:
      – Single workbook  → <base>_sig4.tex
      – Directory        → generate_significance_table_4methods_full_latex.tex
Colour logic, statistics, and column mapping unchanged.
"""

import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# ------------------------------------------------------------------- config
ALPHA     = 0.05
COLS      = ["C", "L", "O", "P"]
LABELS    = ["Ensemble Classification",
             "Brute Force",
             "Simulated Annealing",
             "Genetic Algorithm"]

# ------------------------------------------------------------------- helpers
def excel_list(path: str):
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in sorted(os.listdir(path))
                if f.endswith(('.xls', '.xlsx'))]
    return [path] if path.endswith(('.xls', '.xlsx')) else []

def load_cols(path, col_letters):
    df = pd.read_excel(path, header=None)
    avrg = df[1] == "AVRG"
    return [df.loc[avrg, ord(c)-65].astype(float).to_numpy() for c in col_letters]

def wilcoxon_matrix(data):
    n = len(data)
    M = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = "diag"
                continue
            diff = data[j] - data[i]
            p_col = wilcoxon(diff, alternative="less").pvalue   # column better
            p_row = wilcoxon(diff, alternative="greater").pvalue # row better
            if p_col < p_row:
                col = "green" if p_col < ALPHA else "gray"; p = p_col
            else:
                col = "red"   if p_row < ALPHA else "gray"; p = p_row
            M[i][j] = (col, f"{p:.3f}", p)
    return M

def p_shade(p, alpha=ALPHA, lo=30, hi=80):
    if p >= alpha: return lo
    return int(lo + (alpha-p)/alpha * (hi-lo))

# ------------------------------------------------------------------- LaTeX
def matrix_to_block(M, labels, caption):
    n = len(labels)
    lines = [r"\pgfplotsset{compat=1.3,width=0.8\columnwidth}",
             r"\begin{figure}[H]\centering\vspace{3ex}",
             r"\begin{adjustbox}{max width=\textwidth}",
             r"\renewcommand{\arraystretch}{1.5}",
             r"\begin{tabular}{c|" + "c"*n + "}"]
    lines.append(" & " + " & ".join(labels) + r" \\ \hline")
    for i,row in enumerate(M):
        ln = labels[i]
        for j,c in enumerate(row):
            if c == "diag":
                ln += " & "
            else:
                col, ptxt, pval = c
                if col == "gray":
                    tikz = rf"\tikz[baseline=(char.base)]{{\node[fill=lightgray,rounded corners=0.5mm,inner sep=5pt] (char) {{{ptxt}}};}}"
                else:
                    base = "green" if col == "green" else "red"
                    tikz = rf"\tikz[baseline=(char.base)]{{\node[fill={base}!{p_shade(pval)}!white,rounded corners=0.5mm,inner sep=5pt] (char) {{{ptxt}}};}}"
                ln += f" & {tikz}"
        lines.append(ln + r" \\")
    lines.extend([r"\end{tabular}", r"\end{adjustbox}",
                  rf"\caption{{{caption}}}", rf"\label{{tab:{caption.replace(' ', '_')}}}",
                  r"\end{figure}", "%----------------------------------------", ""])
    return "\n".join(lines)

# ------------------------------------------------------------------- PNG
def matrix_to_png(M, labels, out_png):
    n=len(labels); cell=1.0
    fig,ax = plt.subplots(figsize=(4+n,4+n)); ax.axis('off')
    for i in range(n):
        for j in range(n):
            x,y=j*cell,(n-1-i)*cell
            if M[i][j]=="diag":
                col=(0.95,0.95,0.95,1); txt=""
            else:
                clr,txt,p = M[i][j]
                if clr=="gray":
                    col=(0.9,0.9,0.9,1)
                else:
                    base=np.array([0.8,1,0.8]) if clr=="green" else np.array([1,0.8,0.8])
                    col=base-((ALPHA-min(p,ALPHA))/ALPHA)*0.4
                ax.add_patch(plt.Rectangle((x,y),cell,cell,facecolor=col,edgecolor='black'))
            ax.text(x+cell/2,y+cell/2,txt,ha='center',va='center',fontsize=10)
    for k,lbl in enumerate(labels):
        ax.text(k*cell+cell/2,n*cell+0.1,lbl,ha='center',va='bottom',rotation=45)
        ax.text(-0.1,(n-1-k)*cell+cell/2,lbl,ha='right',va='center')
    ax.set_xlim(0,n*cell); ax.set_ylim(0,n*cell); plt.tight_layout()
    plt.savefig(out_png,dpi=300); plt.close()

# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="Excel file or directory")
    ap.add_argument("--nopng", action="store_true", help="skip PNG creation")
    args = ap.parse_args()

    sheets = excel_list(args.target)
    if not sheets:
        print("No Excel files found."); return

    out_dir = args.target if os.path.isdir(args.target) else os.path.dirname(args.target) or "."
    blocks  = []

    for xl in sheets:
        data   = load_cols(xl, COLS)
        M      = wilcoxon_matrix(data)
        parts = os.path.splitext(os.path.basename(xl))[0].split('_')
        token = parts[1] if len(parts) > 1 else parts[0]
        blocks.append(matrix_to_block(M, LABELS, token))
        if not args.nopng:
            png_path = os.path.join(out_dir, os.path.splitext(os.path.basename(xl))[0] + "_sig4.png")
            matrix_to_png(M, LABELS, png_path)
            print("PNG:", png_path)

    # master tex filename
    if len(sheets) == 1:
        base = os.path.splitext(os.path.basename(sheets[0]))[0]
        master = base + "_sig4.tex"
    else:
        master = os.path.splitext(os.path.basename(__file__))[0] + ".tex"

    master_path = os.path.join(out_dir, master)
    with open(master_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    print("LaTeX:", master_path)

if __name__ == "__main__":
    main()
