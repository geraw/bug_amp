
"""
generate_significance_table_4methods_light.py

• Four methods: Ensemble classification, Brute Force, Simulated Annealing, Genetic Algorithm
• Directional one‑sided Wilcoxon signed‑rank (paired), α=0.05
• LaTeX table with lighter pastel shades + \scriptsize p‑values
• Optional PNG preview (lighter shades)
"""

import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ALPHA   = 0.05
COLS    = ["C", "L", "O", "P"]  # Excel columns
LABELS  = ["Ensemble classification",
           "Brute Force",
           "Simulated Annealing",
           "Genetic Algorithm"]

# ------------------------- Data loading --------------------------------
def load_cols(xls_path, columns):
    df = pd.read_excel(xls_path, header=None)
    avrg = df[1] == "AVRG"
    return [df.loc[avrg, ord(c)-65].astype(float).to_numpy() for c in columns]

# ------------------------- Statistics ----------------------------------
def compute_matrix(data):
    n = len(data)
    M = [["" for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = "diag"
                continue
            diff = data[j] - data[i]
            p_col = wilcoxon(diff, alternative="less").pvalue  # col better
            p_row = wilcoxon(diff, alternative="greater").pvalue  # row better
            if p_col < p_row:
                winner = "green" if p_col < ALPHA else "gray"
                p = p_col
            else:
                winner = "red" if p_row < ALPHA else "gray"
                p = p_row
            M[i][j] = (winner, f"{p:.3f}", p)
    return M

# ------------------------- Helpers -------------------------------------
def pct_shade(p, alpha=ALPHA, min_=20, max_=60):
    if p >= alpha:
        return min_
    frac = (alpha-p)/alpha
    return int(min_ + frac*(max_-min_))

# ------------------------- LaTeX ---------------------------------------
HEADER = r"""
\documentclass{article}
\usepackage{colortbl}
\usepackage{tikz}
\usepackage{adjustbox}
\usepackage{array}
\begin{document}
\begin{table}[h!]
\centering
\caption{Directional Wilcoxon significance matrix ($p<0.05$)}
\begin{adjustbox}{max width=\textwidth}
\renewcommand{\arraystretch}{1.5}
"""

FOOTER = r"""\end{adjustbox}
\end{table}
\end{document}
"""

def to_latex(M, labels, out_tex):
    n=len(labels)
    tex = HEADER
    tex += r"\begin{tabular}{p{3.5cm}|"+ "c"*n + "}\n"
    tex += " & " + " & ".join(labels) + r" \\\hline" + "\n"
    for i,row in enumerate(M):
        line = labels[i]
        for j,cell in enumerate(row):
            if cell=="diag":
                line += " & "
            else:
                colour, ptxt, pval = cell
                if colour=="gray":
                    tikz = rf"\tikz[baseline=(char.base)]{{\node[fill=lightgray,rounded corners=0.5mm,inner sep=4pt,font=\scriptsize] (char) {{{ptxt}}};}}"
                else:
                    base = "green" if colour=="green" else "red"
                    pct = pct_shade(pval)
                    tikz = rf"\tikz[baseline=(char.base)]{{\node[fill={base}!{pct}!white,rounded corners=0.5mm,inner sep=4pt,font=\scriptsize] (char) {{{ptxt}}};}}"
                line += f" & {tikz}"
        tex += line + r" \\" + "\n"
    tex += r"\end{tabular}" + FOOTER
    with open(out_tex,"w") as f:
        f.write(tex)

# ------------------------- PNG -----------------------------------------
def to_png(M, labels, out_png):
    n=len(labels); cell=1.0
    fig,ax = plt.subplots(figsize=(4+n,4+n))
    ax.axis('off')
    for i in range(n):
        for j in range(n):
            x,y = j*cell, (n-1-i)*cell
            if M[i][j]=="diag":
                col=(0.95,0.95,0.95,1); txt=""
            else:
                clr, txt, p = M[i][j]
                if clr=="gray":
                    col=(0.93,0.93,0.93,1)
                else:
                    base = np.array([0.85,1,0.85]) if clr=="green" else np.array([1,0.85,0.85])
                    intensity = (ALPHA - min(p,ALPHA))/ALPHA
                    col = base - intensity*0.35
                    col = np.clip(col,0,1)
                ax.add_patch(plt.Rectangle((x,y),cell,cell,facecolor=col,edgecolor='black'))
            ax.text(x+cell/2,y+cell/2,txt,ha='center',va='center',fontsize=8)
    for idx,lbl in enumerate(labels):
        ax.text(idx*cell+cell/2,n*cell+0.05,lbl,ha='center',va='bottom',fontsize=9,rotation=45)
        ax.text(-0.1,(n-1-idx)*cell+cell/2,lbl,ha='right',va='center',fontsize=9)
    ax.set_xlim(0,n*cell); ax.set_ylim(0,n*cell); plt.tight_layout()
    plt.savefig(out_png,dpi=300); plt.close()

# ------------------------- Main ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("excel")
    parser.add_argument("--png",action="store_true")
    args = parser.parse_args()

    data = load_cols(args.excel, COLS)
    M = compute_matrix(data)
    base = os.path.splitext(args.excel)[0]
    out_tex = base + "_sig_light.tex"
    to_latex(M, LABELS, out_tex)
    print("LaTeX:", out_tex)
    if args.png:
        out_png = base + "_sig_light.png"
        to_png(M, LABELS, out_png)
        print("PNG:", out_png)

if __name__=="__main__":
    main()
