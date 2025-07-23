#!/usr/bin/env python
"""
generate_significance_table_4methods_light.py
--------------------------------------------
Directory or single Excel → many PNGs + ONE LaTeX file.
Colour scheme: lighter pastel, \scriptsize p-values.
"""

import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ALPHA  = 0.05
COLS   = ["C","L","O","P"]
LABELS = ["Ensemble classification","Brute Force","Simulated Annealing","Genetic Algorithm"]

# -------------------------------------------------------------------
def list_excels(p):
    return ([os.path.join(p,f) for f in sorted(os.listdir(p)) if f.endswith(('.xls','.xlsx'))]
            if os.path.isdir(p) else
            [p] if p.endswith(('.xls','.xlsx')) else [])

def load_cols(path, cols):
    df = pd.read_excel(path, header=None); avg = df[1]=="AVRG"
    return [df.loc[avg, ord(c)-65].astype(float).to_numpy() for c in cols]

def matrix(data):
    n=len(data); M=[[""]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i==j: M[i][j]="diag"; continue
            d = data[j]-data[i]
            p_col = wilcoxon(d,alternative="less").pvalue
            p_row = wilcoxon(d,alternative="greater").pvalue
            if p_col < p_row: c="green" if p_col<ALPHA else "gray"; p=p_col
            else:             c="red"   if p_row<ALPHA else "gray"; p=p_row
            M[i][j]=(c,f"{p:.3f}",p)
    return M

def pct(p,lo=20,hi=60):
    return lo if p>=ALPHA else int(lo+(ALPHA-p)/ALPHA*(hi-lo))

# ------------------------------------------------------------------- LaTeX
def latex_block(M, labels, caption):
    n=len(labels); lines=[r"\pgfplotsset{compat=1.3,width=0.8\columnwidth}",
                          r"\begin{figure}[H]\centering\vspace{3ex}",
                          r"\begin{adjustbox}{max width=\textwidth}",
                          r"\renewcommand{\arraystretch}{1.5}",
                          r"\begin{tabular}{p{3.5cm}|"+ "c"*n + "}"]
    lines.append(" & " + " & ".join(labels) + r" \\ \hline")
    for i,row in enumerate(M):
        ln = labels[i]
        for j,c in enumerate(row):
            if c=="diag": ln+=" & "; continue
            clr,txt,p=c
            if clr=="gray":
                node=rf"\tikz[baseline=(char.base)]{{\node[fill=lightgray,rounded corners=0.5mm,inner sep=4pt,font=\scriptsize] (char) {{{txt}}};}}"
            else:
                base="green" if clr=="green" else "red"
                node=rf"\tikz[baseline=(char.base)]{{\node[fill={base}!{pct(p)}!white,rounded corners=0.5mm,inner sep=4pt,font=\scriptsize] (char) {{{txt}}};}}"
            ln += " & " + node
        lines.append(ln + r" \\")
    lines.extend([r"\end{tabular}", r"\end{adjustbox}",
                  rf"\caption{{{caption}}}", rf"\label{{tab:{caption.replace(' ','_')}}}",
                  r"\end{figure}", "%----------------------------------------", ""])
    return "\n".join(lines)

# ------------------------------------------------------------------- PNG
def png(M, labels, out_png):
    n=len(labels); cell=1.0
    fig,ax=plt.subplots(figsize=(4+n,4+n)); ax.axis('off')
    for i in range(n):
        for j in range(n):
            x,y=j*cell,(n-1-i)*cell
            if M[i][j]=="diag": col=(0.95,0.95,0.95,1); txt=""
            else:
                clr,txt,p=M[i][j]
                if clr=="gray": col=(0.93,0.93,0.93,1)
                else:
                    base=np.array([0.85,1,0.85]) if clr=="green" else np.array([1,0.85,0.85])
                    col=base-((ALPHA-min(p,ALPHA))/ALPHA)*0.35
                ax.add_patch(plt.Rectangle((x,y),cell,cell,facecolor=col,edgecolor='black'))
            ax.text(x+cell/2,y+cell/2,txt,ha='center',va='center',fontsize=8)
    for k,lbl in enumerate(labels):
        ax.text(k*cell+cell/2,n*cell+0.05,lbl,ha='center',va='bottom',rotation=45,fontsize=9)
        ax.text(-0.1,(n-1-k)*cell+cell/2,lbl,ha='right',va='center',fontsize=9)
    ax.set_xlim(0,n*cell); ax.set_ylim(0,n*cell)
    plt.tight_layout(); plt.savefig(out_png,dpi=300); plt.close()

# ------------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("target"); ap.add_argument("--nopng",action="store_true")
    args=ap.parse_args()

    sheets=list_excels(args.target)
    if not sheets: print("No Excel files."); return
    out_dir = args.target if os.path.isdir(args.target) else os.path.dirname(args.target) or "."
    blocks=[]
    for xl in sheets:
        M  = matrix(load_cols(xl,COLS))
        cap= os.path.splitext(os.path.basename(xl))[0].split('_')[1] if len(os.path.splitext(os.path.basename(xl))[0].split('_'))>1 else os.path.splitext(os.path.basename(xl))[0]
        blocks.append(latex_block(M,LABELS,cap))
        if not args.nopng:
            png_path=os.path.join(out_dir, os.path.splitext(os.path.basename(xl))[0] + "_sig_light.png")
            png(M,LABELS,png_path); print("PNG:",png_path)

    master = (os.path.splitext(os.path.basename(sheets[0]))[0] + "_sig_light.tex"
              if len(sheets)==1 else
              os.path.splitext(os.path.basename(__file__))[0] + ".tex")
    master_path=os.path.join(out_dir, master)
    with open(master_path,"w",encoding="utf-8") as f: f.write("\n".join(blocks))
    print("LaTeX:", master_path)

if __name__=="__main__":
    main()
