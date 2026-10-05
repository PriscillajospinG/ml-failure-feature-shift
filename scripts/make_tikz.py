#!/usr/bin/env python3
"""Write the five paper figures as native pgfplots/TikZ files (paper/figures/*.tex).

Reads results/figdata.json (src/dump_figdata.py). With the figures as LaTeX code the paper
needs no image files at all, so the single paper/main.tex compiles in Overleaf as is.
"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = json.load(open(os.path.join(ROOT, "results", "figdata.json")))
OUT = os.path.join(ROOT, "paper", "figures"); os.makedirs(OUT, exist_ok=True)

COL = {"blue": "2A78D6", "orange": "EB6834", "green": "1BAF7A", "gold": "EDA100", "pink": "E87BA4",
       "dgreen": "008300", "purple": "4A3AA7", "red": "E34948", "grey": "777777"}
DEFS = "".join(f"\\definecolor{{fc{k}}}{{HTML}}{{{v}}}\n" for k, v in COL.items())
DS = ["adult", "compas", "housing", "bank", "elec", "airlines", "covtype"]
LAB = {"adult": "Adult", "compas": "COMPAS", "housing": "Housing", "bank": "Bank", "elec": "Elec2",
       "airlines": "Airlines", "covtype": "Covertype"}
DSC = ["blue", "orange", "green", "gold", "pink", "dgreen", "purple"]
MK = ["*", "square*", "triangle*", "diamond*", "pentagon*", "star", "x"]
STYLE = r"""\pgfplotsset{compat=1.16, fig/.style={tick label style={font=\scriptsize}, label style={font=\scriptsize},
  title style={font=\scriptsize}, every axis plot/.append style={}, grid=major, grid style={draw=black!10, line width=0.3pt},
  axis line style={draw=black!50}, tick style={draw=black!50}, legend style={font=\tiny, draw=none, fill=none}}}
"""

def co(pts, nd=3):
    return " ".join(f"({x:.{nd}f},{y:.{nd}f})" for x, y in pts)

def write(name, body):
    open(os.path.join(OUT, name + ".tex"), "w").write(DEFS + STYLE + body.strip() + "\n")

# ---- scatter (figure*, 3 panels)
xl = {"dc_auc": "Domain-classifier AUC", "iws_ks": "IWS-KS", "iw_drop": "Importance-weighted drop"}
b = "\\begin{center}\n\\begin{tikzpicture}\n"
for i, s in enumerate(["dc_auc", "iws_ks", "iw_drop"]):
    pos = "" if i == 0 else f", at={{(a{i-1}.east)}}, anchor=west, xshift=0.9cm, yticklabels={{}}"
    leg = ", legend to name=scleg, legend columns=7, legend style={font=\\tiny, draw=none, /tikz/every even column/.append style={column sep=3pt}}" if i == 0 else ""
    yl = ", ylabel={Accuracy drop $\\Delta$ (pp)}" if i == 0 else ""
    allx = [p[0] for ds_ in DS for p in D["scatter"][s][ds_]]
    lo, hi = min(allx), max(allx); pad = 0.04 * (hi - lo)
    lim = f", xmin={lo - pad:.3f}, xmax={hi + pad:.3f}"
    b += (f"\\begin{{axis}}[name=a{i}, fig, width=0.355\\textwidth, height=5.2cm, xlabel={{{xl[s]}}}{lim}, "
          f"title={{Spearman $\\rho$ = {D['rho'][s]:.2f}}}{yl}{pos}{leg}]\n")
    for j, ds in enumerate(DS):
        pts = D["scatter"][s][ds]
        b += (f"\\addplot[only marks, mark={MK[j]}, mark size=1pt, fcl{DSC[j]}, draw opacity=0.6, fill opacity=0.5] "
              f"coordinates {{{co(pts)}}};\n")
        if i == 0:
            b += f"\\addlegendentry{{{LAB[ds]}}}\n"
    b += f"\\addplot[dashed, black!60, thin, forget plot, no marks] coordinates {{({lo - pad:.3f},2) ({hi + pad:.3f},2)}};\n\\end{{axis}}\n"
b = b.replace("fcl", "fc")
b += "\\end{tikzpicture}\\\\[2pt]\n\\ref{scleg}\n\\end{center}\n"
write("fig_scatter", b)

# ---- synthetic (5 panels)
fams = [("irrelevant", "(a) Ignored features"), ("important", "(b) Used feat., mean"),
        ("spread", "(c) Used feat., spread"), ("concept", "(d) Concept $P(Y|X)$"), ("prior", "(e) Label prior $P(Y)$")]
ser = [("delta", "Accuracy drop $\\Delta$", "blue", "*"), ("dc_auc_n", "Domain-clf. AUC (rescaled)", "orange", "square*"),
       ("iws_ks", "IWS-KS (ours)", "green", "triangle*"), ("iw_drop", "Importance-wtd. drop", "purple", "diamond*")]
b = "\\begin{center}\n\\begin{tikzpicture}\n"
for i, (f, t) in enumerate(fams):
    pos = "" if i == 0 else f", at={{(s{i-1}.east)}}, anchor=west, xshift=0.35cm, yticklabels={{}}"
    leg = ", legend to name=syleg, legend columns=4" if i == 0 else ""
    yl = ", ylabel={Value}" if i == 0 else ""
    b += (f"\\begin{{axis}}[name=s{i}, fig, width=0.215\\textwidth, height=4.6cm, ymin=-0.1, ymax=1.05, "
          f"title={{{t}}}, xlabel={{Shift magnitude}}{yl}{pos}{leg}]\n")
    for c, lab, colr, mk in ser:
        b += f"\\addplot[color=fc{colr}, mark={mk}, mark size=1.2pt, line width=0.7pt] coordinates {{{co(D['syn'][f][c])}}};\n"
        if i == 0:
            b += f"\\addlegendentry{{{lab}}}\n"
    b += "\\end{axis}\n"
b += "\\end{tikzpicture}\\\\[2pt]\n\\ref{syleg}\n\\end{center}\n"
write("fig_synthetic", b)

# ---- decomposition (grouped bars)
ds = [d for d in DS if d in D["decomp"]]
ticks = ",".join(LAB[d] for d in ds)
cov = " ".join(f"({LAB[d]},{D['decomp'][d][0]:.2f})" for d in ds)
yx = " ".join(f"({LAB[d]},{D['decomp'][d][1]:.2f})" for d in ds)
tot = " ".join(f"({LAB[d]},{D['decomp'][d][2]:.2f})" for d in ds)
write("fig_decomposition", rf"""
\begin{{tikzpicture}}
\begin{{axis}}[fig, ybar, bar width=4.5pt, width=\columnwidth, height=6.2cm, ymax=75, symbolic x coords={{{ticks}}}, xtick=data,
  x tick label style={{rotate=25, anchor=east}}, ylabel={{Mean contribution (pp)}}, enlarge x limits=0.1,
  legend style={{at={{(0.02,0.98)}}, anchor=north west, font=\tiny}}, ymajorgrids=true, xmajorgrids=false]
\addplot[fill=fcblue, draw=none] coordinates {{{cov}}};
\addplot[fill=fcorange, draw=none] coordinates {{{yx}}};
\addplot[only marks, mark=-, mark size=4pt, black!80, line width=1pt] coordinates {{{tot}}};
\legend{{Covariate part $P(X)$, Residual part ($P(Y|X)$ + low overlap), Total $\Delta$}}
\end{{axis}}
\end{{tikzpicture}}
""")

# ---- PR curves
sty = {"IWS-KS (ours)": ("fcgreen", "solid"), "All signals (harm score)": ("fcblue", "solid"),
       "Data-only signals": ("fcorange", "dashed"), "Domain-clf. AUC": ("fcgrey", "dotted"),
       "Importance-wtd. drop": ("fcpurple", "dash dot")}
b = ("\\begin{tikzpicture}\n\\begin{axis}[fig, width=\\columnwidth, height=5.8cm, xlabel={Recall (harmful shifts)}, ylabel={Precision}, "
     "ymin=0.6, ymax=1.02, xmin=0, xmax=1, legend style={at={(0.02,0.02)}, anchor=south west, font=\\tiny, inner sep=1pt}]\n")
for k, (c, ls) in sty.items():
    d = D["pr"][k]
    b += f"\\addplot[{c}, {ls}, line width=0.7pt, no marks] coordinates {{{co(d['pts'])}}};\n\\addlegendentry{{{k} (AP {d['ap']:.2f})}}\n"
b += f"\\addplot[black!40, dotted, thin, forget plot] coordinates {{(0,{D['base']:.3f}) (1,{D['base']:.3f})}};\n\\end{{axis}}\n\\end{{tikzpicture}}\n"
write("fig_pr", b)

# ---- case study (2 horizontal bar panels)
b = "\\begin{center}\n\\begin{tikzpicture}\n"
for i, c in enumerate(D["case"]):
    names = ",".join(r[0].replace("_", " ") for r in c["rows"])
    ks = " ".join(f"({r[1]:.3f},{j})" for j, r in enumerate(c["rows"]))
    im = " ".join(f"({r[2]:.3f},{j})" for j, r in enumerate(c["rows"]))
    pos = "" if i == 0 else f", at={{(c{i-1}.east)}}, anchor=west, xshift=2.2cm"
    leg = ", legend to name=cleg, legend columns=2" if i == 0 else ""
    b += (f"\\begin{{axis}}[name=c{i}, fig, xbar, bar width=3.5pt, width=0.34\\textwidth, height=4.8cm, xmin=0, xmax=1, "
          f"ytick={{0,...,5}}, yticklabels={{{names}}}, y tick label style={{font=\\tiny}}, enlarge y limits=0.12, "
          f"title={{{c['title']}\\quad acc {100*c['acc'][0]:.1f}$\\to${100*c['acc'][1]:.1f}\\%}}{pos}{leg}]\n"
          f"\\addplot[fill=fcorange, draw=none] coordinates {{{ks}}};\n"
          f"\\addplot[fill=fcblue, draw=none] coordinates {{{im}}};\n")
    if i == 0:
        b += "\\addlegendentry{Drift (KS)}\n\\addlegendentry{Model importance $w_j$}\n"
    b += "\\end{axis}\n"
b += "\\end{tikzpicture}\\\\[2pt]\n\\ref{cleg}\n\\end{center}\n"
write("fig_case", b)
print("wrote", sorted(os.listdir(OUT)))
