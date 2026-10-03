#!/usr/bin/env python3
"""Build paper/main.tex: ONE self-contained file (tables and bibliography inlined).

paper/main_modular.tex is the editable source (it \\input's tables/*.tex and tikz/*.tex and uses refs.bib).
paper/main.tex is generated from it and compiles on its own (no figures, tables or .bib needed),
e.g. in Overleaf. Run after `latexmk -pdf main_modular.tex` so main_modular.bbl exists.
"""
import os, re
os.chdir(os.path.join(os.path.dirname(__file__), "..", "paper"))
s = open("main_modular.tex").read()
s = re.sub(r"\\input\{((?:tables|tikz)/[^}]+\.tex)\}", lambda m: open(m.group(1)).read().strip(), s)
bib = "\\bibliographystyle{IEEEtran}\n\\bibliography{refs}"
assert bib in s
s = s.replace(bib, "% Bibliography embedded so this file compiles without refs.bib\n" + open("main_modular.bbl").read().strip())
assert "\\input{tables" not in s and "\\input{tikz" not in s and "includegraphics" not in s and "\\bibliography{" not in s
open("main.tex", "w").write(s)
print("wrote paper/main.tex", len(s.splitlines()), "lines")
