# When Does a Machine Learning Model Fail?
### Linking Feature-Space Changes to Prediction Degradation

Code, results and IEEE conference paper for an empirical study of **which label-free
feature-space signals actually predict a model's accuracy drop under distribution shift**,
and why most drift alarms do not.

**Paper:** [`paper/main.pdf`](paper/main.pdf) (IEEEtran, two-column conference format)

## Key findings

| Finding | Number |
|---|---|
| Natural shift pairs / total evaluations | 173 pairs × 3 models × 3 seeds = **1,557** |
| Evaluations where a domain classifier separates source vs target (AUC ≥ 0.95) | **68%** |
| Drift alarms (AUC ≥ 0.75) that are benign (no harm) | **25%** |
| Low-drift evaluations that still fail (silent failures) | **48%** |
| Best single signal: Importance-Weighted Shift (IWS-KS) | AUROC **0.70**, AUPRC **0.86** |
| Confidence-based estimators (ATC, DoC) — Spearman ρ with the drop | **≈ 0.07** |
| Harmful evaluations dominated by the conditional P(Y\|X) part | **76%** |

## Method in one line

**IWS** = Σ_j w_j · d_j, where d_j is the drift (KS or Wasserstein) of feature j between
source and target, and w_j is the model's normalized permutation importance for feature j
(computed once on labelled source data). Drift in features the model ignores contributes
nothing.

## Repository layout

```
src/
  data.py               dataset loaders + natural domain splits (7 public datasets)
  signals.py            13 label-free signals incl. IWS, ATC, DoC, MMD, domain classifier
  run_natural.py        E1/E3/E4: every source->target pair x model (one CSV per dataset/seed)
  run_synthetic.py      E2: controlled shifts with known ground truth (5 families)
  analysis.py           all tables (LaTeX + CSV), figures, significance tests, taxonomy
  case_study.py         per-feature case-study figure (Elec2, Housing)
  ablation_samplesize.py how many unlabeled target rows each signal needs
research/
  RESEARCH.md           research questions, literature review, gap, method, experiments, sources
scripts/
  download_data.sh      fetch the raw data into data/raw/
  run_all.sh            reproduce everything end to end
results/                raw per-evaluation CSVs, results/tables/*.tex|csv, summary.json
figures/                all figures (PDF + PNG)
paper/                  main.tex (single self-contained file, upload this to Overleaf), main_modular.tex (editable source), refs.bib, figures/, tables/, main.pdf
```

## Reproduce

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # ~30-40 min on a 2-core laptop
```

Or step by step:

```bash
bash scripts/download_data.sh
cd src
OMP_NUM_THREADS=1 python run_natural.py --seed 0   # repeat for seeds 1 and 2
python run_synthetic.py
python analysis.py
python case_study.py
python ablation_samplesize.py
cd ../paper && latexmk -pdf main_modular.tex && python3 ../scripts/make_single_file.py
```

## Datasets

| Dataset | Shift axes | Source |
|---|---|---|
| Adult | sex, race, age, education, workclass | UCI (Kohavi, 1996) |
| COMPAS | race, sex, age group | ProPublica (2016) |
| California Housing | ocean proximity, north/south | Pace & Barry (1997) |
| Bank Marketing | time (6 blocks, 2008–2010) | Moro et al. (2014) |
| Elec2 | time (8 blocks) | Harries (1999) |
| Airlines | carrier (top 8) | Ikonomovska et al. (2011) |
| Covertype | wilderness area | Blackard & Dean (1999) |

Raw data files are not committed (see `.gitignore`); `scripts/download_data.sh` fetches them.

## Before submitting

- Fill in author names and affiliations in `paper/main.tex` (or keep them anonymous if the venue is double-blind).
- Check the target conference's page limit and whether references count toward it.

## Overleaf

`paper/main.tex` is self-contained (tables and bibliography are inlined). Upload only `main.tex` and the `figures/` folder (5 PDFs) and compile with pdfLaTeX. Edit `main_modular.tex` and re-run `scripts/make_single_file.py` to regenerate it.
