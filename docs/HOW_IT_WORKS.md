# How the project works

## Where everything is
- Project folder (your Mac): `Downloads/ml-failure-feature-shift/` (git repo, remote `PriscillajospinG/ml-failure-feature-shift`, nothing pushed yet)
- Paper to submit: `paper/main.tex` (one file, paste into Overleaf) and `paper/main.pdf`
- Research: `research/RESEARCH.md` (plan, literature) and `research/FINDINGS.md` (results, analysis)
- Code: `src/`  |  Pipeline scripts: `scripts/`  |  Raw results: `results/`  |  Figures: `figures/`

## The idea in one paragraph
A deployed model never sees target labels, so teams watch "drift" statistics instead. We ask which
label-free signals actually predict the true accuracy drop. For each source -> target pair we train
a model on source, compute 13 label-free signals on the target features, then use held-out target
labels (evaluation only) to measure the real drop and see which signals track it.

## Pipeline (what runs, in order)
1. `scripts/download_data.sh` fetches 6 public datasets into `data/raw/` (Housing loads via scikit-learn).
2. `src/data.py` builds natural domain splits per dataset (sex, race, age, time blocks, carrier, region...),
   giving 173 ordered source -> target pairs.
3. `src/run_natural.py` (per seed 0/1/2), for every pair and model (LogReg, GBDT, MLP):
   - train on up to 20k source rows, hold out 5k source rows for validation, sample 5k target rows
   - `src/signals.py`: compute the signals (below) from source-val X and target X
   - measure `acc_s`, `acc_t`, `delta = acc_s - acc_t`, and the covariate/conditional split
   - write one CSV per dataset/seed to `results/` (1,557 rows in total)
4. `src/run_synthetic.py`: Gaussian data with controlled shifts (ignored features, used features by
   mean or spread, concept P(Y|X), label prior P(Y)) at 6 magnitudes, so the true cause is known.
5. `src/analysis.py`: correlations, AUROC/AUPRC, significance tests, decomposition, harm-score
   meta-regressor (leave-one-dataset-out), taxonomy, ablations -> `results/tables/`, `figures/`, `results/summary.json`.
6. `src/case_study.py`, `src/ablation_samplesize.py`: per-feature case study and target-size ablation.
7. `src/dump_figdata.py` + `scripts/make_tikz.py`: figures as pgfplots code in `paper/figures/`.
`bash scripts/run_all.sh` does all of it (about 30-40 minutes on a laptop).

## The 13 signals (all computed without target labels)
- Data-only: KS mean, KS max, Wasserstein, PSI, MMD^2, domain-classifier AUC (C2ST)
- Model-aware (ours): IWS-KS and IWS-Wass = sum_j w_j * d_j, with w_j the model's normalised
  permutation importance (source labels only) and d_j the drift of feature j
- Model-output: confidence drop (DoC), prediction shift, ATC drop, importance-weighted
  (density-ratio) accuracy drop, ensemble disagreement increase

## What is measured
- Target: `delta = Acc_source - Acc_target`; a shift is harmful when `delta > 2 pp`
- Decomposition: `delta = (Acc_S - Acc_S->T) + (Acc_S->T - Acc_T)`, covariate part from density-ratio
  reweighting, residual = conditional P(Y|X) part (plus low-overlap error)
- Scores: Spearman rho with delta, AUROC/AUPRC for harmful, FPR at 90% recall, bootstrap 95% CIs

## Files in `results/`
- `natural_<dataset>_seed<k>.csv`: one row per pair x model (accuracies, signals, decomposition)
- `synthetic.csv`: same columns for controlled shifts
- `summary.json`: headline numbers quoted in the paper; `figdata.json`: numbers behind the figures
- `tables/`: every paper table as .tex and .csv, plus raw ablation outputs

## Editing the paper
The paper is `paper/main.tex`. It pulls in separate files: `paper/figures/*.tex` (the figures as pgfplots code),
`paper/tables/*.tex` (the tables) and `paper/refs.bib` (the bibliography). Edit `main.tex` for text, rerun the scripts to
refresh figures and tables, then compile with `latexmk -pdf main.tex`. For Overleaf, upload the whole `paper/` folder (or its zip) and choose pdfLaTeX.
