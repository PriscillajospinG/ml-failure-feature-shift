# When Does a Machine Learning Model Fail?
### Linking Feature-Space Changes to Prediction Degradation

Code, results and the IEEE conference paper for an empirical study of **which label-free feature-space signals actually predict a model's accuracy drop under distribution shift**, and why most drift alarms do not.

- **Paper:** [`paper/main.pdf`](paper/main.pdf), source in [`paper/main.tex`](paper/main.tex) (one self-contained file, paste it into Overleaf and compile with pdfLaTeX)
- **Research notes:** [`research/RESEARCH.md`](research/RESEARCH.md) (questions, literature, plan) and [`research/FINDINGS.md`](research/FINDINGS.md) (results and analysis)
- **Step-by-step pipeline guide:** [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md)

---

## 1. The problem in plain words

A model is trained on past data and deployed. Later the data it receives changes (new customers, a new year, a different region). The model's labels usually arrive late or never, so teams cannot measure the accuracy loss directly. They watch **drift detectors** instead: statistical tests that raise an alarm when the input features look different from the training data.

There are two ways this goes wrong:

1. **False alarm.** The features change, but the model does not use them, or the change moves data away from the decision boundary. Accuracy is fine, yet the detector fires. Teams retrain for nothing.
2. **Silent failure.** The relation between features and label, P(Y|X), changes. The features look the same, so no detector fires, yet accuracy collapses.

The question of this research: **given only unlabeled deployment data, which measurable change in feature space predicts that a model will degrade, and why do the others not?**

## 2. Research questions

| # | Question | Answered by |
|---|---|---|
| RQ1 | How strongly do standard drift statistics (KS, PSI, MMD, domain-classifier AUC) correlate with the real accuracy drop across many natural shifts? | Experiment E1 |
| RQ2 | Does drift weighted by how much the model relies on each feature predict harm better than raw drift? | E2 (synthetic) and E1 |
| RQ3 | Can the drop be split into a covariate part and a conditional P(Y\|X) part, and which dominates in practice? | E3 |
| RQ4 | Can a lightweight harm score built from label-free signals flag harmful shifts better than a drift detector alone? | E4 |

## 3. Contributions

1. **A shift-degradation testbed:** 173 natural source to target pairs from 7 public tabular datasets, 3 model classes, 3 seeds, giving **1,557 evaluations**. Each has the true accuracy drop and 13 label-free signals.
2. **Importance-Weighted Shift (IWS):** a simple model-aware statistic that weights per-feature drift by permutation importance.
3. **A covariate / conditional decomposition** of every drop through density-ratio reweighting, validated on synthetic shifts with known ground truth.
4. **A failure taxonomy** (stable, benign drift, harmful covariate shift, conditional failure) with guidance on when a drift alarm should trigger retraining.
5. **Open, reproducible code:** one script per table and figure.

## 4. Method

### 4.1 Setup

For every source domain S and target domain T:

1. Train a model f on source rows (up to 20,000). Hold out source validation rows (up to 5,000).
2. Sample up to 5,000 **unlabeled** target rows. This is all a deployed monitor would see.
3. Compute 13 label-free signals from source-validation features and target features.
4. Only then use target labels, purely for evaluation, to measure the real drop `Delta = Acc_S - Acc_T`.
5. A shift is **harmful** when `Delta > 2 percentage points` (varied from 1 to 10 in an ablation).
6. Score each signal against `Delta`: Spearman rank correlation, AUROC and AUPRC for detecting harmful shifts, false-positive rate at 90% recall.

### 4.2 The 13 label-free signals

| Group | Signals |
|---|---|
| Data-only (D) | mean KS, max KS, mean Wasserstein-1, mean PSI, MMD² (RBF kernel), domain-classifier AUC (C2ST) |
| Model-aware (M), proposed | **IWS-KS**, **IWS-Wass** |
| Model-output (O) | confidence drop (DoC), prediction shift, ATC drop, importance-weighted (density-ratio) drop, disagreement increase |

### 4.3 Importance-Weighted Shift (IWS)

```
IWS(f) = sum_j  w_j * d_j
w_j = max(I_j, 0) / sum_k max(I_k, 0)
```

- `d_j`: drift of feature j between source and target (KS statistic or standardized Wasserstein distance).
- `I_j`: permutation importance of feature j for model f, the accuracy lost on source validation data when feature j is shuffled. It needs source labels once and **no target labels**.
- **Intuition:** drift in a feature the model ignores gets weight zero. Drift in the features the model relies on counts fully. The paper proves a small invariance result: if only ignored features move, accuracy does not change, and IWS stays near zero while raw detectors fire.
- **Cost:** one permutation-importance pass per deployed model, then one KS test per feature for each new batch.

### 4.4 Covariate / conditional decomposition

A domain classifier gives density-ratio weights `r(x) = p(T|x) / (1 - p(T|x)) * n_S / n_T`. The reweighted source accuracy estimates what target accuracy would be **if only P(X) had changed**:

```
Delta = (Acc_S - Acc_S->T)        covariate part  (label-free, equals IW-drop)
      + (Acc_S->T - Acc_T)        residual part   (conditional P(Y|X) shift + low overlap)
```

The residual needs target labels and is used for analysis only. When source and target barely overlap, the residual also absorbs out-of-support error, so results are also reported on well-overlapping pairs.

### 4.5 Harm score

A gradient-boosted regressor maps all 13 signals to `Delta`. It is evaluated **leave-one-dataset-out**: it never sees the dataset it is scored on, which tests generalization to unseen domains.

## 5. Data and experiments

### 5.1 Datasets (natural shifts)

| Dataset | Shift axes | Pairs |
|---|---|---|
| Adult | sex, race, age, education, workclass | 34 |
| COMPAS | race, sex, age group | 14 |
| California Housing | ocean proximity, north/south | 14 |
| Bank Marketing | time (6 blocks, 2008 to 2010) | 15 |
| Elec2 | time (8 blocks) | 28 |
| Airlines | carrier (top 8) | 56 |
| Covertype | wilderness area (4) | 12 |

Every ordered pair of domains with at least 400 rows is a source to target pair, except temporal splits where only past to future pairs are kept. The split attribute is removed from the features. Raw data are not committed; `scripts/download_data.sh` fetches them.

### 5.2 Models

Logistic regression, histogram gradient-boosted trees (GBDT) and a two-layer MLP (128 and 64 units), all from scikit-learn, with three seeds that change sampling and initialization.

### 5.3 Synthetic suite

`X ~ N(0, I_10)` and `P(Y=1|x) = sigmoid(2[1.5 x1 + 2 sin(1.5 x2) + x1 x3 - 0.3])`, so only three features matter. Five shift families at six magnitudes: (a) mean shift of ignored features, (b) mean shift of used features, (c) spread (variance) of used features, (d) concept shift of P(Y|X) with P(X) fixed, (e) label-prior shift. The true cause is known, so each signal can be checked against ground truth.

### 5.4 Experiments

| ID | What it does | Output |
|---|---|---|
| E1 | Correlate every signal with the real drop over all natural pairs | correlation table, scatter figure |
| E2 | Controlled synthetic shifts | synthetic figure |
| E3 | Covariate / conditional decomposition | decomposition figure and table |
| E4 | Harm-score classifier, leave-one-dataset-out | AUROC/AUPRC table, precision-recall figure |
| E5 | Ablations: model class, harm threshold, target sample size | ablation tables |

## 6. Results

### 6.1 Headline numbers

| Finding | Number |
|---|---|
| Natural pairs / total evaluations | 173 pairs, **1,557** evaluations |
| Mean accuracy drop; evaluations above 2 pp | 10.9 pp; **71%** |
| Evaluations where the target is easier (negative drop) | 17% |
| Domain classifier separates source from target (AUC ≥ 0.95) | **68%** |
| Drift alarms (AUC ≥ 0.75) that turn out benign | **25%** |
| Low-drift evaluations that still fail (silent failures) | **48%** |
| Best single signal for harm detection: IWS-KS | AUROC **0.70**, AUPRC **0.86** |
| Highest rank correlation with the drop: IWS-Wass | ρ = **0.42** |
| Confidence-based estimators (ATC, DoC) | ρ ≈ **0.07** |
| Harmful evaluations dominated by the conditional P(Y\|X) part | **76%** |

### 6.2 What the results say

- **Drift is nearly universal but weakly informative.** A domain classifier alarms almost always, yet a quarter of those alarms are harmless and almost half of low-drift evaluations still fail. The alarm lifts the harm rate only from 48% to 75%.
- **Importance weighting helps.** IWS roughly doubles the correlation of plain KS (0.22 to 0.40) and gives the best harm detection among single signals. The gain over the domain classifier is modest: the pooled correlations are equal and the AUROC difference is not significant. IWS is the most reliable single signal, not a dominant one.
- **No signal wins everywhere.** ATC works on Adult (ρ = 0.87) but is wrong-signed on Elec2 (ρ = -0.50). Airlines defeats almost every signal because carrier changes alter delay patterns rather than inputs.
- **Synthetic checks match theory.** Shifting ignored features drives the domain-classifier AUC to 1.0 while accuracy moves about 1 pp and IWS stays flat. Shifting used features can even *improve* accuracy when samples move away from the decision boundary. Under pure concept shift the drop reaches 28.8 pp while the AUC stays at 0.50: **every label-free signal is blind to it.**
- **Conditional shift dominates natural failures.** The residual part dominates 76% of harmful evaluations (55% to 94% of the mean absolute drop depending on the dataset), and still 62% on pairs with good overlap.
- **A learned harm score does not transfer.** Leave-one-dataset-out, data-only signals reach AUROC 0.55, adding IWS gives 0.59 and all 13 signals give 0.67, still below IWS-KS alone (0.70).
- **Robustness.** IWS-KS works for every model class (ρ = 0.45 GBDT, 0.42 MLP, 0.33 logistic regression), is stable across harm thresholds (AUROC 0.70 to 0.71), and with only 100 unlabeled target rows its AUROC (0.73) matches the 5,000-row value.

### 6.3 Failure taxonomy

| Regime | Share | Meaning |
|---|---|---|
| Stable | 7% | little drift, no harm |
| Benign drift | 22% | drift alarm, no harm (wasted retraining) |
| Harmful covariate shift | 17% | the case drift detectors are built for |
| Conditional failure | 54% | P(Y\|X) changed, unnoticed by unlabeled monitoring |

### 6.4 Practical guidance

1. Replace raw per-feature drift dashboards with importance-weighted drift.
2. Do not rely on model confidence (ATC, DoC) for tabular monitoring.
3. Budget for a small stream of delayed labels, because conditional shift dominates.
4. Treat a saturated domain classifier as "the data changed", never as "the model broke".

## 7. Limitations and honest caveats

- Seven public datasets and binary classification only; regression and high-dimensional data are out of scope.
- Pairs from the same dataset are not independent, so within-dataset statistics and leave-one-dataset-out evaluation are reported.
- The decomposition depends on density-ratio estimation and is unreliable under near-complete separation (shown on synthetic data).
- Permutation importance is computed on source data and held fixed under shift.
- PAPE and Detectron were not re-implemented; the importance-weighted drop and disagreement signals stand in for them.
- The 2 pp harm threshold is a choice; the ablation shows the conclusions are stable from 1 to 10 pp.

## 8. Reproduce

```bash
pip install -r requirements.txt
bash scripts/run_all.sh          # about 30-40 minutes on a 2-core laptop
```

Step by step:

```bash
bash scripts/download_data.sh
cd src
OMP_NUM_THREADS=1 python run_natural.py --seed 0   # repeat for seeds 1 and 2
python run_synthetic.py
python analysis.py            # tables, significance tests, taxonomy, figures
python case_study.py          # per-feature case study (Elec2, Housing)
python ablation_samplesize.py # target sample-size ablation
python dump_figdata.py        # numbers behind the figures
python ../scripts/make_tikz.py          # figures as pgfplots code
cd ../paper && latexmk -pdf main_modular.tex && python3 ../scripts/make_single_file.py
```

## 9. Repository layout

```
src/
  data.py                 dataset loaders and natural domain splits
  signals.py              the 13 label-free signals (IWS, ATC, DoC, MMD, C2ST, ...)
  run_natural.py          every source->target pair x model (one CSV per dataset and seed)
  run_synthetic.py        controlled shifts with known ground truth
  analysis.py             tables, figures, significance tests, decomposition, taxonomy
  case_study.py           per-feature case-study figure
  ablation_samplesize.py  how many unlabeled target rows each signal needs
  dump_figdata.py         numbers behind the figures -> results/figdata.json
scripts/
  download_data.sh        fetch raw data into data/raw/
  run_all.sh              reproduce everything end to end
  make_tikz.py            write the figures as pgfplots code (paper/tikz/)
  make_single_file.py     inline tables, figures and bibliography into paper/main.tex
docs/HOW_IT_WORKS.md      pipeline step by step
research/                 RESEARCH.md (plan, literature) and FINDINGS.md (analysis)
results/                  raw per-evaluation CSVs, tables/, summary.json, figdata.json
figures/                  figures as PDF and PNG
paper/                    main.tex (single file), main_modular.tex (editable source),
                          refs.bib, tables/, tikz/, main.pdf
```

## 10. Overleaf

`paper/main.tex` is self-contained: tables, figures (native pgfplots) and the bibliography are inlined. Paste it into a new Overleaf project, choose **pdfLaTeX**, and compile. Edit `paper/main_modular.tex` and run `scripts/make_single_file.py` to regenerate it.

## 11. References used in the paper (all 2023 or later)

Hinder, Vaquet and Hammer (2024), concept-drift survey, parts A and B. Liu, Wang, Cui and Namkoong (2023), WhyShift. Biggs, Schrab and Gretton (2023), MMD-FUSE. Pandeva et al. (2024), E-valuating classifier two-sample tests. Nguyen et al. (2025), reliably detecting model failures without labels. Gardner, Popovic and Schmidt (2023), TableShift. Deng et al. (2023), confidence and dispersity for accuracy estimation. Xie et al. (2024), MaNo. Bialek et al. (2025), PAPE. Kato, Matsui and Inokuchi (2023), double debiased covariate shift adaptation. Ginsberg et al. (2023), Detectron. Amoukou et al. (2024), sequential harmful shift detection. Zhang et al. (2023), why did the model fail. Cai, Namkoong and Yadlowsky (2023), diagnosing model performance. Kulinski and Inouye (2023), explaining distribution shifts. Mougan et al. (2023), explanation shift. Decker et al. (2024), explanatory model monitoring. Zafar et al. (2026), FMMO. Kimura (2025), graph-smoothed Bayesian black-box shift estimator. Chamma, Thirion and Engemann (2024), variable importance and grouping. McElfresh et al. (2023), neural nets versus boosted trees on tabular data. Full entries are in `paper/refs.bib`.

## 12. Before submitting

- Replace the placeholder author names and affiliations in `paper/main.tex` (or anonymize if the venue is double-blind).
- Check the target conference's page limit and whether references count toward it.
