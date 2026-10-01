# Research notes: When Does a Machine Learning Model Fail?

Linking feature-space changes to prediction degradation (tabular data, natural shifts).

## 1. Problem and research questions

Drift detectors raise an alarm on any change in P(X), yet many such changes leave the model's
accuracy untouched, and some harmful failures (changes in P(Y|X)) are invisible to them.
Core question: given only unlabeled deployment data, can we tell *when* and *why* a trained
model will degrade?

- **RQ1 (correlation).** How strongly do standard feature-shift statistics (KS, PSI, MMD,
  domain-classifier AUC) correlate with the measured accuracy drop across many real shifts?
- **RQ2 (where the shift lands).** Does shift weighted by the model's reliance on each feature
  (Importance-Weighted Shift, IWS) predict degradation better than raw shift?
- **RQ3 (mechanism).** Can the drop be split into a covariate part and a conditional P(Y|X)
  part, and which dominates in practice?
- **RQ4 (early warning).** Can a lightweight "harm score" built from label-free signals flag
  harmful shifts with a better precision-recall trade-off than drift detectors alone?

## 2. Related work (four camps that rarely talk to each other)

| Work | Venue | Camp | Limit our paper targets |
|---|---|---|---|
| Rabanser et al., Failing Loudly | NeurIPS 2019 | Detection | Detects shift, not whether it hurts |
| Garg et al., ATC | ICLR 2022 | Estimation | Black-box number, no link to which features moved |
| Podkopaev & Ramdas, Tracking the risk of a deployed model | ICLR 2022 | Harmful shift | Needs labels |
| Ginsberg et al., Detectron | ICLR 2023 | Harmful shift | Mostly images; no feature-level explanation |
| Zhang et al., Why did the model fail? | ICML 2023 | Attribution | Needs target labels and a causal graph |
| Mougan et al., Explanation Shift | NeurIPS 2022 WS | Detection | Synthetic shifts only |
| Wang et al., WhyShift | NeurIPS 2023 | Empirical | 172 tabular pairs; Y|X shifts dominate; does not test detectors as predictors of the drop |
| Gardner et al., TableShift | NeurIPS 2023 D&B | Benchmark | Benchmark, not a degradation model |
| Biatek et al., PAPE | NeurIPS 2025 | Estimation | Covariate shift only; no "why" |
| Amoukou et al. | NeurIPS 2024 | Harmful shift | Detects harm, does not explain feature causes |

Anchoring facts: WhyShift shows Y|X shift dominates natural tabular shifts (invisible to P(X)
detectors); TableShift shows the accuracy gap tracks label shift. Hypothesis: raw feature
drift correlates only weakly with degradation.

## 3. Gap and contributions

No prior study measures, across hundreds of natural tabular shifts, how well each feature-space
signal predicts the actual accuracy drop, and why it fails when it does.

1. Large shift-degradation study: 7 datasets, 173 source-to-target pairs x 3 models x 3 seeds
   = 1,557 evaluations.
2. Importance-Weighted Shift (IWS): per-feature drift weighted by permutation importance
   (computed once, on labelled source data only).
3. Covariate / conditional decomposition via density-ratio reweighting.
4. Failure taxonomy (stable / benign drift / harmful covariate / conditional failure) and a
   leave-one-dataset-out harm score.
5. Open, reproducible code (one script per table/figure).

Novelty risk: PAPE and Amoukou et al. already estimate/test harm without labels. We position
as *explanatory* (which features, which shift type) and include their ideas as baselines
(importance-weighted density-ratio drop, disagreement increase).

## 4. Method

For each source-to-target pair compute label-free signals on the target X only, then measure
the true accuracy drop with held-out target labels and test which signals predict it.

- Raw drift: per-feature KS (mean/max), Wasserstein, PSI, MMD^2, domain-classifier AUC.
- Model-aware drift: IWS-KS and IWS-Wass, `IWS = sum_j w_j d_j`, `w_j` = normalized permutation
  importance, `d_j` = per-feature drift.
- Model-output signals: confidence drop (DoC), prediction shift, ATC drop, density-ratio
  importance-weighted drop, ensemble disagreement increase.
- Target: `Delta = Acc_S - Acc_T`; harmful iff `Delta > tau` with `tau = 2 pp`.
- Decomposition: `Delta = (Acc_S - Acc_S->T) + (Acc_S->T - Acc_T)` (covariate + conditional).
- Analysis: Spearman rho, AUROC/AUPRC for harmful pairs, leave-one-dataset-out meta-regressor.

## 5. Datasets, models, experiments

Natural shifts (see README for sources): Adult, COMPAS, California Housing, Bank Marketing,
Elec2, Airlines, Covertype. Synthetic suite: 5 shift families (important, spread, irrelevant,
concept, prior) x 6 magnitudes. Models: logistic regression, HistGradientBoosting, MLP.

| # | Experiment | Answers | Output |
|---|---|---|---|
| E1 | Correlate every signal with Delta across natural pairs | RQ1 | Correlation table, scatter figure |
| E2 | Synthetic shifts on important vs irrelevant features | RQ2 | Synthetic figure |
| E3 | Covariate / conditional decomposition | RQ3 | Decomposition figure |
| E4 | Harm-score classifier, leave-one-dataset-out | RQ4 | AUROC/AUPRC table, PR figure |
| E5 | Ablations: model class, target sample size, threshold tau | Robustness | Sample-size table |

## 6. Main findings (numbers from `results/summary.json`)

- 68% of evaluations: a domain classifier separates source from target (AUC >= 0.95).
- 25% of drift alarms (AUC >= 0.75) are benign (no harm); 48% of low-drift evaluations still fail.
- Best single signal: IWS-KS, AUROC 0.70, AUPRC 0.86; confidence-based estimators (ATC, DoC)
  have Spearman rho of about 0.07 with the drop.
- 76% of harmful evaluations are dominated by the conditional P(Y|X) component.

## 7. Limitations / risks

- IWS gives a modest, not dramatic, gain over raw drift; reported with honest significance.
- Conditional shift dominates and is not visible to any label-free signal (a "silent failure"
  result); a small-label-budget variant is natural future work.
- Datasets are public benchmarks, not Folktables/TableShift/WhyShift suites as first planned.

## 8. Target venues (check each CFP for deadlines and page limits)

IEEE ICDM, IEEE BigData, IEEE DSAA, IJCNN, IEEE ICMLA. Paper format: IEEEtran two-column,
8 pages plus references.

## 9. Sources

- Rabanser, Gunnemann, Lipton. Failing Loudly. NeurIPS 2019. http://papers.neurips.cc/paper/8420
- Garg et al. Leveraging Unlabeled Data to Predict OOD Performance. ICLR 2022. https://arxiv.org/abs/2201.04234
- Podkopaev, Ramdas. Tracking the Risk of a Deployed Model. ICLR 2022.
- Ginsberg et al. A Learning Based Hypothesis Test for Harmful Covariate Shift. ICLR 2023. https://arxiv.org/abs/2212.02742
- Zhang et al. "Why did the Model Fail?" ICML 2023. https://proceedings.mlr.press/v202/zhang23ai.html
- Mougan et al. Explanation Shift. NeurIPS 2022 workshop. https://arxiv.org/abs/2210.12369
- Wang, Liu, Cui, Namkoong. Rethinking Distribution Shifts (WhyShift). https://arxiv.org/abs/2307.05284
- Gardner, Popovic, Schmidt. TableShift. NeurIPS 2023. https://arxiv.org/abs/2312.07577
- Bialek et al. PAPE. NeurIPS 2025. https://arxiv.org/abs/2401.08348
- Amoukou et al. Sequential Harmful Shift Detection Without Labels. NeurIPS 2024. https://arxiv.org/abs/2412.12910
- Zafar, El-Sharif, Khan. FMMO. arXiv 2026. https://arxiv.org/html/2609.06173
- SGShift. arXiv 2025. https://arxiv.org/abs/2505.20634
- Root-Causing Performance Degradation Using Explainability. arXiv 2024. https://arxiv.org/abs/2403.02439
