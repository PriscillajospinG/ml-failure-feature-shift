# Findings and analysis

Numbers come from `results/summary.json` and `results/tables/`. 1,557 natural evaluations
(173 pairs x 3 models x 3 seeds, 7 datasets), plus a synthetic suite with known ground truth.

## Hypotheses and verdicts
| Hypothesis | Verdict |
|---|---|
| H1: raw feature drift correlates only weakly with accuracy loss | Supported: KS mean rho 0.22, PSI 0.23, MMD^2 0.33, domain-clf. AUC 0.40 (pooled) |
| H2: weighting drift by model importance predicts harm better | Supported but modest: IWS-KS AUROC 0.70 / AUPRC 0.86 vs 0.66 / 0.81 for the domain-clf. AUC; no significant gain in rho |
| H3: conditional P(Y\|X) change dominates natural failures | Supported: the residual part dominates 76% of harmful evaluations |
| H4: a learned harm score beats drift detectors | Not supported across datasets: the leave-one-dataset-out meta-regressor reaches AUROC 0.67, below IWS-KS alone (0.70) |

## Main results
- 71% of evaluations are harmful (> 2 pp); 17% have a negative drop (the target is easier).
- 87% of evaluations look drifted (domain-clf. AUC >= 0.75) and 68% are saturated (AUC >= 0.95), so
  the detector cannot rank them. Among drift alarms, 25% are benign.
- 48% of low-drift evaluations still lose more than 2 pp (silent failures).
- Pooled rho with the drop: IWS-Wass 0.42, IWS-KS 0.40, domain-clf. AUC 0.40, KS max 0.35, MMD^2 0.33;
  confidence-based estimators are near zero (DoC 0.07, ATC 0.07).
- Taxonomy of all evaluations: stable 7%, benign drift 22%, harmful covariate 17%, conditional failure 54%.
- Per-dataset behaviour differs: ATC works on Adult (rho 0.87) but is wrong-signed on Elec2 (-0.50);
  IWS-KS is best on Housing and Elec2; Airlines defeats every signal (carrier changes alter delay patterns, not inputs).

## Synthetic suite (what each signal can and cannot see)
- Ignored features shifted: domain-clf. AUC goes to 1.0, accuracy moves about 1 pp, IWS stays near 0.03.
- Used features shifted (mean): IWS tracks the change; accuracy can even improve when samples move away from the boundary.
- Concept shift P(Y|X): the drop reaches 28.8 pp while the AUC stays at 0.50. No label-free signal sees it.
- Label-prior shift: small drop through P(Y|X), partly visible to output-based signals.

## Ablations
- Model class: IWS-KS correlates with the drop for every model (GBDT 0.45, MLP 0.42, LogReg 0.33); ATC only works for GBDT.
- Harm threshold tau in {1, 2, 5, 10} pp: IWS-KS AUROC stays 0.70-0.71, domain classifier 0.66-0.69.
- Target sample size (100 to 5,000 rows): IWS-KS keeps the highest rank correlation at every size and
  already matches its full-sample AUROC with 1,000 rows.

## Interpretation
- Detecting a change in P(X) and being hurt by it are different questions; most alarms are uninformative.
- The failures that matter are mostly conditional, which only labels (even a small batch) can reveal.
- IWS is cheap (one permutation-importance pass per model) and removes false alarms from unused features.

## Threats to validity
- 7 public datasets with binary labels; regression and high-dimensional data may behave differently.
- Pairs from one dataset are correlated, so we report within-dataset statistics and leave-one-dataset-out results.
- The decomposition relies on density-ratio estimation; low overlap leaks into the residual (checked on synthetic data).
- Permutation importance is computed on source data and can change under strong shift.
- Detectron and PAPE were not re-implemented; the importance-weighted drop and disagreement stand in for them.

## Future work
- Small label budget (50 to 200 target labels) to catch conditional shift.
- Re-run on Folktables, TableShift and WhyShift for a direct comparison with prior benchmarks.
- Add regression targets, calibrated estimators (PAPE) and SHAP-based importance weights.
- Online monitoring: a streaming version of IWS with a sequential test.

## Practical guidance
1. Replace per-feature drift dashboards with importance-weighted drift.
2. Do not trust model confidence (ATC, DoC) under tabular shift.
3. Budget for a small stream of delayed labels, because conditional shift dominates.
4. Treat a saturated domain classifier as "data changed", not "model broke".
