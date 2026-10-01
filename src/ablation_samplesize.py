"""Ablation: how many unlabeled target rows does each signal need?

GBDT, seed 0, all 173 natural pairs. For n_T in {100, 300, 1000, 5000} we recompute
mean KS, domain-classifier AUC, IWS-KS and ATC drop on a random target subsample and
score them against the accuracy drop measured on the full 5,000-row target sample.
Writes results/tables/e5_sample_size.csv and .tex
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import data as D
from analysis import TAB, TAU
from run_natural import MAX_TGT, MAX_TRAIN, MAX_VAL, make_model
from signals import atc_estimate, domain_classifier, feature_importance, per_feature_drift

warnings.filterwarnings("ignore")
SIZES = [100, 300, 1000, 5000]


def main(seed=0):
    rows = []
    for name in D.LOADERS:
        ds = D.LOADERS[name]()
        X_all, y_all = ds["X"], ds["y"]
        rng = np.random.default_rng(seed)
        by_source = {}
        for attr, s, t in D.domain_pairs(ds):
            by_source.setdefault((attr, s), []).append(t)
        for (attr, s), targets in by_source.items():
            dom = np.asarray(ds["domains"][attr])
            Xa = X_all.drop(columns=ds["drop"][attr]).values.astype(float)
            src = np.where(dom == s)[0]
            rng.shuffle(src)
            n_val = min(MAX_VAL, len(src) // 4)
            v, tr = src[:n_val], src[n_val:n_val + MAX_TRAIN]
            m = make_model("gbdt", seed).fit(Xa[tr], y_all[tr])
            w = feature_importance(m, Xa[v], y_all[v], seed=seed)
            ps = m.predict_proba(Xa[v])
            cs, correct = ps.max(1), (ps.argmax(1) == y_all[v]).astype(float)
            for t in targets:
                tgt = np.where(dom == t)[0]
                tgt = rng.choice(tgt, min(MAX_TGT, len(tgt)), replace=False)
                delta = correct.mean() - (m.predict(Xa[tgt]) == y_all[tgt]).mean()
                for n in SIZES:
                    sub = tgt[:n]
                    ks, _, _ = per_feature_drift(Xa[v], Xa[sub])
                    auc, _ = domain_classifier(Xa[v], Xa[sub], seed=seed)
                    ct = m.predict_proba(Xa[sub]).max(1)
                    rows.append(dict(dataset=name, attr=attr, source=s, target=t, n=n, delta=delta,
                                     ks_mean=ks.mean(), dc_auc=auc, iws_ks=float(np.sum(w * ks)),
                                     atc_drop=correct.mean() - atc_estimate(cs, correct, ct)))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "e5_sample_size_raw.csv"), index=False)
    out = []
    for n, g in df.groupby("n"):
        harm = (g.delta > TAU).astype(int)
        r = dict(n=n)
        for s in ["ks_mean", "dc_auc", "iws_ks", "atc_drop"]:
            r[f"{s}_rho"] = spearmanr(g[s], g.delta).correlation
            r[f"{s}_auroc"] = roc_auc_score(harm, g[s])
        out.append(r)
    out = pd.DataFrame(out)
    out.to_csv(os.path.join(TAB, "e5_sample_size.csv"), index=False)
    lines = [r"\begin{tabular}{rcccc}", r"\toprule",
             r"$n_T$ & KS (mean) & DC-AUC & ATC drop & IWS-KS$^\dagger$ \\", r"\midrule"]
    for _, r in out.iterrows():
        cells = [f"{r[s + '_rho']:.2f} / {r[s + '_auroc']:.2f}" for s in ["ks_mean", "dc_auc", "atc_drop", "iws_ks"]]
        lines.append(f"{int(r.n):,} & " .replace(",", "{,}") + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e5_sample_size.tex"), "w").write("\n".join(lines))
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
