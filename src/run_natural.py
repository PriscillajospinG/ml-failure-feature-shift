"""Experiment E1/E3/E4 data: every natural source->target pair x model class.

Usage:  python src/run_natural.py --datasets adult compas --seed 0
Writes  results/natural_<dataset>_seed<k>.csv  (one row per pair x model)
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import data as D
from signals import compute_signals, feature_importance, model_free_signals

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")

MAX_TRAIN, MAX_VAL, MAX_TGT = 20000, 5000, 5000


def make_model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if name == "gbdt":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              max_leaf_nodes=31, early_stopping=True,
                                              random_state=seed)
    if name == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-4, early_stopping=True,
            max_iter=200, random_state=seed))
    raise ValueError(name)


MODELS = ["logreg", "gbdt", "mlp"]
REF = {"logreg": "gbdt", "gbdt": "logreg", "mlp": "gbdt"}  # reference model for disagreement


def run_dataset(name: str, seed: int):
    ds = D.LOADERS[name]()
    X_all, y_all = ds["X"], ds["y"]
    rng = np.random.default_rng(seed)
    rows = []
    pairs = list(D.domain_pairs(ds))
    by_source = {}
    for attr, s, t in pairs:
        by_source.setdefault((attr, s), []).append(t)
    print(f"[{name}] {len(pairs)} pairs, {len(by_source)} sources", flush=True)
    for (attr, s), targets in by_source.items():
        dom = np.asarray(ds["domains"][attr])
        Xa = X_all.drop(columns=ds["drop"][attr]).values.astype(float)
        src = np.where(dom == s)[0]
        rng.shuffle(src)
        n_val = min(MAX_VAL, len(src) // 4)
        val_idx, tr_idx = src[:n_val], src[n_val:n_val + MAX_TRAIN]
        Xtr, ytr, Xv, yv = Xa[tr_idx], y_all[tr_idx], Xa[val_idx], y_all[val_idx]
        if len(np.unique(ytr)) < 2:
            continue
        fitted = {}
        for m in MODELS:
            mdl = make_model(m, seed)
            mdl.fit(Xtr, ytr)
            fitted[m] = mdl
        imps = {m: feature_importance(fitted[m], Xv, yv, seed=seed) for m in MODELS}
        for t in targets:
            tgt = np.where(dom == t)[0]
            tgt = rng.choice(tgt, min(MAX_TGT, len(tgt)), replace=False)
            Xt, yt = Xa[tgt], y_all[tgt]
            cache = model_free_signals(Xv, Xt, seed=seed)
            for m in MODELS:
                t0 = time.time()
                sig, ks, wd = compute_signals(fitted[m], fitted[REF[m]], Xv, yv, Xt, imps[m], seed=seed, cache=cache)
                acc_s = float((fitted[m].predict(Xv) == yv).mean())
                acc_t = float((fitted[m].predict(Xt) == yt).mean())
                row = dict(dataset=name, attr=attr, source=str(s), target=str(t), model=m,
                           seed=seed, n_train=len(tr_idx), n_val=len(val_idx), n_tgt=len(tgt),
                           acc_s=acc_s, acc_t=acc_t, delta=acc_s - acc_t,
                           prior_s=float(yv.mean()), prior_t=float(yt.mean()),
                           cov_part=acc_s - sig["acc_iw"], yx_part=sig["acc_iw"] - acc_t,
                           secs=time.time() - t0)
                row.update({k: v for k, v in sig.items() if k != "acc_iw"})
                rows.append(row)
            print(f"  {attr}: {s} -> {t}  delta(gbdt)={rows[-2]['delta']:+.3f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, f"natural_{name}_seed{seed}.csv"), index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(D.LOADERS))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for d in a.datasets:
        t0 = time.time()
        run_dataset(d, a.seed)
        print(f"[{d}] done in {time.time() - t0:.0f}s", flush=True)
