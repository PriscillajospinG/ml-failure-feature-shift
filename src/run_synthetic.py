"""Experiment E2: controlled synthetic shifts with known ground-truth type.

Five shift families x six magnitudes x three seeds x three model classes.
  important   : mean shift of the features the label depends on (x1..x3)
  spread      : variance inflation of used features x2, x3 (forces extrapolation)
  irrelevant  : mean shift of the 7 features the label ignores (x4..x10)
  concept     : P(X) unchanged, P(Y|X) changed (effect of x1 reversed gradually)
  prior       : P(Y) changed by class-conditional resampling (label shift)
Writes results/synthetic.csv
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from run_natural import MODELS, REF, make_model
from signals import compute_signals, feature_importance, model_free_signals

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
D = 10
MAGS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
FAMILIES = ["important", "spread", "irrelevant", "concept", "prior"]


def f_true(X, concept_m=0.0):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    base = 1.5 * x1 + 2.0 * np.sin(1.5 * x2) + 1.0 * x3 * x1 - 0.3
    return base - concept_m * 1.5 * x1  # concept shift weakens / reverses the x1 effect


def sample(n, rng, mean_shift=None, concept_m=0.0):
    X = rng.normal(size=(n, D))
    if mean_shift is not None:
        X = X + mean_shift
    p = 1 / (1 + np.exp(-2.0 * f_true(X, concept_m)))
    y = (rng.random(n) < p).astype(int)
    return X, y


def sample_prior(n, rng, target_pos):
    X, y = sample(4 * n, rng)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    k = int(round(n * target_pos))
    k = min(k, len(pos))
    idx = np.r_[rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)]
    rng.shuffle(idx)
    return X[idx], y[idx]


def target(family, m, n, rng):
    if family == "important":
        s = np.zeros(D); s[:3] = m
        return sample(n, rng, mean_shift=s)
    if family == "spread":  # used features x2, x3 widen -> model must extrapolate
        X = rng.normal(size=(n, D))
        X[:, 1:3] *= 1 + m
        p = 1 / (1 + np.exp(-2.0 * f_true(X)))
        return X, (rng.random(n) < p).astype(int)
    if family == "irrelevant":
        s = np.zeros(D); s[3:] = m
        return sample(n, rng, mean_shift=s)
    if family == "concept":
        return sample(n, rng, concept_m=m)
    if family == "prior":
        return sample_prior(n, rng, min(0.5 + 0.2 * m, 0.92))
    raise ValueError(family)


def main():
    rows = []
    for seed in range(3):
        rng = np.random.default_rng(100 + seed)
        Xtr, ytr = sample(10000, rng)
        Xv, yv = sample(3000, rng)
        fitted = {m: make_model(m, seed).fit(Xtr, ytr) for m in MODELS}
        imps = {m: feature_importance(fitted[m], Xv, yv, seed=seed) for m in MODELS}
        for fam in FAMILIES:
            for mag in MAGS:
                Xt, yt = target(fam, mag, 3000, rng)
                cache = model_free_signals(Xv, Xt, seed=seed)
                for m in MODELS:
                    sig, ks, wd = compute_signals(fitted[m], fitted[REF[m]], Xv, yv, Xt, imps[m], seed=seed, cache=cache)
                    acc_s = float((fitted[m].predict(Xv) == yv).mean())
                    acc_t = float((fitted[m].predict(Xt) == yt).mean())
                    row = dict(dataset="synthetic", attr=fam, source="base", target=f"{fam}@{mag}",
                               magnitude=mag, model=m, seed=seed, acc_s=acc_s, acc_t=acc_t,
                               delta=acc_s - acc_t, prior_s=float(yv.mean()), prior_t=float(yt.mean()),
                               cov_part=acc_s - sig["acc_iw"], yx_part=sig["acc_iw"] - acc_t)
                    row.update({k: v for k, v in sig.items() if k != "acc_iw"})
                    rows.append(row)
                print(seed, fam, mag, f"delta(gbdt)={rows[-2]['delta']:+.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "synthetic.csv"), index=False)


if __name__ == "__main__":
    main()
