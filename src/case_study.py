"""Case study figure: per-feature drift (KS) vs. model importance for two pairs.

Elec2 block 0 -> 7 (large drift in unused features, no harm) and
Housing <1H OCEAN -> INLAND (drift in used features, large harm). GBDT, seed 0.
Writes figures/fig_case.pdf and results/tables/case_study.csv
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
from analysis import FIG, TAB, PALETTE
from run_natural import make_model
from signals import feature_importance, per_feature_drift

CASES = [("elec", "time", 0, 7, "Elec2: time block 0 $\\to$ 7"),
         ("housing", "proximity", "<1H OCEAN", "INLAND", "Housing: coastal $\\to$ inland")]


def run_case(name, attr, s, t):
    ds = D.LOADERS[name]()
    dom = np.asarray(ds["domains"][attr])
    Xdf = ds["X"].drop(columns=ds["drop"][attr])
    X, y = Xdf.values.astype(float), ds["y"]
    rng = np.random.default_rng(0)
    src = np.where(dom == s)[0]
    rng.shuffle(src)
    v, tr = src[:5000], src[5000:25000]
    tg = rng.choice(np.where(dom == t)[0], 5000, replace=False)
    m = make_model("gbdt", 0).fit(X[tr], y[tr])
    w = feature_importance(m, X[v], y[v])
    ks, _, _ = per_feature_drift(X[v], X[tg])
    acc_s, acc_t = (m.predict(X[v]) == y[v]).mean(), (m.predict(X[tg]) == y[tg]).mean()
    df = pd.DataFrame(dict(feature=Xdf.columns, ks=ks, importance=w))
    return df, acc_s, acc_t


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.9))
    rows = []
    for ax, (name, attr, s, t, title) in zip(axes, CASES):
        full, acc_s, acc_t = run_case(name, attr, s, t)
        df = full.sort_values("ks", ascending=False).head(6).iloc[::-1]
        y = np.arange(len(df))
        ax.barh(y + 0.19, df.ks, 0.36, color=PALETTE[1], label="Drift (KS)")
        ax.barh(y - 0.19, df.importance, 0.36, color=PALETTE[0], label="Model importance $w_j$")
        ax.set_yticks(y, df.feature.str.replace("_", " "))
        ax.set_title(f"{title}   acc {100*acc_s:.1f}$\\to${100*acc_t:.1f}%", fontsize=8)
        ax.set_xlim(0, 1)
        rows.append(dict(case=name, acc_s=acc_s, acc_t=acc_t, ks_mean=full.ks.mean(),
                         iws_ks=float((full.ks * full.importance).sum())))
    axes[1].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_case.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_case.png"), dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "case_study.csv"), index=False)


if __name__ == "__main__":
    main()
