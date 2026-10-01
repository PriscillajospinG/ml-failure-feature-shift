"""Analysis for all experiments: tables (LaTeX + CSV) and figures (PDF).

Usage: python src/analysis.py
Reads results/natural_*.csv and results/synthetic.csv
Writes results/tables/*.tex|csv, figures/*.pdf and results/summary.json
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signals import SIGNALS, SIGNAL_NAMES

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
TAB = os.path.join(RES, "tables")
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

TAU = 0.02  # harmful if accuracy drops by more than 2 percentage points
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
DS_ORDER = ["adult", "compas", "housing", "bank", "elec", "airlines", "covtype"]
DS_LABEL = {"adult": "Adult", "compas": "COMPAS", "housing": "Housing", "bank": "Bank",
            "elec": "Elec2", "airlines": "Airlines", "covtype": "Covertype"}
MODEL_LABEL = {"logreg": "LogReg", "gbdt": "GBDT", "mlp": "MLP"}

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.5, "axes.edgecolor": "#888888", "legend.frameon": False,
                     "pdf.fonttype": 42})


def load():
    nat = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "natural_*.csv")))],
                    ignore_index=True)
    syn = pd.read_csv(os.path.join(RES, "synthetic.csv"))
    return nat, syn


def boot_ci(x, y, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        v = fn(x[i], y[i])
        if np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, [2.5, 97.5])


def sp(x, y):
    return spearmanr(x, y).correlation


def safe_auc(lbl, s):
    return roc_auc_score(lbl, s) if 0 < lbl.sum() < len(lbl) else np.nan


# ----------------------------------------------------------------------------- E1
def e1_correlation(nat: pd.DataFrame):
    """Pooled & within-dataset rank correlation of each signal with the accuracy drop."""
    rows = []
    harm = (nat.delta > TAU).astype(int).values
    for s in SIGNALS:
        x, y = nat[s].values, nat.delta.values
        rho = sp(x, y)
        lo, hi = boot_ci(x, y, sp)
        within = [sp(g[s].values, g.delta.values) for _, g in nat.groupby(["dataset", "model", "seed"])
                  if g[s].nunique() > 1 and len(g) >= 5]
        aucs = [safe_auc((g.delta > TAU).astype(int).values, g[s].values)
                for _, g in nat.groupby(["dataset", "model", "seed"])]
        rows.append(dict(signal=s, name=SIGNAL_NAMES[s], rho=rho, rho_lo=lo, rho_hi=hi,
                         tau=kendalltau(x, y).correlation, rho_within=np.nanmean(within),
                         auroc=roc_auc_score(harm, x), auprc=average_precision_score(harm, x),
                         auroc_within=np.nanmean(aucs)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "e1_correlation.csv"), index=False)
    return df


def e1_tests(nat: pd.DataFrame):
    """Paired comparisons: per (dataset, model, seed) group Spearman, Wilcoxon signed-rank,
    Holm-corrected; plus paired bootstrap of the pooled rho difference."""
    from scipy.stats import wilcoxon
    groups = list(nat.groupby(["dataset", "model", "seed"]))
    def per_group(s):
        return np.array([sp(g[s].values, g.delta.values) for _, g in groups])
    ours = per_group("iws_ks")
    rows = []
    for b in ["ks_mean", "wass_mean", "psi_mean", "mmd", "dc_auc", "atc_drop", "conf_drop", "iw_drop"]:
        base = per_group(b)
        ok = np.isfinite(ours) & np.isfinite(base)
        p = wilcoxon(ours[ok], base[ok]).pvalue
        rng = np.random.default_rng(0)
        diffs = []
        x1, x2, y = nat.iws_ks.values, nat[b].values, nat.delta.values
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            diffs.append(sp(x1[i], y[i]) - sp(x2[i], y[i]))
        rows.append(dict(baseline=b, mean_diff_within=float(np.mean(ours[ok] - base[ok])),
                         wins=int(np.sum(ours[ok] > base[ok])), n_groups=int(ok.sum()), p=p,
                         pooled_diff=float(np.mean(diffs)), pooled_lo=float(np.percentile(diffs, 2.5)),
                         pooled_hi=float(np.percentile(diffs, 97.5))))
    df = pd.DataFrame(rows).sort_values("p")
    m = len(df)
    df["p_holm"] = np.minimum(1, np.maximum.accumulate(df.p.values * (m - np.arange(m))))
    df.to_csv(os.path.join(TAB, "e1_tests.csv"), index=False)
    return df


def e1_per_dataset(nat: pd.DataFrame):
    rows = []
    for ds in DS_ORDER:
        g = nat[nat.dataset == ds]
        r = dict(dataset=ds)
        for s in ["ks_mean", "dc_auc", "iws_ks", "atc_drop", "iw_drop"]:
            r[s] = sp(g[s], g.delta)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, "e1_per_dataset.csv"), index=False)
    lines = [r"\begin{tabular}{lccccc}", r"\toprule",
             r"Dataset & KS & DC-AUC & ATC & IW-drop & IWS-KS \\", r"\midrule"]
    for _, r in df.iterrows():
        vals = [r.ks_mean, r.dc_auc, r.atc_drop, r.iw_drop, r.iws_ks]
        best = np.nanmax(vals)
        cells = [(r"\underline{%.2f}" % v) if np.isclose(v, best) else "%.2f" % v for v in vals]
        lines.append(f"{DS_LABEL[r.dataset]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e1_per_dataset.tex"), "w").write("\n".join(lines))
    return df


# ----------------------------------------------------------------------------- E2
def e2_synthetic(syn: pd.DataFrame):
    g = syn.groupby(["attr", "magnitude"]).agg(
        delta=("delta", "mean"), delta_sd=("delta", "std"), dc_auc=("dc_auc", "mean"),
        ks_mean=("ks_mean", "mean"), iws_ks=("iws_ks", "mean"), iws_wass=("iws_wass", "mean"),
        wass_mean=("wass_mean", "mean"), iw_drop=("iw_drop", "mean"), atc_drop=("atc_drop", "mean"),
        conf_drop=("conf_drop", "mean"), cov_part=("cov_part", "mean"), yx_part=("yx_part", "mean"),
    ).reset_index()
    g.to_csv(os.path.join(TAB, "e2_synthetic.csv"), index=False)
    return g


# ----------------------------------------------------------------------------- E3
def e3_decomposition(nat: pd.DataFrame, syn: pd.DataFrame):
    rows = []
    for ds, g in nat.groupby("dataset"):
        h = g[g.delta > TAU]
        cov, yx = h.cov_part.abs().mean(), h.yx_part.abs().mean()
        rows.append(dict(dataset=ds, n_pairs=len(g), n_harm=len(h), frac_harm=len(h) / len(g),
                         mean_delta_harm=h.delta.mean(), cov_abs=cov, yx_abs=yx,
                         yx_share=yx / (cov + yx) if len(h) else np.nan,
                         yx_dominant=(h.yx_part.abs() > h.cov_part.abs()).mean() if len(h) else np.nan,
                         prior_gap=(g.prior_t - g.prior_s).abs().mean()))
    dec = pd.DataFrame(rows)
    ov = nat[(nat.delta > TAU) & (nat.dc_auc < 0.95)]
    dec_overlap = dict(n=len(ov), yx_share=float(ov.yx_part.abs().mean() / (ov.yx_part.abs().mean() + ov.cov_part.abs().mean())),
                       yx_dominant=float((ov.yx_part.abs() > ov.cov_part.abs()).mean()))
    json.dump(dec_overlap, open(os.path.join(TAB, "e3_overlap.json"), "w"), indent=1)
    dec.to_csv(os.path.join(TAB, "e3_decomposition.csv"), index=False)
    s2 = syn[syn.magnitude == syn.magnitude.max()].groupby("attr")[["delta", "cov_part", "yx_part"]].mean()
    s2.to_csv(os.path.join(TAB, "e3_synthetic_check.csv"))
    return dec, s2


# ----------------------------------------------------------------------------- E4
def e4_harm_score(nat: pd.DataFrame):
    """Leave-one-dataset-out meta-regressor that predicts the accuracy drop from signals."""
    data_free = ["ks_mean", "ks_max", "wass_mean", "psi_mean", "mmd", "dc_auc"]
    model_aware = data_free + ["iws_wass", "iws_ks"]
    full = SIGNALS
    configs = {"Data-only signals": data_free, "+ IWS": model_aware, "All signals (harm score)": full}
    out_rows, preds = [], {}
    for cname, feats in configs.items():
        pred = np.zeros(len(nat))
        for ds in nat.dataset.unique():
            tr, te = nat.dataset != ds, nat.dataset == ds
            reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                min_samples_leaf=20, random_state=0)
            reg.fit(nat.loc[tr, feats], nat.loc[tr, "delta"])
            pred[te.values] = reg.predict(nat.loc[te, feats])
        preds[cname] = pred
    # best single signals for reference
    for s in ["dc_auc", "iws_ks", "iw_drop", "atc_drop"]:
        preds[SIGNAL_NAMES[s]] = nat[s].values
    harm = (nat.delta > TAU).astype(int).values
    for cname, pred in preds.items():
        per = []
        for ds in DS_ORDER:
            m = (nat.dataset == ds).values
            if m.sum() == 0:
                continue
            per.append(dict(rho=sp(pred[m], nat.delta.values[m]), auroc=safe_auc(harm[m], pred[m])))
        per = pd.DataFrame(per)
        fpr_at_90 = _fpr_at_recall(harm, pred, 0.9)
        out_rows.append(dict(method=cname, rho_pooled=sp(pred, nat.delta.values),
                             auroc_pooled=roc_auc_score(harm, pred),
                             auprc_pooled=average_precision_score(harm, pred),
                             rho_lodo_mean=per.rho.mean(), auroc_lodo_mean=per.auroc.mean(),
                             fpr_at_90recall=fpr_at_90))
    df = pd.DataFrame(out_rows)
    df.to_csv(os.path.join(TAB, "e4_harm_score.csv"), index=False)
    # MAE of the regressor as an accuracy-drop estimator
    mae = {k: float(np.mean(np.abs(v - nat.delta.values))) for k, v in preds.items() if k in configs}
    return df, preds, mae


def _fpr_at_recall(lbl, score, recall):
    order = np.argsort(-score)
    l = lbl[order]
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    k = np.searchsorted(tp, recall * l.sum())
    return float(fp[min(k, len(fp) - 1)] / max((1 - l).sum(), 1))


# ----------------------------------------------------------------------------- E5
def e5_ablations(nat: pd.DataFrame):
    rows = []
    for m, g in nat.groupby("model"):
        for s in ["dc_auc", "ks_mean", "iws_ks", "iw_drop", "atc_drop", "conf_drop"]:
            rows.append(dict(model=m, signal=s, rho=sp(g[s], g.delta)))
    by_model = pd.DataFrame(rows).pivot(index="signal", columns="model", values="rho")
    by_model.to_csv(os.path.join(TAB, "e5_by_model.csv"))
    rows = []
    for tau in [0.01, 0.02, 0.05, 0.10]:
        lbl = (nat.delta > tau).astype(int).values
        for s in ["dc_auc", "ks_mean", "iws_ks", "iw_drop", "atc_drop"]:
            rows.append(dict(tau=tau, signal=s, auroc=roc_auc_score(lbl, nat[s]), base_rate=lbl.mean()))
    by_tau = pd.DataFrame(rows)
    by_tau.to_csv(os.path.join(TAB, "e5_by_tau.csv"), index=False)
    return by_model, by_tau


# ----------------------------------------------------------------------------- taxonomy
def taxonomy(nat: pd.DataFrame, auc_thr: float = 0.75):
    high = nat.dc_auc >= auc_thr
    harm = nat.delta > TAU
    yx_dom = nat.yx_part.abs() > nat.cov_part.abs()
    reg = np.select(
        [~high & ~harm, high & ~harm, harm & ~yx_dom, harm & yx_dom],
        ["Stable", "Benign drift", "Harmful covariate shift", "Concept (Y|X) failure"], "other")
    nat = nat.assign(regime=reg, high_drift=high)
    t = pd.crosstab(nat.regime, nat.high_drift, margins=True)
    t.to_csv(os.path.join(TAB, "taxonomy.csv"))
    return nat, t


def tex_taxonomy(nat2: pd.DataFrame):
    cols = ["Stable", "Benign drift", "Harmful covariate shift", "Concept (Y|X) failure"]
    t = (pd.crosstab(nat2.dataset, nat2.regime, normalize="index") * 100).reindex(DS_ORDER)
    t = t.reindex(columns=cols).fillna(0)
    allrow = (nat2.regime.value_counts(normalize=True) * 100).reindex(cols).fillna(0)
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Dataset & Stable & Benign drift & Harmful cov. & Conditional \\", r"\midrule"]
    for ds, r in t.iterrows():
        lines.append(f"{DS_LABEL[ds]} & " + " & ".join(f"{v:.0f}\\%" for v in r.values) + r" \\")
    lines.append(r"\midrule All & " + " & ".join(f"{v:.0f}\\%" for v in allrow.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "taxonomy.tex"), "w").write("\n".join(lines))
    t.to_csv(os.path.join(TAB, "taxonomy_by_dataset.csv"))
    return t


# ----------------------------------------------------------------------------- figures
def fig_scatter(nat):
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3), sharey=True)
    for ax, s in zip(axes, ["dc_auc", "iws_ks", "iw_drop"]):
        for i, ds in enumerate(DS_ORDER):
            g = nat[nat.dataset == ds]
            ax.scatter(g[s], 100 * g.delta, s=7, alpha=0.55, color=PALETTE[i], marker=MARKERS[i],
                       linewidths=0, label=DS_LABEL[ds])
        ax.axhline(100 * TAU, color="#555555", lw=0.7, ls="--")
        ax.set_xlabel(SIGNAL_NAMES[s].replace(" (ours)", ""))
        ax.set_title(f"Spearman $\\rho$ = {sp(nat[s], nat.delta):.2f}", fontsize=8)
    axes[0].set_ylabel("Accuracy drop $\\Delta$ (pp)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=7, bbox_to_anchor=(0.5, 1.07), markerscale=1.8,
               handletextpad=0.1, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_scatter.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_synthetic(syn):
    fams = ["irrelevant", "important", "spread", "concept", "prior"]
    titles = {"irrelevant": "(a) Ignored features", "important": "(b) Used feat., mean",
              "spread": "(c) Used feat., spread", "concept": "(d) Concept $P(Y|X)$",
              "prior": "(e) Label prior $P(Y)$"}
    series = [("delta", "Accuracy drop $\\Delta$", PALETTE[0], "o"),
              ("dc_auc_n", "Domain-clf. AUC (rescaled)", PALETTE[1], "s"),
              ("iws_ks", "IWS-KS (ours)", PALETTE[2], "^"),
              ("iw_drop", "Importance-wtd. drop", PALETTE[6], "D")]
    g = syn.groupby(["attr", "magnitude"]).mean(numeric_only=True).reset_index()
    g["dc_auc_n"] = 2 * (g.dc_auc - 0.5)  # 0.5..1 -> 0..1
    fig, axes = plt.subplots(1, 5, figsize=(7.16, 1.9), sharey=True)
    for ax, f in zip(axes, fams):
        h = g[g.attr == f]
        for col, lab, c, mk in series:
            ax.plot(h.magnitude, h[col], color=c, marker=mk, ms=3, lw=1.3, label=lab)
        ax.set_title(titles[f], fontsize=8)
        ax.set_xlabel("Shift magnitude")
        ax.axhline(0, color="#999999", lw=0.5)
    axes[0].set_ylabel("Value")
    hdl, lab = axes[0].get_legend_handles_labels()
    fig.legend(hdl, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_synthetic.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_synthetic.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_decomposition(nat):
    h = nat[nat.delta > TAU]
    agg = h.groupby("dataset")[["cov_part", "yx_part", "delta"]].mean().reindex(DS_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    x = np.arange(len(agg))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, 100 * agg.cov_part, w, color=PALETTE[0], label="Covariate part $P(X)$")
    ax.bar(x + w / 2 + 0.01, 100 * agg.yx_part, w, color=PALETTE[1], label="Residual part ($P(Y|X)$ + low overlap)")
    ax.scatter(x, 100 * agg.delta, color="#222222", marker="_", s=120, zorder=3, label="Total $\\Delta$")
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(x, [DS_LABEL[d] for d in agg.index], rotation=25)
    ax.set_ylabel("Mean contribution (pp)")
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_decomposition.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_decomposition.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pr(nat, preds):
    from sklearn.metrics import precision_recall_curve
    harm = (nat.delta > TAU).astype(int).values
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    show = [("IWS-KS (ours)", PALETTE[2], "-"), ("All signals (harm score)", PALETTE[0], "-"),
            ("Data-only signals", PALETTE[1], "--"), ("Domain-clf. AUC", "#777777", ":"),
            ("Importance-wtd. drop", PALETTE[6], "-.")]
    for k, c, ls in show:
        p, r, _ = precision_recall_curve(harm, preds[k])
        ax.plot(r, p, color=c, ls=ls, lw=1.3, label=f"{k} (AP {average_precision_score(harm, preds[k]):.2f})")
    ax.axhline(harm.mean(), color="#999999", lw=0.6, ls=":")
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("Recall (harmful shifts)")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pr.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, "fig_pr.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- LaTeX tables
def tex_e1(df):
    df = df.sort_values("rho", ascending=False)
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Signal & $\rho$ pooled [95\% CI] & $\bar\rho$ within & AUROC & AUPRC \\", r"\midrule"]
    groups = {"ks_mean": "D", "ks_max": "D", "wass_mean": "D", "psi_mean": "D", "mmd": "D", "dc_auc": "D",
              "iws_wass": "M", "iws_ks": "M", "conf_drop": "O", "pred_shift": "O", "atc_drop": "O",
              "iw_drop": "O", "disagree_inc": "O"}
    best = {c: df[c].max() for c in ["rho", "rho_within", "auroc", "auprc"]}
    for _, r in df.iterrows():
        f = lambda c, v: (r"\underline{%.2f}" % v) if np.isclose(v, best[c]) else "%.2f" % v
        nm = r['name'].replace(" (ours)", r"$^\dagger$")
        lines.append(f"{nm} ({groups[r.signal]}) & {f('rho', r.rho)} [{r.rho_lo:.2f}, {r.rho_hi:.2f}] & "
                     f"{f('rho_within', r.rho_within)} & {f('auroc', r.auroc)} & {f('auprc', r.auprc)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e1_correlation.tex"), "w").write("\n".join(lines))


def tex_e3(dec):
    dec = dec.set_index("dataset").reindex(DS_ORDER).dropna(how="all")
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Dataset & Evals & Harmful & $\bar\Delta_{\text{harm}}$ & $Y|X$ share & $|\pi_T-\pi_S|$ \\",
             r"\midrule"]
    for ds, r in dec.iterrows():
        lines.append(f"{DS_LABEL[ds]} & {int(r.n_pairs)} & {100*r.frac_harm:.0f}\\% & {100*r.mean_delta_harm:.1f} & "
                     f"{100*r.yx_share:.0f}\\% & {r.prior_gap:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e3_decomposition.tex"), "w").write("\n".join(lines))


def tex_e4(df, mae):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Method & $\rho$ & AUROC & AUPRC & FPR@90 \\", r"\midrule"]
    order = ["Domain-clf. AUC", "ATC drop", "Importance-wtd. drop", "IWS-KS (ours)",
             "Data-only signals", "+ IWS", "All signals (harm score)"]
    df = df.set_index("method").loc[order]
    for k, r in df.iterrows():
        name = {"All signals (harm score)": "All signals (harm score)$^\\dagger$",
                "IWS-KS (ours)": "IWS-KS$^\\dagger$", "+ IWS": "Data-only + IWS"}.get(k, k)
        lines.append(f"{name} & {r.rho_pooled:.2f} & {r.auroc_pooled:.2f} & {r.auprc_pooled:.2f} & "
                     f"{r.fpr_at_90recall:.2f} \\\\")
        if k == "IWS-KS (ours)":
            lines.append(r"\midrule \multicolumn{5}{l}{\emph{Meta-regressor, leave-one-dataset-out}} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "e4_harm_score.tex"), "w").write("\n".join(lines))


def main():
    nat, syn = load()
    e1 = e1_correlation(nat)
    tex_e1(e1)
    tests = e1_tests(nat)
    perds = e1_per_dataset(nat)
    e2 = e2_synthetic(syn)
    dec, s2 = e3_decomposition(nat, syn)
    tex_e3(dec)
    e4, preds, mae = e4_harm_score(nat)
    tex_e4(e4, mae)
    by_model, by_tau = e5_ablations(nat)
    nat2, tax = taxonomy(nat)
    tax_ds = tex_taxonomy(nat2)
    fig_scatter(nat)
    fig_synthetic(syn)
    fig_decomposition(nat)
    fig_pr(nat, preds)
    summary = dict(
        n_rows=len(nat), n_pairs=int(nat.groupby(["dataset", "attr", "source", "target"]).ngroups),
        n_seeds=int(nat.seed.nunique()), harm_rate=float((nat.delta > TAU).mean()),
        delta_mean=float(nat.delta.mean()), delta_max=float(nat.delta.max()),
        frac_negative=float((nat.delta < 0).mean()),
        high_drift_rate=float((nat.dc_auc >= 0.75).mean()),
        harm_given_high=float((nat[nat.dc_auc >= 0.75].delta > TAU).mean()),
        harm_given_low=float((nat[nat.dc_auc < 0.75].delta > TAU).mean()),
        regimes=nat2.regime.value_counts().to_dict(), mae=mae,
        yx_dominant_overall=float((nat[nat.delta > TAU].yx_part.abs() > nat[nat.delta > TAU].cov_part.abs()).mean()),
        synthetic_check=s2.round(4).to_dict(),
        frac_saturated=float((nat.dc_auc >= 0.95).mean()),
        harm_given_saturated=float((nat[nat.dc_auc >= 0.95].delta > TAU).mean()),
        rho_iwsks_unsaturated=float(sp(nat[nat.dc_auc < 0.95].iws_ks, nat[nat.dc_auc < 0.95].delta)),
        rho_dcauc_unsaturated=float(sp(nat[nat.dc_auc < 0.95].dc_auc, nat[nat.dc_auc < 0.95].delta)),
        rho_iwsks_saturated=float(sp(nat[nat.dc_auc >= 0.95].iws_ks, nat[nat.dc_auc >= 0.95].delta)),
        rho_dcauc_saturated=float(sp(nat[nat.dc_auc >= 0.95].dc_auc, nat[nat.dc_auc >= 0.95].delta)),
        decomposition_overlap=json.load(open(os.path.join(TAB, "e3_overlap.json"))),
    )
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=2, default=float)
    pd.set_option("display.width", 200)
    print(e1.round(3).to_string())
    print(tests.round(4).to_string())
    print(perds.round(3).to_string())
    print(e2.round(3).to_string())
    print(dec.round(3).to_string())
    print(s2.round(4))
    print(e4.round(3).to_string())
    print(by_model.round(3))
    print(by_tau.round(3).to_string())
    print(tax)
    print(tax_ds.round(1))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
