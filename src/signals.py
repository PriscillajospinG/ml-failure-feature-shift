"""Label-free shift signals.

Every function here uses only: labelled SOURCE data (train/validation), the trained
model, and UNLABELLED target features. Target labels are never touched.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------- per-feature drift
def per_feature_drift(Xs: np.ndarray, Xt: np.ndarray, n_bins: int = 10):
    """Return per-feature KS statistic, standardized Wasserstein distance and PSI."""
    p = Xs.shape[1]
    ks = np.zeros(p)
    wd = np.zeros(p)
    psi = np.zeros(p)
    sd = Xs.std(0)
    sd[sd == 0] = 1.0
    for j in range(p):
        a, b = Xs[:, j], Xt[:, j]
        if np.all(a == a[0]) and np.all(b == a[0]):
            continue
        ks[j] = ks_2samp(a, b).statistic
        wd[j] = wasserstein_distance(a / sd[j], b / sd[j])
        edges = np.unique(np.quantile(a, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([a.min() - 1e-9, a.max() + 1e-9])
        edges[0], edges[-1] = -np.inf, np.inf
        pa = np.histogram(a, edges)[0] / len(a) + 1e-4
        pb = np.histogram(b, edges)[0] / len(b) + 1e-4
        psi[j] = np.sum((pb - pa) * np.log(pb / pa))
    return ks, wd, psi


def mmd_rbf(Xs: np.ndarray, Xt: np.ndarray, n: int = 1000, rng=None) -> float:
    """Unbiased MMD^2 with an RBF kernel, median-heuristic bandwidth, on standardized data."""
    rng = rng or np.random.default_rng(0)
    a = Xs[rng.choice(len(Xs), min(n, len(Xs)), replace=False)]
    b = Xt[rng.choice(len(Xt), min(n, len(Xt)), replace=False)]
    mu, sd = Xs.mean(0), Xs.std(0)
    sd[sd == 0] = 1.0
    a, b = (a - mu) / sd, (b - mu) / sd
    z = np.vstack([a, b])
    sq = np.sum(z ** 2, 1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * z @ z.T, 0)
    med = np.median(d2[np.triu_indices_from(d2, 1)])
    k = np.exp(-d2 / (med + 1e-12))
    m = len(a)
    kxx, kyy, kxy = k[:m, :m], k[m:, m:], k[:m, m:]
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (len(b) * (len(b) - 1)) - 2 * kxy.mean())


# ---------------------------------------------------------------- domain classifier
def domain_classifier(Xs_val: np.ndarray, Xt: np.ndarray, n: int = 4000, seed: int = 0):
    """Classifier two-sample test (C2ST).

    Returns (AUC, density-ratio weights for every row of Xs_val).
    Weights w(x) = p(T|x)/p(S|x) * n_S/n_T are out-of-fold for the source rows,
    so they can be used for importance-weighted accuracy without overfitting.
    """
    rng = np.random.default_rng(seed)
    it = rng.choice(len(Xt), min(n, len(Xt)), replace=False)
    Z = np.vstack([Xs_val, Xt[it]])
    d = np.r_[np.zeros(len(Xs_val)), np.ones(len(it))]
    prob = np.zeros(len(Z))
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    for tr, te in skf.split(Z, d):
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
                                             max_leaf_nodes=15, random_state=seed)
        clf.fit(Z[tr], d[tr])
        prob[te] = clf.predict_proba(Z[te])[:, 1]
    auc = roc_auc_score(d, prob)
    ps = np.clip(prob[: len(Xs_val)], 0.01, 0.99)
    w = ps / (1 - ps) * (len(Xs_val) / len(it))
    w = np.minimum(w, 50.0)
    return float(max(auc, 0.5)), w


# ---------------------------------------------------------------- model-aware signals
def feature_importance(model, Xv: np.ndarray, yv: np.ndarray, n: int = 2000, seed: int = 0):
    """Permutation importance (accuracy drop) on SOURCE validation data, clipped at 0
    and normalized to sum to 1. Model-agnostic, uses source labels only."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xv), min(n, len(Xv)), replace=False)
    r = permutation_importance(model, Xv[idx], yv[idx], n_repeats=3, random_state=seed,
                               scoring="accuracy", n_jobs=1)
    imp = np.clip(r.importances_mean, 0, None)
    if imp.sum() <= 0:
        imp = np.ones_like(imp)
    return imp / imp.sum()


def atc_estimate(conf_s: np.ndarray, correct_s: np.ndarray, conf_t: np.ndarray) -> float:
    """Average Thresholded Confidence (Garg et al., ICLR 2022), max-softmax score.
    Threshold chosen on source validation so that P(conf > t) = source accuracy."""
    acc = correct_s.mean()
    t = np.quantile(conf_s, 1 - acc)
    return float((conf_t > t).mean())


def model_free_signals(Xs_val, Xt, seed=0):
    """Signals that depend only on the data (computed once per source/target pair)."""
    out = {}
    ks, wd, psi = per_feature_drift(Xs_val, Xt)
    out["ks_mean"] = ks.mean()
    out["ks_max"] = ks.max()
    out["wass_mean"] = wd.mean()
    out["psi_mean"] = psi.mean()
    out["mmd"] = mmd_rbf(Xs_val, Xt, rng=np.random.default_rng(seed))
    auc, w = domain_classifier(Xs_val, Xt, seed=seed)
    out["dc_auc"] = auc
    return out, ks, wd, w


def compute_signals(model, ref_model, Xs_val, ys_val, Xt, importance, seed=0, cache=None):
    """All label-free signals for one (model, source, target) triple.
    `cache` = output of model_free_signals for this pair (recomputed if None)."""
    base, ks, wd, w = cache if cache is not None else model_free_signals(Xs_val, Xt, seed)
    out = dict(base)
    # importance-weighted shift (proposed)
    out["iws_wass"] = float(np.sum(importance * wd))
    out["iws_ks"] = float(np.sum(importance * ks))
    # model-output signals
    ps = model.predict_proba(Xs_val)
    pt = model.predict_proba(Xt)
    cs, ct = ps.max(1), pt.max(1)
    correct_s = (ps.argmax(1) == ys_val).astype(float)
    acc_s = correct_s.mean()
    out["conf_drop"] = cs.mean() - ct.mean()  # = DoC (Guillory et al., 2021)
    out["pred_shift"] = abs(ps[:, 1].mean() - pt[:, 1].mean())
    out["atc_drop"] = acc_s - atc_estimate(cs, correct_s, ct)
    acc_iw = float(np.sum(w * correct_s) / np.sum(w))
    out["iw_drop"] = acc_s - acc_iw
    dis_s = (ref_model.predict(Xs_val) != ps.argmax(1)).mean()
    dis_t = (ref_model.predict(Xt) != pt.argmax(1)).mean()
    out["disagree_inc"] = dis_t - dis_s
    out["acc_iw"] = acc_iw
    return out, ks, wd


SIGNALS = ["ks_mean", "ks_max", "wass_mean", "psi_mean", "mmd", "dc_auc",
           "iws_wass", "iws_ks", "conf_drop", "pred_shift", "atc_drop", "iw_drop",
           "disagree_inc"]

SIGNAL_NAMES = {
    "ks_mean": "KS (mean)", "ks_max": "KS (max)", "wass_mean": "Wasserstein (mean)",
    "psi_mean": "PSI (mean)", "mmd": "MMD$^2$", "dc_auc": "Domain-clf. AUC",
    "iws_wass": "IWS-Wass. (ours)", "iws_ks": "IWS-KS (ours)",
    "conf_drop": "Confidence drop (DoC)", "pred_shift": "Prediction shift",
    "atc_drop": "ATC drop", "iw_drop": "Importance-wtd. drop", "disagree_inc": "Disagreement incr.",
}
