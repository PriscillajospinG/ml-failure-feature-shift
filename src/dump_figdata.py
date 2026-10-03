"""Dump the numbers behind every figure to results/figdata.json (used by scripts/make_tikz.py)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, analysis as A, case_study as C
nat, syn = A.load()
e4, preds, mae = A.e4_harm_score(nat)
out = {}
out['scatter'] = {s: {ds: [[float(a), float(100*b)] for a,b in zip(nat[nat.dataset==ds][s], nat[nat.dataset==ds].delta)] for ds in A.DS_ORDER} for s in ["dc_auc","iws_ks","iw_drop"]}
out['rho'] = {s: float(A.sp(nat[s], nat.delta)) for s in ["dc_auc","iws_ks","iw_drop"]}
g = syn.groupby(["attr","magnitude"]).mean(numeric_only=True).reset_index()
g["dc_auc_n"] = 2*(g.dc_auc-0.5)
out['syn'] = {f: {c: [[float(m), float(v)] for m,v in zip(g[g.attr==f].magnitude, g[g.attr==f][c])] for c in ["delta","dc_auc_n","iws_ks","iw_drop"]} for f in ["irrelevant","important","spread","concept","prior"]}
h = nat[nat.delta > A.TAU]
agg = h.groupby("dataset")[["cov_part","yx_part","delta"]].mean().reindex(A.DS_ORDER).dropna()
out['decomp'] = {d: [float(100*agg.loc[d,c]) for c in ["cov_part","yx_part","delta"]] for d in agg.index}
from sklearn.metrics import precision_recall_curve, average_precision_score
harm = (nat.delta > A.TAU).astype(int).values
out['pr'] = {}
for k in ["IWS-KS (ours)","All signals (harm score)","Data-only signals","Domain-clf. AUC","Importance-wtd. drop"]:
    p,r,_ = precision_recall_curve(harm, preds[k])
    idx = np.unique(np.linspace(0,len(p)-1,60).astype(int))
    out['pr'][k] = dict(ap=float(average_precision_score(harm,preds[k])), pts=[[float(r[i]),float(p[i])] for i in idx])
out['base'] = float(harm.mean())
out['case'] = []
for name, attr, s, t, title in C.CASES:
    full, a, b = C.run_case(name, attr, s, t)
    df = full.sort_values("ks", ascending=False).head(6).iloc[::-1]
    out['case'].append(dict(title=title, acc=[float(a),float(b)], rows=[[r.feature, float(r.ks), float(r.importance)] for r in df.itertuples()]))
json.dump(out, open(os.path.join(A.RES, 'figdata.json'), 'w'))
print('ok', {k: (len(v) if hasattr(v,'__len__') else v) for k,v in out.items()})
