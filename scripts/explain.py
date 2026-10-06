"""Phase 7: SHAP explanations for the tabular model.

SHAP splits each individual prediction into a contribution per feature. This
script summarises those contributions over the test participants:

  * importance per feature and per feature group (mean absolute SHAP)
  * how stable the ranking is when the model is retrained on resampled data
  * for the leading features, the average contribution at each feature level

Only aggregates are saved. Plots with one dot per participant are avoided on
purpose (data use agreement).

Output  reports/phase7_shap_results.json
Usage:  python scripts/explain.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

import modeling as M
import nacc_utils as nu

TARGET = "y_dementia_3y"
N_REFITS = 8
GROUP_OF = {f: g for g, feats in M.FEATURE_GROUPS.items() for f in feats}


def shap_values(d, seed, rows=None):
    train = d["train"] if rows is None else (d["train"][0][rows], d["train"][1][rows])
    predict = M.fit_xgboost({**d, "train": train}, seed=seed, depths=(4,))
    return shap.TreeExplainer(predict.model).shap_values(d["test"][0])        # log-odds units


def main():
    df = M.load_visits()
    df = df[(df.NACCVNUM == 1) & df.eligible & df[TARGET].notna()].reset_index(drop=True)
    d = M.split_xy(df, M.CORE, TARGET)
    Xte = pd.DataFrame(d["test"][0], columns=M.CORE)
    dx = df.loc[df.split == "test", "NACCUDSD"].to_numpy()

    sv = shap_values(d, seed=0)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=M.CORE).sort_values(ascending=False)
    results = {
        "n_test": int(len(Xte)), "units": "mean absolute SHAP value, log-odds",
        "feature_importance": imp.round(5).to_dict(),
        "group_importance": imp.groupby(GROUP_OF).sum().sort_values(ascending=False).round(5).to_dict(),
        "group_importance_by_index_dx": {
            name: pd.Series(np.abs(sv[dx == code]).mean(axis=0), index=M.CORE).groupby(GROUP_OF).sum().round(5).to_dict()
            for name, code in [("Normal at index", 1), ("MCI at index", 3)]},
    }

    # Stability: retrain on bootstrap resamples of the training participants.
    rng = np.random.default_rng(0)
    n = len(d["train"][1])
    refits = [pd.Series(np.abs(shap_values(d, seed=s + 1, rows=rng.integers(0, n, n))).mean(axis=0), index=M.CORE) for s in range(N_REFITS)]
    ranks = pd.DataFrame([r.rank(ascending=False) for r in refits])
    top10 = set(imp.index[:10])
    results["stability"] = {
        "refits": N_REFITS,
        "spearman_with_main_model": [float(spearmanr(imp, r[imp.index]).correlation) for r in refits],
        "top10_overlap_with_main_model": [len(top10 & set(r.sort_values(ascending=False).index[:10])) for r in refits],
        "rank_range_top15": {f: [int(ranks[f].min()), int(ranks[f].max())] for f in imp.index[:15]},
        "group_share_range": pd.DataFrame([r.groupby(GROUP_OF).sum() / r.sum() for r in refits]).agg(["min", "max"]).round(4).to_dict(),
    }

    # Direction of effect: mean SHAP within bins of the feature value.
    dep = {}
    for f in imp.index[:9]:
        x, s = Xte[f], pd.Series(sv[:, M.CORE.index(f)])
        if x.nunique() <= 5:
            b = x
        else:
            b = pd.qcut(x, 8, duplicates="drop")
        t = pd.DataFrame({"x": x, "s": s}).groupby(b, observed=True).agg(n=("s", "size"), value=("x", "mean"), shap=("s", "mean"))
        t = t[t.n >= 20]
        dep[f] = {"value": t.value.round(3).tolist(), "mean_shap": t.shap.round(4).tolist(), "n": t.n.tolist(),
                  "missing_n": int(x.isna().sum()), "missing_mean_shap": float(s[x.isna().to_numpy()].mean()) if x.isna().any() else None}
    results["dependence"] = dep

    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "SHAP results")
    (nu.REPORTS / "phase7_shap_results.json").write_text(text)
    print(json.dumps({k: results[k] for k in ["group_importance", "stability"]}, indent=1)[:2500])
    print(imp.head(15).round(4).to_string())


if __name__ == "__main__":
    main()
