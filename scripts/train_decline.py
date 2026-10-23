"""Cognitive decline as the outcome: classification and progression regression.

The main analysis predicts a dementia diagnosis. The project title promises
early prediction of cognitive *decline*, which also covers people who worsen
without reaching dementia. Two decline outcomes are built from CDR-SB, which is
recorded at every visit in every form version:

  classification  CDR-SB rises by 1 point or more at any visit within 3 years
                  (y_cdrsb_rise_3y, built in build_dataset.py). Unknown if the
                  person was neither seen to rise nor followed for 3 years.
  regression      change in CDR-SB from the index visit to the visit closest to
                  3 years later (within 6 months either side)

Same cohort rules (first visit, not demented, in person, form versions 1-3),
features (the 48 used everywhere), splits and model family (XGBoost, three
seeds) as train_tabular.py. Each result is reported for all participants and
separately for cognitively normal and MCI participants.

Output  reports/phase3b_decline_results.json   (aggregate only)
Usage:  python scripts/train_decline.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor

import modeling as M
import nacc_utils as nu

CLS = "y_cdrsb_rise_3y"
SEEDS = [0, 1, 2]
GROUPS = {"All non-demented": [1, 2, 3], "Cognitively normal": [1], "MCI": [3]}
STAGE = M.FEATURE_GROUPS["Clinical stage (CDR-SB, diagnosis)"]
COG = M.FEATURE_GROUPS["Cognitive tests"]
CARDIO = M.FEATURE_GROUPS["Cardiovascular / medical"]


def log(msg):
    print(f"[decline] {msg}", flush=True)


def xgb_cls(df, features, target):
    d = M.split_xy(df, features, target)
    ps = []
    for seed in SEEDS:
        predict = M.fit_xgboost(d, seed=seed, depths=(4,))
        ps.append({s: predict(d[s][0]) for s in ["test", "external"]})
    return {s: np.mean([p[s] for p in ps], axis=0) for s in ["test", "external"]}


def by_group(df, preds, target, ref=None):
    out = {}
    for split in ["test", "external"]:
        part = df[df.split == split]
        y, dx = part[target].to_numpy(int), part.NACCUDSD.to_numpy()
        out[split] = {g: M.bootstrap(y[np.isin(dx, c)], preds[split][np.isin(dx, c)], n_boot=500,
                                     p_ref=None if ref is None else ref[split][np.isin(dx, c)]) for g, c in GROUPS.items()}
    return out


def three_year_change(visits: pd.DataFrame) -> pd.Series:
    """CDR-SB at the visit closest to 3 years after the first visit, minus CDR-SB at the first visit."""
    v = visits[["NACCID", "years", "CDRSUM"]].dropna()
    base = v[v.years == 0].drop_duplicates("NACCID").set_index("NACCID").CDRSUM
    near = v[(v.years - 3).abs() <= 0.5].assign(dist=lambda d: (d.years - 3).abs())
    near = near.sort_values("dist").drop_duplicates("NACCID").set_index("NACCID").CDRSUM
    return (near - base.reindex(near.index)).rename("cdrsb_change_3y")


def regression(df):
    d = M.split_xy(df, M.CORE, "cdrsb_change_3y")
    d = {s: (x, df.loc[df.split == s, "cdrsb_change_3y"].to_numpy(float)) for s, (x, _) in d.items()}
    preds = []
    for seed in SEEDS:
        m = XGBRegressor(n_estimators=3000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, reg_lambda=2.0, objective="reg:absoluteerror", early_stopping_rounds=100,
                         tree_method="hist", random_state=seed, n_jobs=-1)
        m.fit(d["train"][0], d["train"][1], eval_set=[(d["val"][0], d["val"][1])], verbose=False)
        preds.append({s: m.predict(d[s][0]) for s in ["test", "external"]})
    p = {s: np.mean([q[s] for q in preds], axis=0) for s in ["test", "external"]}
    train_median = float(np.median(d["train"][1]))
    out = {}
    for split in ["test", "external"]:
        part = df[df.split == split]
        y, dx = d[split][1], part.NACCUDSD.to_numpy()
        out[split] = {}
        for g, codes in GROUPS.items():
            m_ = np.isin(dx, codes)
            yy, pp = y[m_], p[split][m_]
            out[split][g] = {
                "n": int(m_.sum()), "mean_change": float(yy.mean()), "share_no_change_or_better": float((yy <= 0).mean()),
                "mae_model": float(mean_absolute_error(yy, pp)), "mae_predict_no_change": float(mean_absolute_error(yy, np.zeros_like(yy))),
                "mae_predict_train_median": float(mean_absolute_error(yy, np.full_like(yy, train_median))),
                "rmse_model": float(np.sqrt(mean_squared_error(yy, pp))), "rmse_predict_no_change": float(np.sqrt(np.mean(yy ** 2))),
                "r2_model": float(r2_score(yy, pp)), "spearman": float(spearmanr(yy, pp).correlation)}
    return out


def main():
    visits = M.load_visits()
    first = visits[(visits.NACCVNUM == 1) & visits.eligible].reset_index(drop=True)
    results = {}

    # ---- classification: CDR-SB rise of 1 or more within 3 years ---------------
    c = first[first[CLS].notna()].reset_index(drop=True)
    results["classification"] = {"target": "CDR-SB rises by >= 1 within 3 years",
                                 "cohort": {g: {"n": int(c.NACCUDSD.isin(k).sum()), "events": int(c.loc[c.NACCUDSD.isin(k), CLS].sum())}
                                            for g, k in GROUPS.items()}}
    full = xgb_cls(c, M.CORE, CLS)
    results["classification"]["xgboost"] = by_group(c, full, CLS)
    results["classification"]["gain_from"] = {}
    for name, feats in [("Cardiovascular / medical", CARDIO), ("Clinical stage (CDR-SB, diagnosis)", STAGE),
                        ("Cognitive tests", COG), ("Clinical stage and cognitive tests", STAGE + COG)]:
        ref = xgb_cls(c, [f for f in M.CORE if f not in feats], CLS)
        results["classification"]["gain_from"][name] = by_group(c, full, CLS, ref=ref)
    lr = {}
    d = M.split_xy(c, M.CORE, CLS)
    predict = M.fit_logistic(d)
    lr = {s: predict(d[s][0]) for s in ["test", "external"]}
    results["classification"]["logistic_regression"] = by_group(c, lr, CLS)
    for s in ["test", "external"]:
        r = results["classification"]["xgboost"][s]
        log(f"decline {s}: " + ", ".join(f"{g} {v['auroc']:.3f} ({v['n']}/{v['events']})" for g, v in r.items()))
    for name, r in results["classification"]["gain_from"].items():
        log(f"gain from {name}: " + ", ".join(f"{g} {v['delta_auroc']:+.4f}" for g, v in r["test"].items()))

    # ---- regression: 3-year change in CDR-SB ----------------------------------
    change = three_year_change(visits)
    r = first.join(change, on="NACCID")
    r = r[r.cdrsb_change_3y.notna()].reset_index(drop=True)
    results["regression"] = {"target": "change in CDR-SB from the index visit to the visit closest to 3 years later (+/- 6 months)",
                             "cohort": {s: int((r.split == s).sum()) for s in M.SPLITS}, "results": regression(r)}
    for s in ["test", "external"]:
        a = results["regression"]["results"][s]["All non-demented"]
        log(f"regression {s}: n={a['n']} MAE {a['mae_model']:.3f} vs no-change {a['mae_predict_no_change']:.3f}; "
            f"RMSE {a['rmse_model']:.3f} vs {a['rmse_predict_no_change']:.3f}; R2 {a['r2_model']:.3f}; Spearman {a['spearman']:.3f}")

    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "decline results")
    (nu.REPORTS / "phase3b_decline_results.json").write_text(text)


if __name__ == "__main__":
    main()
