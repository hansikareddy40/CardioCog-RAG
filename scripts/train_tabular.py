"""Phase 3a/3b: tabular models and the feature-group experiment.

Cohort   participants not demented at their first visit, 3-year status known
Target   dementia diagnosis within 3 years
Output   reports/phase3_tabular_results.json   (aggregate metrics only)
         data/processed/preds_tabular.parquet  (per-participant predictions, gitignored)

Usage:  python scripts/train_tabular.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import modeling as M
import nacc_utils as nu

TARGET = "y_dementia_3y"
SEEDS = [0, 1, 2]


def log(msg):
    print(f"[tabular] {msg}", flush=True)


def evaluate(df, preds: dict, ref: dict | None = None) -> dict:
    """Score predictions on the test split and the held-out centres, overall
    and separately by diagnosis at the index visit."""
    out = {}
    for split in ["test", "external"]:
        d = df[df.split == split]
        y, p = d[TARGET].to_numpy(int), preds[split]
        r = M.bootstrap(y, p, p_ref=None if ref is None else ref[split])
        for name, mask in [("normal_at_index", d.NACCUDSD.to_numpy() == 1), ("mci_at_index", d.NACCUDSD.to_numpy() == 3)]:
            r[name] = M.bootstrap(y[mask], p[mask], n_boot=500)
        out[split] = r
    return out


def main():
    t0 = time.time()
    df = M.load_visits()
    df = df[(df.NACCVNUM == 1) & df.eligible & df[TARGET].notna()].reset_index(drop=True)
    log(f"cohort: {len(df):,} participants, {int(df[TARGET].sum()):,} events")
    results = {"target": TARGET, "cohort": {s: {"n": int((df.split == s).sum()), "events": int(df.loc[df.split == s, TARGET].sum())} for s in M.SPLITS}}
    keep = df[["NACCID", "split", TARGET, "NACCUDSD"]].copy()

    # ---- 1. Model comparison on the core feature set ----------------------
    d = M.split_xy(df, M.CORE, TARGET)
    results["models"] = {}
    for name, fit in M.MODELS.items():
        seeds = SEEDS if name not in ("Logistic regression",) else [0]
        per_seed = []
        for seed in seeds:
            predict = fit(d, seed=seed)
            per_seed.append({s: predict(d[s][0]) for s in ["val", "test", "external"]})
        mean_pred = {s: np.mean([p[s] for p in per_seed], axis=0) for s in ["val", "test", "external"]}
        res = evaluate(df, mean_pred)
        res["seeds"] = len(seeds)
        res["test_auroc_per_seed"] = [float(M.roc_auc_score(d["test"][1], p["test"])) for p in per_seed]
        results["models"][name] = res
        for s in ["test", "external"]:
            keep.loc[keep.split == s, f"p_{name}"] = mean_pred[s]
        log(f"{name}: test AUROC {res['test']['auroc']:.3f} AUPRC {res['test']['auprc']:.3f} | external AUROC {res['external']['auroc']:.3f}  ({time.time()-t0:.0f}s)")

    # ---- 2. Feature-group experiment with XGBoost --------------------------
    def xgb_preds(features):
        dd = M.split_xy(df, features, TARGET)
        ps = []
        for seed in SEEDS:
            predict = M.fit_xgboost(dd, seed=seed, depths=(4,))
            ps.append({s: predict(dd[s][0]) for s in ["test", "external"]})
        return {s: np.mean([p[s] for p in ps], axis=0) for s in ["test", "external"]}

    full = xgb_preds(M.CORE)
    results["groups_alone"], results["groups_cumulative"], results["groups_removed"] = {}, {}, {}
    used = []
    for group, feats in M.FEATURE_GROUPS.items():
        results["groups_alone"][group] = evaluate(df, xgb_preds(feats))
        used = used + feats
        results["groups_cumulative"][group] = evaluate(df, xgb_preds(used))
        without = [f for f in M.CORE if f not in feats]
        # paired difference: full model minus the model without this group
        results["groups_removed"][group] = evaluate(df, full, ref=xgb_preds(without))
        log(f"group {group}: alone {results['groups_alone'][group]['test']['auroc']:.3f}, "
            f"cumulative {results['groups_cumulative'][group]['test']['auroc']:.3f}, "
            f"gain from adding to the rest {results['groups_removed'][group]['test']['delta_auroc']:+.4f}")

    # A sharper version of research question 1: cardiovascular features on top
    # of demographics + APOE only (before any cognitive information).
    early = M.FEATURE_GROUPS["Demographics"] + M.FEATURE_GROUPS["Genetics / APOE"]
    results["cardio_over_demographics_apoe"] = evaluate(df, xgb_preds(early + M.FEATURE_GROUPS["Cardiovascular / medical"]), ref=xgb_preds(early))

    # ---- 3. With the close-to-diagnosis variables --------------------------
    results["with_proximal"] = evaluate(df, xgb_preds(M.CORE + M.PROXIMAL), ref=full)

    # ---- 4. Calibration and the "no model" reference ------------------------
    for s in ["test", "external"]:
        y = df.loc[df.split == s, TARGET].to_numpy(int)
        results.setdefault("calibration_xgboost", {})[s] = M.calibration_table(y, full[s])
    rate = df.loc[df.split == "train", TARGET].mean()
    results["reference"] = {"train_event_rate": float(rate),
                            "brier_predicting_base_rate_test": float(np.mean((df.loc[df.split == "test", TARGET] - rate) ** 2))}

    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "tabular results")
    (nu.REPORTS / "phase3_tabular_results.json").write_text(text)
    keep.to_parquet(M.PROCESSED / "preds_tabular.parquet", index=False)
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
