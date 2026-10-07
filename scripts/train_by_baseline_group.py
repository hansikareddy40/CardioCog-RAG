"""Phase 3c: the same prediction, reported separately by diagnosis at baseline.

"Not demented at baseline" mixes cognitively normal people with people who
already have MCI, and the two groups have very different risk. This script
reports every result three ways: all non-demented, cognitively normal only,
and MCI only. It also measures how much the clinical stage variables (CDR-SB
and the current diagnosis) and the cardiovascular group add inside each group.

Cohort   participants not demented at their first visit, 3-year status known
Target   dementia diagnosis within 3 years
Output   reports/phase3_by_baseline_group.json   (aggregate metrics only)

Usage:  python scripts/train_by_baseline_group.py
"""

from __future__ import annotations

import json
import time

import numpy as np

import modeling as M
import nacc_utils as nu

TARGET = "y_dementia_3y"
SEEDS = [0, 1, 2]
# NACCUDSD at the index visit: 1 normal, 2 impaired-not-MCI, 3 MCI
GROUPS = {"All non-demented": [1, 2, 3], "Cognitively normal": [1], "Impaired, not MCI": [2], "MCI": [3]}
STAGE = M.FEATURE_GROUPS["Clinical stage (CDR-SB, diagnosis)"]
COGNITIVE = M.FEATURE_GROUPS["Cognitive tests"]
CARDIO = M.FEATURE_GROUPS["Cardiovascular / medical"]


def log(msg):
    print(f"[by group] {msg}", flush=True)


def xgb_preds(df, features):
    """Mean prediction of three seeds on the test split and the held-out centres."""
    d = M.split_xy(df, features, TARGET)
    ps = []
    for seed in SEEDS:
        predict = M.fit_xgboost(d, seed=seed, depths=(4,))
        ps.append({s: predict(d[s][0]) for s in ["test", "external"]})
    return {s: np.mean([p[s] for p in ps], axis=0) for s in ["test", "external"]}


def evaluate(df, preds, ref=None, groups=GROUPS) -> dict:
    """Score in each baseline group. With ref, also the paired AUROC gain over ref."""
    out = {}
    for split in ["test", "external"]:
        d = df[df.split == split]
        y, dx = d[TARGET].to_numpy(int), d.NACCUDSD.to_numpy()
        out[split] = {}
        for name, codes in groups.items():
            m = np.isin(dx, codes)
            out[split][name] = M.bootstrap(y[m], preds[split][m], p_ref=None if ref is None else ref[split][m])
    return out


def main():
    t0 = time.time()
    df = M.load_visits()
    df = df[(df.NACCVNUM == 1) & df.eligible & df[TARGET].notna()].reset_index(drop=True)
    results = {"target": TARGET, "cohort": {}}
    for name, codes in GROUPS.items():
        g = df[df.NACCUDSD.isin(codes)]
        results["cohort"][name] = {s: {"n": int((g.split == s).sum()), "events": int(g.loc[g.split == s, TARGET].sum())}
                                   for s in M.SPLITS}
        results["cohort"][name]["total"] = {"n": int(len(g)), "events": int(g[TARGET].sum()),
                                            "event_rate": float(g[TARGET].mean())}
        log(f"{name}: {len(g):,} participants, {int(g[TARGET].sum()):,} events ({g[TARGET].mean():.1%})")

    # ---- 1. One model trained on everyone, scored inside each group ---------
    full = xgb_preds(df, M.CORE)
    results["trained_on_all"] = {"Full model": evaluate(df, full)}
    variants = {
        "Clinical stage only (CDR-SB, diagnosis)": STAGE,
        "CDR-SB only": ["CDRSUM"],
        "Diagnosis only": ["is_mci", "is_impaired_not_mci"],
        "Cognitive tests only": COGNITIVE,
        "Everything except clinical stage": [f for f in M.CORE if f not in STAGE],
        "Everything except clinical stage and cognitive tests": [f for f in M.CORE if f not in STAGE + COGNITIVE],
    }
    for name, feats in variants.items():
        results["trained_on_all"][name] = evaluate(df, xgb_preds(df, feats))
        log(f"{name}: " + ", ".join(f"{g} {r['auroc']:.3f}" for g, r in results["trained_on_all"][name]["test"].items()))

    # Paired gain: full model minus the model without the group, inside each baseline group
    results["gain_from_adding"] = {}
    for name, feats in [("Clinical stage (CDR-SB, diagnosis)", STAGE), ("Cognitive tests", COGNITIVE),
                        ("Clinical stage and cognitive tests", STAGE + COGNITIVE), ("Cardiovascular / medical", CARDIO)]:
        without = xgb_preds(df, [f for f in M.CORE if f not in feats])
        results["gain_from_adding"][name] = evaluate(df, full, ref=without)
        log(f"gain from {name}: " + ", ".join(f"{g} {r['delta_auroc']:+.4f}"
                                             for g, r in results["gain_from_adding"][name]["test"].items()))

    # ---- 2. Models trained inside one group only ----------------------------
    # Within a group the diagnosis flags are constant, so only CDR-SB can carry
    # clinical-stage information.
    results["trained_within_group"] = {}
    for name in ["Cognitively normal", "MCI"]:
        g = df[df.NACCUDSD.isin(GROUPS[name])].reset_index(drop=True)
        own = {name: GROUPS[name]}
        full_g = xgb_preds(g, M.CORE)
        r = {"Full model": evaluate(g, full_g, groups=own)}
        r["Gain from CDR-SB"] = evaluate(g, full_g, ref=xgb_preds(g, [f for f in M.CORE if f not in STAGE]), groups=own)
        r["Gain from cardiovascular / medical"] = evaluate(
            g, full_g, ref=xgb_preds(g, [f for f in M.CORE if f not in CARDIO]), groups=own)
        results["trained_within_group"][name] = {k: {s: v[s][name] for s in v} for k, v in r.items()}
        log(f"trained on {name} only: test AUROC {r['Full model']['test'][name]['auroc']:.3f}, "
            f"external {r['Full model']['external'][name]['auroc']:.3f}")

    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "by-group results")
    (nu.REPORTS / "phase3_by_baseline_group.json").write_text(text)
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
