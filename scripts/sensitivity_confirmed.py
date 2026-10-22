"""Sensitivity analysis: does the result hold if a dementia diagnosis must be confirmed?

A clinician's diagnosis can be reversed at a later visit (notebook 02 shows
people moving back from MCI to normal, and some from dementia). The main
label counts the first dementia diagnosis within 3 years. Here a conversion
counts only if it was never reversed: no later visit records the person as
not demented.

  confirmed    dementia within 3 years, and every later visit is still dementia
               (or there is no later visit to contradict it)
  reverted     dementia within 3 years, but a later visit records normal,
               impaired-not-MCI or MCI. Left out of this analysis.

Same cohort, features, splits and model (XGBoost, three seeds) as train_tabular.py.

Output  reports/sensitivity_confirmed_conversion.json   (aggregate only)
Usage:  python scripts/sensitivity_confirmed.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import modeling as M
import nacc_utils as nu

TARGET = "y_dementia_3y"
SEEDS = [0, 1, 2]
GROUPS = {"All non-demented": [1, 2, 3], "Cognitively normal": [1], "MCI": [3]}


def main():
    visits = M.load_visits()
    first = visits[(visits.NACCVNUM == 1) & visits.eligible & visits[TARGET].notna()].reset_index(drop=True)

    # For each converter: was any visit after the first dementia diagnosis recorded as not demented?
    v = visits[["NACCID", "years", "NACCUDSD"]].dropna(subset=["NACCUDSD"])
    first_dem = v[v.NACCUDSD == 4].groupby("NACCID").years.min().rename("t_dem")
    later = v.join(first_dem, on="NACCID").dropna(subset=["t_dem"])
    later = later[later.years > later.t_dem]
    reverted_ids = set(later.loc[later.NACCUDSD.isin([1, 2, 3]), "NACCID"])
    confirmed_by_later_visit = set(later.loc[later.NACCUDSD == 4, "NACCID"])

    conv = first[first[TARGET] == 1]
    counts = {"converters": int(len(conv)),
              "reverted_later": int(conv.NACCID.isin(reverted_ids).sum()),
              "confirmed_by_a_later_dementia_visit": int((conv.NACCID.isin(confirmed_by_later_visit) & ~conv.NACCID.isin(reverted_ids)).sum()),
              "no_later_visit_to_check": int((~conv.NACCID.isin(confirmed_by_later_visit | reverted_ids)).sum())}

    keep = ~((first[TARGET] == 1) & first.NACCID.isin(reverted_ids))
    df = first[keep].reset_index(drop=True)
    counts["cohort_after_removing_reverted"] = {"n": int(len(df)), "events": int(df[TARGET].sum())}

    d = M.split_xy(df, M.CORE, TARGET)
    preds = []
    for seed in SEEDS:
        predict = M.fit_xgboost(d, seed=seed, depths=(4,))
        preds.append({s: predict(d[s][0]) for s in ["test", "external"]})
    p = {s: np.mean([q[s] for q in preds], axis=0) for s in ["test", "external"]}

    results = {"target": "dementia within 3 years, reversed diagnoses removed", "counts": counts, "auroc": {}}
    for split in ["test", "external"]:
        part = df[df.split == split]
        y, dx = part[TARGET].to_numpy(int), part.NACCUDSD.to_numpy()
        results["auroc"][split] = {name: M.bootstrap(y[np.isin(dx, codes)], p[split][np.isin(dx, codes)], n_boot=500)
                                   for name, codes in GROUPS.items()}
        print(f"[confirmed] {split}: " + ", ".join(f"{g} {r['auroc']:.3f} ({r['n']} / {r['events']})"
                                                   for g, r in results["auroc"][split].items()), flush=True)
    print("[confirmed] counts:", counts, flush=True)

    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "confirmed-conversion sensitivity")
    (nu.REPORTS / "sensitivity_confirmed_conversion.json").write_text(text)


if __name__ == "__main__":
    main()
