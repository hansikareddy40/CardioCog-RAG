"""Phase 4: do amyloid and tau PET measures add to the clinical model?

The PET cohort is small (hundreds of training participants, tens of events),
so the design keeps the number of fitted parameters low:

  1. A clinical model (XGBoost, no PET) is trained on *all* landmark visits of
     the training participants, PET or not. This uses the large cohort.
  2. Its output for a PET participant is one number: the clinical risk.
  3. A small logistic regression then combines that clinical risk with a
     handful of PET measures. Only this step is fitted on PET participants.

This is "late fusion": each data source has its own model and a small model
combines them. A participant without PET simply keeps the clinical risk.

Comparisons are paired: the same test participants, with and without PET.

Output  reports/phase4_pet_results.json (aggregate only)
Usage:  python scripts/train_pet.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import modeling as M
import nacc_utils as nu

SEEDS = [0, 1, 2]
AMYLOID = ["amyloid_centiloids", "amyloid_amyloid_status"]
TAU = ["tau_meta_temporal_suvr_z", "tau_ctx_entorhinal_suvr_z"]


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def clinical_risk(df: pd.DataFrame, target: str) -> pd.Series:
    """Clinical-only risk for every PET index visit, from a model trained on the big cohort."""
    lm = df[df.eligible & df[target].notna()]
    d = {s: (lm.loc[lm.split == s, M.CORE].to_numpy(np.float32), lm.loc[lm.split == s, target].to_numpy(int)) for s in M.SPLITS}
    pet = df[df.is_pet_index]
    X = pet[M.CORE].to_numpy(np.float32)
    return pd.Series(np.mean([M.fit_xgboost(d, seed=s, depths=(4,))(X) for s in SEEDS], axis=0), index=pet.index)


def fuse(train: pd.DataFrame, test: pd.DataFrame, pet_cols: list[str], target: str) -> np.ndarray:
    """Logistic regression on [clinical risk (log-odds), PET measures]."""
    def design(d, med):
        cols = [d["clin_logit"].to_numpy()]
        for c in pet_cols:
            cols += [d[c].fillna(med[c]).to_numpy(), d[c].isna().to_numpy(float)]
        return np.column_stack(cols)
    med = train[pet_cols].median()
    Xtr, Xte = design(train, med), design(test, med)
    keep = Xtr.std(axis=0) > 0                       # drop 'missing' flags that never fire
    mu, sd = Xtr[:, keep].mean(axis=0), Xtr[:, keep].std(axis=0)
    m = LogisticRegression(C=1.0, max_iter=2000).fit((Xtr[:, keep] - mu) / sd, train[target].to_numpy(int))
    return m.predict_proba((Xte[:, keep] - mu) / sd)[:, 1]


def run(df: pd.DataFrame, target: str) -> dict:
    pet = df[df.is_pet_index & df[target].notna()].copy()
    pet["clin_logit"] = logit(clinical_risk(df, target).loc[pet.index])
    fit_on = pet[pet.split.isin(["train", "val"])]            # the fusion step has no tuning, so val joins train
    out = {"cohort": {s: {"n": int((pet.split == s).sum()), "events": int(pet.loc[pet.split == s, target].sum())} for s in M.SPLITS}}

    subsets = {"has amyloid PET": pet.has_amyloid == 1, "has amyloid and tau PET": (pet.has_amyloid == 1) & (pet.has_tau == 1)}
    for sub_name, mask in subsets.items():
        tr = fit_on[mask.loc[fit_on.index]]
        variants = {"Clinical only": [], "Clinical + amyloid": AMYLOID}
        if "tau" in sub_name:
            variants["Clinical + amyloid + tau"] = AMYLOID + TAU
        res = {"train_n": int(len(tr)), "train_events": int(tr[target].sum())}
        # held-out evaluation: internal test, external centres, and both pooled
        for eval_name, splits in [("test", ["test"]), ("external", ["external"]), ("test + external", ["test", "external"])]:
            te = pet[mask & pet.split.isin(splits)]
            y = te[target].to_numpy(int)
            preds = {name: fuse(tr, te, cols, target) for name, cols in variants.items()}
            block = {name: M.bootstrap(y, p, p_ref=None if name == "Clinical only" else preds["Clinical only"]) for name, p in preds.items()}
            for dx_name, code in [("normal at index", 1), ("MCI at index", 3)]:
                k = te.NACCUDSD.to_numpy() == code
                if y[k].sum() >= 10:
                    block[f"{dx_name}"] = {name: M.bootstrap(y[k], p[k], n_boot=500, p_ref=None if name == "Clinical only" else preds["Clinical only"][k])
                                           for name, p in preds.items()}
            res[eval_name] = block
        out[sub_name] = res
    # How PET relates to the outcome on its own, for context
    te = pet[pet.split.isin(["test", "external"]) & (pet.has_amyloid == 1)]
    out["conversion_rate_by_amyloid_status"] = {
        dx: te[te.NACCUDSD == code].groupby("amyloid_amyloid_status")[target].agg(["size", "sum"]).rename(
            index={0.0: "negative", 1.0: "positive"}, columns={"size": "n", "sum": "events"}).to_dict("index")
        for dx, code in [("normal at index", 1), ("MCI at index", 3)]}
    return out


def main():
    df = M.load_visits()
    results = {t: run(df, t) for t in ["y_dementia_2y", "y_dementia_3y"]}
    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "PET results")
    (nu.REPORTS / "phase4_pet_results.json").write_text(text)
    for t, r in results.items():
        print(t, r["cohort"])
        for sub in ["has amyloid PET", "has amyloid and tau PET"]:
            for ev in ["test", "external", "test + external"]:
                for name, m in r[sub][ev].items():
                    if "auroc" in m:
                        print(f"  {sub} | {ev} | {name}: n={m['n']} ev={m['events']} AUROC {m['auroc']:.3f} "
                              f"gain {m.get('delta_auroc', 0):+.4f} {[round(x, 4) for x in m.get('delta_auroc_ci', [])]}")
        print("  ", r["conversion_rate_by_amyloid_status"])


if __name__ == "__main__":
    main()
