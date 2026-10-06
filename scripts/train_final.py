"""Phase 6: a model that still works when whole groups of inputs are missing.

In a clinic not every patient has every assessment. A model trained only on
complete records has never seen "no cognitive tests at all" and may behave
badly when given that. Two remedies are compared:

  standard    the ordinary model, simply handed missing values at prediction time
  dropout     the same model trained on extra copies of the training data in
              which whole groups were blanked at random ("modality dropout"),
              so it has practised with every combination
  specialist  a separate model retrained for each combination. This is the
              best achievable for that combination, used as the yardstick.

The dropout model is then saved for the interface, together with the small
PET fusion model (clinical risk + PET measures).

Output  reports/phase6_fusion_results.json   aggregate metrics
        models/                              trained models (not committed)

Usage:  python scripts/train_final.py
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import modeling as M
import nacc_utils as nu
from train_pet import AMYLOID, TAU, logit

TARGET = "y_dementia_3y"
SEEDS = [0, 1, 2]
MODELS_DIR = nu.REPO / "models"
OPTIONAL = ["Genetics / APOE", "Cardiovascular / medical", "Cognitive tests", "Clinical stage (CDR-SB, diagnosis)"]
DROP_PROB = 0.35
N_COPIES = 4

# The combinations evaluated: which optional groups are available.
PATTERNS = {
    "Everything": OPTIONAL,
    "No APOE": [g for g in OPTIONAL if g != "Genetics / APOE"],
    "No cardiovascular data": [g for g in OPTIONAL if g != "Cardiovascular / medical"],
    "No cognitive tests": [g for g in OPTIONAL if g != "Cognitive tests"],
    "No clinical stage (CDR)": [g for g in OPTIONAL if g != "Clinical stage (CDR-SB, diagnosis)"],
    "No cognitive assessment at all": ["Genetics / APOE", "Cardiovascular / medical"],
    "Demographics only": [],
}
COLS = {g: [M.CORE.index(f) for f in feats] for g, feats in M.FEATURE_GROUPS.items()}


def mask(X: np.ndarray, available: list[str]) -> np.ndarray:
    X = X.copy()
    for g in OPTIONAL:
        if g not in available:
            X[:, COLS[g]] = np.nan
    return X


def with_dropout(X: np.ndarray, y: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Original rows plus N_COPIES copies with random groups blanked."""
    Xs, ys = [X], [y]
    for _ in range(N_COPIES):
        Xc = X.copy()
        for g in OPTIONAL:
            rows = rng.random(len(X)) < DROP_PROB
            Xc[np.ix_(rows, COLS[g])] = np.nan
        Xs.append(Xc); ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


def main():
    df = M.load_visits()
    base = df[(df.NACCVNUM == 1) & df.eligible & df[TARGET].notna()].reset_index(drop=True)
    d = M.split_xy(base, M.CORE, TARGET)
    rng = np.random.default_rng(0)

    standard = [M.fit_xgboost(d, seed=s, depths=(4,)) for s in SEEDS]
    d_aug = dict(d)
    d_aug["train"] = with_dropout(*d["train"], rng)
    d_aug["val"] = with_dropout(*d["val"], rng)
    dropout = [M.fit_xgboost(d_aug, seed=s, depths=(4,)) for s in SEEDS]
    avg = lambda models, X: np.mean([m(X) for m in models], axis=0)

    results = {"target": TARGET, "drop_probability": DROP_PROB, "copies": N_COPIES, "patterns": {}}
    for name, available in PATTERNS.items():
        feats = M.FEATURE_GROUPS["Demographics"] + [f for g in available for f in M.FEATURE_GROUPS[g]]
        spec_d = M.split_xy(base, feats, TARGET)
        specialist = [M.fit_xgboost(spec_d, seed=s, depths=(4,)) for s in SEEDS]
        block = {}
        for s in ["test", "external"]:
            y = d[s][1]
            Xm = mask(d[s][0], available)
            p_spec = avg(specialist, spec_d[s][0])
            block[s] = {"specialist": M.bootstrap(y, p_spec, n_boot=500),
                        "standard": M.bootstrap(y, avg(standard, Xm), n_boot=500, p_ref=p_spec),
                        "dropout": M.bootstrap(y, avg(dropout, Xm), n_boot=500, p_ref=p_spec)}
        results["patterns"][name] = block
        t = block["test"]
        print(f"{name}: specialist {t['specialist']['auroc']:.3f} | standard {t['standard']['auroc']:.3f} (Brier {t['standard']['brier']:.3f}) "
              f"| dropout {t['dropout']['auroc']:.3f} (Brier {t['dropout']['brier']:.3f}) | specialist Brier {t['specialist']['brier']:.3f}", flush=True)

    # ---- PET fusion on top of the dropout model ----------------------------
    pet = df[df.is_pet_index & df[TARGET].notna()].copy()
    pet["clin_logit"] = logit(avg(dropout, pet[M.CORE].to_numpy(np.float32)))
    fit = pet[pet.split.isin(["train", "val"]) & (pet.has_amyloid == 1)]
    cols = AMYLOID + TAU
    med = fit[cols].median()
    design = lambda t: np.column_stack([t["clin_logit"]] + [t[c].fillna(med[c]) for c in cols] + [t[c].isna().astype(float) for c in TAU])
    Xf = design(fit)
    mu, sd = Xf.mean(axis=0), Xf.std(axis=0) + 1e-9
    lr = LogisticRegression(C=0.3, max_iter=2000).fit((Xf - mu) / sd, fit[TARGET].to_numpy(int))      # few events: shrink firmly
    held = pet[pet.split.isin(["test", "external"]) & (pet.has_amyloid == 1)]
    p_clin = 1 / (1 + np.exp(-held["clin_logit"].to_numpy()))
    p_fused = lr.predict_proba((design(held) - mu) / sd)[:, 1]
    y = held[TARGET].to_numpy(int)
    results["pet_fusion"] = {"fitted_on": {"n": int(len(fit)), "events": int(fit[TARGET].sum())},
                             "held_out": {"clinical_only": M.bootstrap(y, p_clin), "clinical_plus_pet": M.bootstrap(y, p_fused, p_ref=p_clin)}}

    # ---- save for the interface --------------------------------------------
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir()
    for i, m in enumerate(dropout):
        m.model.save_model(MODELS_DIR / f"clinical_xgb_{i}.json")
    shutil.copy(M.PROCESSED / "cog_norms.json", MODELS_DIR / "cog_norms.json")
    shutil.copy(M.PROCESSED / "tau_norms.json", MODELS_DIR / "tau_norms.json")
    tr = base[base.split == "train"]
    meta = {
        "target": "dementia diagnosis within 3 years", "features": M.CORE, "feature_groups": M.FEATURE_GROUPS,
        "pet_fusion": {"columns": ["clin_logit"] + cols + [c + "_missing" for c in TAU], "median": med.to_dict(), "mean": mu.tolist(), "sd": sd.tolist(),
                       "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
        # cohort reference values shown next to a prediction (aggregates)
        "reference": {"event_rate": float(tr[TARGET].mean()),
                      "event_rate_by_dx": {nu.DX_LABELS[k]: float(v) for k, v in tr.groupby("NACCUDSD")[TARGET].mean().items()},
                      "risk_percentiles": np.percentile(avg(dropout, d["train"][0]), np.arange(0, 101, 5)).round(4).tolist(),
                      "n_train": int(len(tr))},
    }
    text = json.dumps(meta, indent=1)
    nu.assert_no_ids(text, "model metadata")
    (MODELS_DIR / "meta.json").write_text(text)
    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "fusion results")
    (nu.REPORTS / "phase6_fusion_results.json").write_text(text)
    print("PET fusion held-out:", {k: round(v["auroc"], 3) for k, v in results["pet_fusion"]["held_out"].items()})
    print("saved models to", MODELS_DIR)


if __name__ == "__main__":
    main()
