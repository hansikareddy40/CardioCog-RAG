"""Phase 3c: does a participant's history improve prediction?

Design
------
A "landmark" is a visit at which we pretend to make the prediction. Here a
landmark is any in-person visit where the participant is not demented, has at
least two earlier visits, and has a known 3-year outcome.

Training uses every landmark of the training participants. Testing uses one
randomly chosen landmark per test participant, so each person counts once.

Three models are compared on exactly the same test landmarks:
  1. XGBoost, landmark-visit features only (no history)
  2. XGBoost, plus engineered trajectory features (slopes and changes)
  3. GRU that reads the sequence of visits, plus the landmark-visit features

Output  reports/phase3_longitudinal_results.json (aggregate only)

Usage:  python scripts/train_longitudinal.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import modeling as M
import nacc_utils as nu

TARGET = "y_dementia_3y"
MAX_VISITS = 6
SEEDS = [0, 1, 2]
STEP_FEATURES = ["z_memory", "z_attention", "z_executive", "z_language", "z_visuospatial", "z_global",
                 "tests_failed_cognitive", "gds", "npi_count", "CDRSUM", "is_mci", "is_impaired_not_mci",
                 "bp_sys", "bp_dia", "heart_rate", "bmi", "med_count"]


def log(msg):
    print(f"[longitudinal] {msg}", flush=True)


class VisitGRU(nn.Module):
    """A GRU reads the visits in time order; its final state is joined with the
    landmark-visit features and passed to a small output network."""

    def __init__(self, n_step, n_static, hidden=48, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_step, hidden, batch_first=True)
        self.static = nn.Sequential(nn.Linear(n_static, 96), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(hidden + 96, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, seq, static):
        _, h = self.gru(seq)                     # sequences are left-padded, so the last step is the landmark
        return self.head(torch.cat([h[-1], self.static(static)], dim=1)).squeeze(-1)


def build_sequences(df: pd.DataFrame, rows: np.ndarray, mean, std) -> np.ndarray:
    """(n, MAX_VISITS, features) array of the visits up to each landmark.

    Per step: standardised values (0 where missing), a 'missing' flag per value,
    years before the landmark, and a flag marking padding.
    """
    values = ((df[STEP_FEATURES].to_numpy(np.float32) - mean) / std)
    years = df["years"].to_numpy(np.float32)
    count = df["n_prior_visits"].to_numpy()
    n, f = len(rows), len(STEP_FEATURES)
    out = np.zeros((n, MAX_VISITS, 2 * f + 2), dtype=np.float32)
    for k in range(MAX_VISITS):
        back = MAX_VISITS - 1 - k                # how many visits before the landmark
        ok = back <= count[rows]
        src = np.where(ok, rows - back, rows)
        v = values[src]
        out[:, k, :f] = np.where(ok[:, None], np.nan_to_num(v), 0)
        out[:, k, f:2 * f] = np.where(ok[:, None], np.isnan(v), 0)
        out[:, k, 2 * f] = np.where(ok, years[rows] - years[src], 0)
        out[:, k, 2 * f + 1] = ~ok
    return out


def main():
    t0 = time.time()
    df = M.load_visits().sort_values(["NACCID", "VISITDATE"]).reset_index(drop=True)
    lm = df.index[df.eligible & (df.n_prior_visits >= 2) & df[TARGET].notna()].to_numpy()
    rng = np.random.default_rng(0)
    rows = {"train": lm[df.split.to_numpy()[lm] == "train"], "val": lm[df.split.to_numpy()[lm] == "val"]}
    for s in ["test", "external"]:             # one random landmark per participant
        cand = pd.Series(lm[df.split.to_numpy()[lm] == s])
        rows[s] = cand.groupby(df.NACCID.to_numpy()[cand]).apply(lambda x: x.iloc[rng.integers(len(x))]).to_numpy()
    y = {s: df[TARGET].to_numpy(int)[r] for s, r in rows.items()}
    results = {"target": TARGET, "landmarks": {s: {"n": int(len(r)), "participants": int(df.NACCID.iloc[r].nunique()),
                                                   "events": int(y[s].sum())} for s, r in rows.items()},
               "prior_visits_at_test_landmark": pd.Series(df.n_prior_visits.to_numpy()[rows["test"]]).describe()[["mean", "50%", "max"]].round(1).to_dict()}
    log(json.dumps(results["landmarks"]))

    def tab(features):
        return {s: (df[features].to_numpy(np.float32)[r], y[s]) for s, r in rows.items()}

    preds = {}
    for name, feats in [("XGBoost, landmark visit only", M.CORE), ("XGBoost + trajectory features", M.CORE + M.TRAJECTORY)]:
        d = tab(feats)
        ps = [M.fit_xgboost(d, seed=s, depths=(4,)) for s in SEEDS]
        preds[name] = {s: np.mean([p(d[s][0]) for p in ps], axis=0) for s in ["test", "external"]}
        log(f"{name}: test AUROC {M.roc_auc_score(y['test'], preds[name]['test']):.3f} ({time.time()-t0:.0f}s)")

    # ---- GRU ---------------------------------------------------------------
    d = tab(M.CORE)
    prep = M.Prep().fit(d["train"][0])
    tr = df[STEP_FEATURES].to_numpy(np.float32)[rows["train"]]
    mean, std = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0) + 1e-6
    tens = {s: (torch.from_numpy(build_sequences(df, r, mean, std)), torch.from_numpy(prep.transform(d[s][0])),
                torch.from_numpy(y[s])) for s, r in rows.items()}
    fwd = lambda m, seq, static: m(seq, static)
    gru_preds = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = VisitGRU(tens["train"][0].shape[2], tens["train"][1].shape[1])
        model = M.train_torch(model, tens["train"], tens["val"], fwd, epochs=40, lr=1e-3, batch=512, patience=6, seed=seed)
        gru_preds.append({s: M.predict_torch(model, tens[s][:2], fwd) for s in ["test", "external"]})
        log(f"GRU seed {seed}: test AUROC {M.roc_auc_score(y['test'], gru_preds[-1]['test']):.3f} ({time.time()-t0:.0f}s)")
    preds["GRU over visit sequence"] = {s: np.mean([p[s] for p in gru_preds], axis=0) for s in ["test", "external"]}

    ref = preds["XGBoost, landmark visit only"]
    results["models"] = {}
    for name, p in preds.items():
        results["models"][name] = {s: M.bootstrap(y[s], p[s], p_ref=None if p is ref else ref[s]) for s in ["test", "external"]}
    # history should matter most for people with more of it
    many = df.n_prior_visits.to_numpy()[rows["test"]] >= 4
    results["test_by_history_length"] = {
        label: {name: M.bootstrap(y["test"][mask], p["test"][mask], n_boot=500, p_ref=None if p is ref else ref["test"][mask])
                for name, p in preds.items()}
        for label, mask in [("2-3 prior visits", ~many), ("4+ prior visits", many)]}

    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "longitudinal results")
    (nu.REPORTS / "phase3_longitudinal_results.json").write_text(text)
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
