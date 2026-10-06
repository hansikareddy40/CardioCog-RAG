"""Independent checks on the data pipeline, the labels and the reported numbers.

Each check recomputes something a second way, or tests a property that must
hold if the pipeline is sound, and reports PASS or FAIL with the evidence.

  A  raw files were not modified
  B  splits: no participant and no centre on both sides
  C  no future information: features at a visit are identical when every later
     visit is deleted before the features are built
  D  labels: recomputed with a separate, simple implementation
  E  shuffled-label test: with labels scrambled, a model must fall to chance
  F  metrics: recomputed from the saved predictions and compared with the reports
  G  retraining from scratch with a new random seed gives the same result
  H  cohort counts in the reports match the processed table; PET values are
     never newer than the visit they are used at

Output  reports/verification_report.json and a printed summary
Usage:  python scripts/verify_results.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import build_dataset as B
import modeling as M
import nacc_utils as nu

TARGET = "y_dementia_3y"
RESULTS = []


def record(check: str, passed: bool, detail: str) -> None:
    RESULTS.append({"check": check, "passed": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {check}: {detail}", flush=True)


def check_raw_untouched():
    audit = {t["file"]: t for t in json.loads(nu.AUDIT_JSON.read_text())["tables"]}
    files = [nu.UDS_CSV, nu.EDC_CSV, *sorted(nu.SCAN_DIR.glob("*.csv")), *sorted(nu.MP_DIR.glob("*.csv"))]
    newest = max(pd.Timestamp(f.stat().st_mtime, unit="s") for f in files)
    sizes_ok = all(abs(f.stat().st_size / 1e6 - audit[f.name]["size_mb"]) < 0.06 for f in files)
    first_commit = pd.Timestamp("2026-10-05")          # date of the first audit of these files
    record("A raw files unmodified", sizes_ok and newest < first_commit,
           f"{len(files)} tables; sizes match the first audit: {sizes_ok}; most recent modification {newest:%Y-%m-%d}, before any processing began")
    n_dicom_dirs = sum(1 for m in ["Amyloid", "Tau", "FDG"] for d in (nu.RAW / "pet" / m).iterdir() if d.is_dir())
    record("A PET folders present", n_dicom_dirs == 599, f"{n_dicom_dirs} scan folders (599 at the first audit)")


def check_splits(df):
    per = df.groupby("NACCID").agg(n_split=("split", "nunique"), split=("split", "first"), centre=("NACCADC", "first"))
    ext, rest = set(per[per.split == "external"].centre), set(per[per.split != "external"].centre)
    record("B participant in one split only", (per.n_split == 1).all(), f"{int((per.n_split > 1).sum())} participants appear in more than one split")
    record("B external centres unseen", not (ext & rest), f"{len(ext)} external centres, {len(rest)} other centres, {len(ext & rest)} shared")
    saved = pd.read_parquet(M.PROCESSED / "splits.parquet").set_index("NACCID").split
    record("B split file matches table", (saved.reindex(per.index) == per.split).all(), "saved split file agrees with the modelling table")


def rebuild_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Run the feature steps of build_dataset on a (possibly truncated) raw table."""
    norms = json.loads((M.PROCESSED / "cog_norms.json").read_text())
    df = nu.add_time(nu.clean(raw))
    raw = raw.loc[df.index] if not raw.index.equals(df.index) else raw
    df = B.add_demographics(df)
    df = B.add_history(df, raw)
    df = B.add_cognition(df, raw, norms)
    return B.add_trajectory(df)


def check_no_future_information(df):
    """Delete all visits after visit k, rebuild, and compare the features at visit k."""
    need = sorted(set(nu.VARIABLES) - {"NACCID", "VISITDATE"} | {"NACCSTYR", "NACCTIYR", *B.NPI_ITEMS, *B.ALL_TESTS})
    raw = nu.load_uds(need)
    rng = np.random.default_rng(1)
    counts = raw.groupby("NACCID").size()
    ids = rng.choice(counts.index[counts >= 5], 1500, replace=False)
    raw = raw[raw.NACCID.isin(ids)].reset_index(drop=True)
    feats = M.CORE + M.PROXIMAL + M.TRAJECTORY
    full = df[df.NACCID.isin(ids)].set_index(["NACCID", "NACCVNUM"])[feats]
    for k in (1, 3):
        trunc = rebuild_features(raw[raw.NACCVNUM <= k].reset_index(drop=True))
        t = trunc[trunc.NACCVNUM == k].set_index(["NACCID", "NACCVNUM"])[feats]
        f = full.loc[t.index]
        diff = ~((t.isna() & f.isna()) | (np.abs(t.astype(float) - f.astype(float)) < 1e-6))
        bad = diff.sum()
        bad = bad[bad > 0]
        record(f"C no future information at visit {k}", bad.empty,
               f"{len(t)} participants x {len(feats)} features compared after deleting later visits; "
               + ("all identical" if bad.empty else f"features that changed: {bad.to_dict()}"))


def check_labels(df):
    """Recompute the 3-year label with plain loops on a sample."""
    rng = np.random.default_rng(2)
    ids = rng.choice(df.NACCID.unique(), 3000, replace=False)
    d = df[df.NACCID.isin(ids)].sort_values(["NACCID", "VISITDATE"])
    wrong = n = 0
    for _, g in d.groupby("NACCID"):
        t, dx = g.years.to_numpy(), g.NACCUDSD.to_numpy()
        for i in range(len(g)):
            later = [(t[j] - t[i], dx[j]) for j in range(i + 1, len(g))]
            dem = [dt for dt, x in later if x == 4]
            expected = 1.0 if dem and min(dem) <= 3 else (0.0 if later and max(dt for dt, _ in later) >= 3 else np.nan)
            got = g[TARGET].iloc[i]
            n += 1
            wrong += not ((np.isnan(expected) and np.isnan(got)) or expected == got)
    record("D label recomputed independently", wrong == 0, f"{n:,} visits of 3,000 participants; {wrong} disagreements")
    elig = df[df.eligible]
    record("D no demented index visits", (elig.NACCUDSD != 4).all() and elig.NACCUDSD.notna().all(), f"{len(elig):,} eligible visits, none with dementia at the visit")


def check_shuffled_labels(base):
    d = M.split_xy(base, M.CORE, TARGET)
    rng = np.random.default_rng(3)
    shuffled = {s: (X, rng.permutation(y)) for s, (X, y) in d.items()}
    p = M.fit_xgboost(shuffled, seed=0, depths=(4,))(d["test"][0])
    auc = roc_auc_score(shuffled["test"][1], p)
    record("E shuffled labels give chance", 0.44 < auc < 0.56, f"AUROC {auc:.3f} with scrambled labels (0.5 is chance)")


def check_metrics(base):
    rep = json.loads((nu.REPORTS / "phase3_tabular_results.json").read_text())
    preds = pd.read_parquet(M.PROCESSED / "preds_tabular.parquet")
    worst = 0.0
    for name, r in rep["models"].items():
        for s in ["test", "external"]:
            p = preds[preds.split == s]
            auc = roc_auc_score(p[TARGET], p[f"p_{name}"])
            worst = max(worst, abs(auc - r[s]["auroc"]))
    record("F reported AUROC equals recomputed", worst < 1e-9, f"largest difference over 5 models x 2 test sets: {worst:.2e}")
    n_ok = all(rep["cohort"][s]["n"] == int((base.split == s).sum()) and rep["cohort"][s]["events"] == int(base.loc[base.split == s, TARGET].sum()) for s in M.SPLITS)
    record("F cohort sizes in model report", n_ok, json.dumps(rep["cohort"]))
    return rep


def check_retrain(base, rep):
    d = M.split_xy(base, M.CORE, TARGET)
    predict = M.fit_xgboost(d, seed=123, depths=(4,))
    out = {s: roc_auc_score(d[s][1], predict(d[s][0])) for s in ["test", "external"]}
    inside = all(rep["models"]["XGBoost"][s]["auroc_ci"][0] <= out[s] <= rep["models"]["XGBoost"][s]["auroc_ci"][1] for s in out)
    record("G retraining reproduces the result", inside,
           f"new seed: test {out['test']:.3f}, external {out['external']:.3f}; reported {rep['models']['XGBoost']['test']['auroc']:.3f} and {rep['models']['XGBoost']['external']['auroc']:.3f}")
    # the cardiovascular conclusion, recomputed with single models
    cardio = M.FEATURE_GROUPS["Cardiovascular / medical"]
    without = [f for f in M.CORE if f not in cardio]
    dw = M.split_xy(base, without, TARGET)
    pw = M.fit_xgboost(dw, seed=123, depths=(4,))
    gain = out["test"] - roc_auc_score(dw["test"][1], pw(dw["test"][0]))
    record("G cardiovascular gain is negligible", abs(gain) < 0.005, f"test AUROC gain from the cardiovascular group with a fresh seed: {gain:+.4f}")


def check_counts(df):
    s = json.loads((nu.REPORTS / "phase2_cohort_summary.json").read_text())
    v1 = df[(df.NACCVNUM == 1) & df.eligible]
    ok = s["flow_baseline"]["...and 3-year status known"] == int(v1[TARGET].notna().sum()) and \
        s["flow_baseline"]["...of whom converted to dementia"] == int(v1[TARGET].sum()) and s["flow_baseline"]["participants"] == df.NACCID.nunique()
    record("H cohort summary matches table", ok, f"{df.NACCID.nunique():,} participants; {int(v1[TARGET].notna().sum()):,} with known status; {int(v1[TARGET].sum()):,} converted")
    record("H PET index visits", s["pet_index"]["participants"] == int(df.is_pet_index.sum()), f"{int(df.is_pet_index.sum()):,} PET index visits")
    pet = df[df.is_pet_index]
    late = int((pet.amyloid_gap_days > 0).sum() + (pet.tau_gap_days > 0).sum())
    record("C PET scan never newer than its index visit", late == 0,
           f"{int(pet.has_amyloid.sum()):,} amyloid and {int(pet.has_tau.sum()):,} tau values; {late} taken after the visit they are used at")


def main():
    df = M.load_visits()
    base = df[(df.NACCVNUM == 1) & df.eligible & df[TARGET].notna()].reset_index(drop=True)
    check_raw_untouched()
    check_splits(df)
    check_no_future_information(df)
    check_labels(df)
    check_shuffled_labels(base)
    rep = check_metrics(base)
    check_retrain(base, rep)
    check_counts(df)
    text = json.dumps({"passed": int(sum(r["passed"] for r in RESULTS)), "total": len(RESULTS), "checks": RESULTS}, indent=1)
    nu.assert_no_ids(text, "verification report")
    (nu.REPORTS / "verification_report.json").write_text(text)
    print(f"\n{sum(r['passed'] for r in RESULTS)} of {len(RESULTS)} checks passed")


if __name__ == "__main__":
    main()
