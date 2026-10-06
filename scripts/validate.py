"""Phase 8: validation of the main tabular model beyond a single AUROC.

Uses the XGBoost predictions saved by train_tabular.py (first-visit cohort,
dementia within 3 years) on the internal test set and the held-out centres.

  1. Subgroups        does it work equally well by sex, age, race, education, APOE?
  2. Thresholds       sensitivity / specificity / PPV / NPV at pre-specified risk cut-offs
  3. Calibration      slope and intercept (1 and 0 are ideal)
  4. Decision curve   is acting on the model better than treating everyone or no one?
  5. Survival check   a Cox model on *everyone* eligible, including people whose
                      3-year status is unknown, to see whether leaving them out
                      changed the conclusions

Output  reports/phase8_validation_results.json (aggregate only)
Usage:  python scripts/validate.py
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LogisticRegression

import modeling as M
import nacc_utils as nu

warnings.filterwarnings("ignore")
TARGET = "y_dementia_3y"
P = "p_XGBoost"
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50]
MIN_EVENTS = 15                     # subgroups with fewer events are reported as "too few"


def subgroup_columns(d: pd.DataFrame) -> dict:
    return {
        "Sex": d.female.map({0.0: "Male", 1.0: "Female"}),
        "Age": pd.cut(d.age, [0, 65, 75, 85, 200], right=False, labels=["Under 65", "65-74", "75-84", "85+"]).astype(str),
        "Race": np.select([d.race_white == 1, d.race_black == 1], ["White", "Black"], "Other / unknown"),
        "Education": pd.cut(d.educ_years, [-1, 12, 16, 99], labels=["12 years or less", "13-16 years", "Over 16 years"]).astype(str),
        "APOE": np.select([d.apoe_unknown == 1, d.apoe_e4_count >= 1], ["Not genotyped", "e4 carrier"], "No e4"),
        "Diagnosis at index": d.NACCUDSD.map({1.0: "Normal", 2.0: "Impaired, not MCI", 3.0: "MCI"}),
    }


def subgroups(d: pd.DataFrame) -> dict:
    out = {}
    for name, col in subgroup_columns(d).items():
        for level, g in d.groupby(np.asarray(col)):
            if level in ("nan", "None"):
                continue
            y, p = g[TARGET].to_numpy(int), g[P].to_numpy()
            r = {"n": int(len(g)), "events": int(y.sum()), "observed_rate": float(y.mean()), "mean_predicted": float(p.mean())}
            if y.sum() >= MIN_EVENTS and (1 - y).sum() >= MIN_EVENTS:
                b = M.bootstrap(y, p, n_boot=500)
                r.update(auroc=b["auroc"], auroc_ci=b["auroc_ci"])
            out[f"{name}: {level}"] = r
    return out


def thresholds(y, p) -> dict:
    out = {}
    for t in THRESHOLDS:
        pos = p >= t
        tp, fp, fn, tn = (pos & (y == 1)).sum(), (pos & (y == 0)).sum(), (~pos & (y == 1)).sum(), (~pos & (y == 0)).sum()
        out[f"{t:.2f}"] = {"flagged_pct": float(pos.mean() * 100), "sensitivity": float(tp / (tp + fn)), "specificity": float(tn / (tn + fp)),
                           "ppv": float(tp / max(tp + fp, 1)), "npv": float(tn / max(tn + fn, 1))}
    return out


def calibration(y, p) -> dict:
    """Logistic recalibration: observed ~ intercept + slope * logit(predicted)."""
    lp = np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return {"slope": float(m.coef_[0, 0]), "intercept": float(m.intercept_[0]), "observed_rate": float(y.mean()), "mean_predicted": float(p.mean())}


def decision_curve(y, p) -> dict:
    """Net benefit = true positives/n - false positives/n * t/(1-t)."""
    ts = np.round(np.arange(0.02, 0.61, 0.02), 2)
    n, prev = len(y), y.mean()
    model = [float(((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)) for t in ts]
    treat_all = [float(prev - (1 - prev) * t / (1 - t)) for t in ts]
    return {"threshold": ts.tolist(), "model": model, "treat_all": treat_all}


def survival_check(df: pd.DataFrame) -> dict:
    """Cox proportional hazards on all eligible first visits, censored included."""
    c = df[(df.NACCVNUM == 1) & df.eligible & (df.event_time > 0)].copy()
    out = {"cohort": {s: {"n": int((c.split == s).sum()), "events": int(c.loc[c.split == s, "event"].sum())} for s in M.SPLITS},
           "note": "includes participants whose 3-year status is unknown"}
    cardio = M.FEATURE_GROUPS["Cardiovascular / medical"]
    sets = {"All feature groups": M.CORE, "Without cardiovascular group": [f for f in M.CORE if f not in cardio]}
    risk = {}
    for name, feats in sets.items():
        tr = c[c.split == "train"]
        med, keep = tr[feats].median(), [f for f in feats if tr[f].std() > 0]
        mu, sd = tr[keep].fillna(med).mean(), tr[keep].fillna(med).std()
        prep = lambda d: ((d[keep].fillna(med) - mu) / sd).assign(event_time=d.event_time.to_numpy(), event=d.event.to_numpy())
        cox = CoxPHFitter(penalizer=0.05).fit(prep(tr), "event_time", "event")
        risk[name] = {s: cox.predict_partial_hazard(prep(c[c.split == s])).to_numpy() for s in ["test", "external"]}
        if name == "All feature groups":
            hr = cox.summary.loc[["CDRSUM", "z_memory", "age", "apoe_e4_count", "htn_ever", "diabetes_ever", "stroke_ever", "bp_sys", "bmi"],
                                 ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%"]]
            out["hazard_ratio_per_sd"] = hr.round(3).to_dict("index")
    rng = np.random.default_rng(0)
    for s in ["test", "external"]:
        d = c[c.split == s]
        t, e = d.event_time.to_numpy(), d.event.to_numpy()
        ci = {name: float(concordance_index(t, -r[s], e)) for name, r in risk.items()}
        diffs = []
        for _ in range(200):
            i = rng.integers(0, len(d), len(d))
            diffs.append(concordance_index(t[i], -risk["All feature groups"][s][i], e[i]) - concordance_index(t[i], -risk["Without cardiovascular group"][s][i], e[i]))
        out[s] = {"c_index": ci, "gain_from_cardiovascular": ci["All feature groups"] - ci["Without cardiovascular group"],
                  "gain_ci": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    return out


def main():
    df = M.load_visits()
    preds = pd.read_parquet(M.PROCESSED / "preds_tabular.parquet")[["NACCID", P]]
    base = df[(df.NACCVNUM == 1) & df.eligible & df[TARGET].notna()].merge(preds, on="NACCID")
    results = {}
    for s in ["test", "external"]:
        d = base[base.split == s]
        y, p = d[TARGET].to_numpy(int), d[P].to_numpy()
        results[s] = {"n": int(len(d)), "events": int(y.sum()), "subgroups": subgroups(d), "thresholds": thresholds(y, p),
                      "calibration": calibration(y, p), "decision_curve": decision_curve(y, p)}
    results["survival"] = survival_check(df)
    text = json.dumps(results, indent=1)
    nu.assert_no_ids(text, "validation results")
    (nu.REPORTS / "phase8_validation_results.json").write_text(text)
    for s in ["test", "external"]:
        print(s, results[s]["calibration"])
        for k, v in results[s]["subgroups"].items():
            print(f"  {k}: n={v['n']} ev={v['events']} obs={v['observed_rate']:.3f} pred={v['mean_predicted']:.3f} AUROC={v.get('auroc', float('nan')):.3f}")
        print("  thresholds", {k: {a: round(b, 2) for a, b in v.items()} for k, v in results[s]["thresholds"].items()})
    print(json.dumps(results["survival"], indent=1))


if __name__ == "__main__":
    main()
