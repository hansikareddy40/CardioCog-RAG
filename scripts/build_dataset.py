"""Phase 2: turn the raw NACC tables into one clean modelling table.

Reads   data/raw (never written to) through nacc_utils
Writes  data/processed/visits.parquet     one row per visit: features, labels, eligibility
        data/processed/splits.parquet     one row per participant: train / val / test / external
        data/processed/cog_norms.json     the regression norms used for cognitive z-scores
        reports/phase2_cohort_summary.json  counts at every step (aggregate only)

Order matters. The split is made first, because the cognitive norms and the tau
scaling are *fitted* and must only ever see training participants.

Usage:  python scripts/build_dataset.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import nacc_utils as nu

PROCESSED = nu.REPO / "data" / "processed"
SEED = 2026
HORIZONS = (2.0, 3.0)
EXTERNAL_CENTRE_FRACTION = 0.20
PET_WINDOW_DAYS = 365

# Cognitive tests grouped into domains. sign = -1 where a higher raw score is worse.
DOMAINS = {
    "memory": ["LOGIMEM", "MEMUNITS", "CRAFTVRS", "CRAFTDVR", "UDSBENTD"],
    "attention": ["DIGIF", "DIGIB", "DIGFORCT", "DIGBACCT"],
    "executive": ["TRAILA", "TRAILB", "WAIS"],
    "language": ["ANIMALS", "VEG", "BOSTON", "MINTTOTS", "UDSVERFC"],
    "visuospatial": ["UDSBENTC"],
    "global": ["NACCMMSE", "MOCATOTS"],
}
HIGHER_IS_WORSE = {"TRAILA", "TRAILB"}
ALL_TESTS = [t for tests in DOMAINS.values() for t in tests]
NPI_ITEMS = ["DEL", "HALL", "AGIT", "DEPD", "ANX", "ELAT", "APA", "DISN", "IRR", "MOT", "NITE", "APP"]

# (feature name, form A5 variable, form D2 variable)
HISTORY = [
    ("htn_ever", "HYPERTEN", "HYPERT"), ("chol_ever", "HYPERCHO", "HYPCHOL"),
    ("diabetes_ever", "DIABETES", "DIABET"), ("mi_ever", "CVHATT", "MYOINF"),
    ("afib_ever", "CVAFIB", "AFIBRILL"), ("chf_ever", "CVCHF", "CONGHRT"),
    ("revasc_ever", "CVANGIO", "ANGIOPCI"), ("bypass_ever", "CVBYPASS", None),
    ("pacemaker_ever", "CVPACE", "PACEMAKE"), ("othercvd_ever", "CVOTHR", None),
]
TRAJECTORY_VARS = ["CDRSUM", "z_memory", "z_attention", "z_executive", "z_language", "z_global", "faq_total"]


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Splits
# ---------------------------------------------------------------------------

def make_splits(first: pd.DataFrame) -> pd.DataFrame:
    """Hold out whole centres, then split the remaining participants 70/15/15.

    Splitting by participant keeps all of a person's visits on one side.
    Holding out whole centres gives a test set from sites the model never saw.
    """
    rng = np.random.default_rng(SEED)
    centres = np.sort(first["NACCADC"].unique())
    external = set(rng.choice(centres, size=round(len(centres) * EXTERNAL_CENTRE_FRACTION), replace=False))
    split = pd.Series("train", index=first.index, name="split")
    split[first["NACCADC"].isin(external)] = "external"
    # stratify the rest by diagnosis at first visit so each part has the same mix
    rest = first[split != "external"]
    for _, ids in rest.groupby(rest["NACCUDSD"].fillna(0)).groups.items():
        ids = rng.permutation(np.asarray(ids))
        n = len(ids)
        split[ids[int(0.70 * n):int(0.85 * n)]] = "val"
        split[ids[int(0.85 * n):]] = "test"
    return split.reset_index()


# ---------------------------------------------------------------------------
# 2. Features
# ---------------------------------------------------------------------------

def carry_forward_ever(value: pd.Series, ids: pd.Series) -> pd.Series:
    """1 once a condition has ever been reported, 0 if reported absent, NaN before any report."""
    return value.fillna(-1).groupby(ids).cummax().replace(-1, np.nan)


def add_history(df: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    for name, a5, d2 in HISTORY:
        v = df[a5].map({0: 0, 1: 1, 2: 1})                 # absent / recent / remote
        if d2:
            v = v.combine_first((df[d2] >= 1).astype(float).where(df[d2].notna()))
        df[name] = carry_forward_ever(v, df["NACCID"])
    # Stroke and TIA: NACC derives the year of the most recent event at every visit.
    year = df["VISITDATE"].dt.year
    for name, col, a5 in [("stroke_ever", "NACCSTYR", "CBSTROKE"), ("tia_ever", "NACCTIYR", "CBTIA")]:
        y = raw[col]
        v = pd.Series(np.nan, index=df.index)
        v[y == 8888] = 0
        v[(y > 1900) & (y <= year)] = 1
        v = v.combine_first(df[a5].map({0: 0, 1: 1, 2: 1}))   # event reported but year unknown
        df[name] = carry_forward_ever(v, df["NACCID"])
    df["smoke_ever"] = carry_forward_ever(df["TOBAC100"], df["NACCID"])
    df["smoke_years"] = df.groupby("NACCID")["SMOKYRS"].ffill()
    df.loc[df["smoke_ever"] == 0, "smoke_years"] = 0
    # Measured vitals: use the latest in-person measurement up to this visit.
    for name, col in [("bp_sys", "BPSYS"), ("bp_dia", "BPDIAS"), ("heart_rate", "HRATE"), ("bmi", "NACCBMI")]:
        df[name] = df.groupby("NACCID")[col].ffill()
    df["pulse_pressure"] = df["bp_sys"] - df["bp_dia"]
    df["vascular_risk_count"] = df[["htn_ever", "chol_ever", "diabetes_ever", "smoke_ever"]].sum(axis=1, min_count=3)
    for name, col in [("med_bp", "NACCAHTN"), ("med_lipid", "NACCLIPL"), ("med_diabetes", "NACCDBMD"),
                      ("med_anticoag", "NACCAC"), ("med_count", "NACCAMD"), ("med_ad", "NACCADMD")]:
        df[name] = df[col]
    return df


def add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df["age"] = df["NACCAGE"]
    df["female"] = (df["NACCSEX"] == 2).astype(float).where(df["NACCSEX"].notna())
    # Fixed characteristics are carried forward from the visit where they were
    # first recorded. (Taking a person's first non-missing value overall would
    # copy a value recorded at a later visit back to earlier ones.)
    df["educ_years"] = df.groupby("NACCID")["EDUC"].ffill()
    race = df.groupby("NACCID")["NACCNIHR"].ffill()
    for name, code in [("race_white", 1), ("race_black", 2), ("race_asian", 5)]:
        df[name] = (race == code).astype(float).where(race.notna())
    df["hispanic"] = df.groupby("NACCID")["NACCHISP"].ffill()
    df["lives_alone"] = (df["NACCLIVS"] == 1).astype(float).where(df["NACCLIVS"].notna())
    # APOE never changes and NACC attaches the genotype to every visit; "not
    # genotyped" is kept as its own flag instead of being filled in.
    e4 = df.groupby("NACCID")["NACCNE4S"].transform("first")
    df["apoe_e4_count"] = e4
    df["apoe_unknown"] = e4.isna().astype(float)
    df["family_history"] = carry_forward_ever(df["NACCFAM"], df["NACCID"])
    return df


def fit_cog_norms(df: pd.DataFrame, train_ids: set) -> dict:
    """Regress each test on age, sex and education among cognitively normal
    training participants at their first visit. A z-score is then "how far is
    this score from what we expect for a healthy person of this age, sex and
    education". This lets version 1-2 and version 3 tests share one scale."""
    ref = df[(df["NACCVNUM"] == 1) & (df["NACCUDSD"] == 1) & df["NACCID"].isin(train_ids)]
    norms = {}
    for test in ALL_TESTS:
        d = ref[[test, "age", "female", "educ_years"]].dropna()
        X = np.column_stack([np.ones(len(d)), d["age"], d["female"], d["educ_years"]])
        beta, *_ = np.linalg.lstsq(X, d[test].to_numpy(float), rcond=None)
        resid = d[test].to_numpy(float) - X @ beta
        norms[test] = {"n": int(len(d)), "intercept": beta[0], "age": beta[1], "female": beta[2],
                       "educ_years": beta[3], "resid_sd": float(resid.std(ddof=4))}
    return norms


def add_cognition(df: pd.DataFrame, raw: pd.DataFrame, norms: dict) -> pd.DataFrame:
    z = {}
    for test, n in norms.items():
        expected = n["intercept"] + n["age"] * df["age"] + n["female"] * df["female"] + n["educ_years"] * df["educ_years"]
        zt = (df[test] - expected) / n["resid_sd"]
        z[test] = (-zt if test in HIGHER_IS_WORSE else zt).clip(-5, 5)
    z = pd.DataFrame(z)
    for domain, tests in DOMAINS.items():
        df[f"z_{domain}"] = z[tests].mean(axis=1)
    # Code 96 (996 for Trails) = could not do the test because of a cognitive or
    # behavioural problem. That is information, so it is counted, not hidden.
    fail = pd.DataFrame({t: raw[t].isin([96, 996]) for t in ALL_TESTS})
    given = pd.DataFrame({t: raw[t] != -4 for t in ALL_TESTS})
    df["tests_failed_cognitive"] = fail.sum(axis=1).where(given.any(axis=1))
    df["gds"] = df["NACCGDS"]
    npi = pd.DataFrame({c: raw[c].where(raw[c].isin([0, 1])) for c in NPI_ITEMS})
    df["npi_count"] = npi.sum(axis=1, min_count=6)
    df["faq_total"] = df[nu.FAQ_ITEMS].sum(axis=1, min_count=8)
    df["independence"] = df["INDEPEND"]
    df["complaint_self"] = df["DECSUB"]
    df["complaint_informant"] = df["DECIN"]
    df["is_mci"] = (df["NACCUDSD"] == 3).astype(float)
    df["is_impaired_not_mci"] = (df["NACCUDSD"] == 2).astype(float)
    return df


def add_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Slope (per year) over all visits up to and including this one, and change
    since the previous visit. Only past and present values are used."""
    g = df["NACCID"]
    t = df["years"]
    df["n_prior_visits"] = df.groupby("NACCID").cumcount()
    df["years_of_history"] = t
    for v in TRAJECTORY_VARS:
        y = df[v]
        m = y.notna().astype(float)
        n = m.groupby(g).cumsum()
        sx, sy = (t * m).groupby(g).cumsum(), y.fillna(0).groupby(g).cumsum()
        sxx, sxy = (t * t * m).groupby(g).cumsum(), (t * y.fillna(0)).groupby(g).cumsum()
        den = n * sxx - sx ** 2
        slope = (n * sxy - sx * sy) / den.where(den > 1e-9)
        df[f"slope_{v}"] = slope.where((n >= 2) & m.astype(bool)).clip(-10, 10)
        prev = y.groupby(g).ffill().groupby(g).shift(1)
        df[f"delta_{v}"] = y - prev
    return df


# ---------------------------------------------------------------------------
# 3. Labels
# ---------------------------------------------------------------------------

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    g = df["NACCID"]
    t = df["years"]
    last = t.groupby(g).transform("max")
    dem_t = t.where(df["NACCUDSD"] == 4)
    # time of the next dementia diagnosis strictly after this visit
    rev = dem_t.fillna(np.inf)[::-1].groupby(g[::-1]).cummin()[::-1]
    next_dem = rev.groupby(g).shift(-1).replace(np.inf, np.nan)
    df["time_to_dementia"] = next_dem - t
    df["followup_after"] = last - t
    df["event_time"] = df["time_to_dementia"].fillna(df["followup_after"])      # for survival analysis
    df["event"] = df["time_to_dementia"].notna().astype(int)
    for h in HORIZONS:
        y = pd.Series(np.nan, index=df.index)
        y[df["followup_after"] >= h] = 0            # seen at or beyond the horizon
        y[df["time_to_dementia"] <= h] = 1          # diagnosed within it
        df[f"y_dementia_{h:.0f}y"] = y              # NaN = status unknown (censored)
    # Secondary target: CDR-SB at least 1 point higher at any visit within 3 years.
    worst = _max_future_within(df, "CDRSUM", 3.0)
    y = pd.Series(np.nan, index=df.index)
    y[df["followup_after"] >= 3.0] = 0
    y[(worst - df["CDRSUM"]) >= 1] = 1
    df["y_cdrsb_rise_3y"] = y.where(df["CDRSUM"].notna())
    ever_dem = (df["NACCUDSD"] == 4).astype(int).groupby(g).cummax()
    in_person = raw_packet_ok(df)
    df["eligible"] = df["NACCUDSD"].isin([1, 2, 3]) & (ever_dem == 0) & in_person
    return df


def raw_packet_ok(df: pd.DataFrame) -> pd.Series:
    """In-person UDS version 1-3 visits. Telephone visits have no tests, and
    version 4 uses different variables; both still count for outcomes."""
    return df["PACKET"].isin(["I", "F"]) & (df["FORMVER"] < 4)


def _max_future_within(df: pd.DataFrame, col: str, horizon: float) -> pd.Series:
    """Largest value of col at later visits within `horizon` years."""
    out = np.full(len(df), np.nan)
    t, v = df["years"].to_numpy(), df[col].to_numpy(float)
    for idx in df.groupby("NACCID", sort=False).indices.values():
        if len(idx) < 2:
            continue
        ti, vi = t[idx], v[idx]
        for k in range(len(idx) - 1):
            w = vi[k + 1:][(ti[k + 1:] - ti[k]) <= horizon]
            if w.size and not np.all(np.isnan(w)):
                out[idx[k]] = np.nanmax(w)
    return pd.Series(out, index=df.index)


# ---------------------------------------------------------------------------
# 4. PET
# ---------------------------------------------------------------------------

def add_pet(df: pd.DataFrame, train_ids: set) -> pd.DataFrame:
    """Attach each participant's first PET scan to a clinic visit (the PET index visit).

    Two designs are built, because they trade off differently:

    strict   (main)  the first eligible visit on or after the scan, within one
             year. A PET value is never newer than the visit it is used at.
             Clean, but small: recent scans have little follow-up after it.
    nearest          the eligible visit closest to the scan, within one year
             either side. Larger, but for most people the scan was taken a few
             weeks after the visit, so the PET value is slightly newer than
             the visit. Columns for this design end in "_nearest".
    """
    amy = nu.first_scan(nu.load_pet_amyloid().query("QC == 1"))
    tau = nu.first_scan(nu.load_pet_tau().query("QC == 1"))
    # Tau SUVR is tracer-specific: scale within tracer against amyloid-negative training participants.
    ref = tau.merge(amy[["NACCID", "AMYLOID_STATUS"]], on="NACCID", how="left")
    ref = ref[ref["NACCID"].isin(train_ids) & (ref["AMYLOID_STATUS"] == 0)]
    tau_norms = {}
    for col in ["META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR"]:
        stats = ref.groupby("TRACER")[col].agg(["mean", "std"])
        tau[col + "_z"] = (tau[col] - tau["TRACER"].map(stats["mean"])) / tau["TRACER"].map(stats["std"])
        tau_norms[col] = {nu.TRACERS[int(k)]: {"mean": float(v["mean"]), "sd": float(v["std"])} for k, v in stats.iterrows()}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    (PROCESSED / "tau_norms.json").write_text(json.dumps(tau_norms, indent=1))
    anchor = pd.concat([amy[["NACCID", "SCANDATE"]], tau[["NACCID", "SCANDATE"]]]).groupby("NACCID")["SCANDATE"].min()
    cand = df.loc[df["eligible"], ["NACCID", "VISITDATE"]].join(anchor.rename("anchor"), on="NACCID").dropna()
    cand["gap"] = (cand["VISITDATE"] - cand["anchor"]).dt.days          # positive = visit after scan

    for suffix, visit_ok, scan_ok in [("", cand["gap"].between(0, PET_WINDOW_DAYS), (-PET_WINDOW_DAYS, 0)),
                                      ("_nearest", cand["gap"].abs() <= PET_WINDOW_DAYS, (-PET_WINDOW_DAYS, PET_WINDOW_DAYS))]:
        c = cand[visit_ok].assign(dist=lambda d: d["gap"].abs())
        pick = c.sort_values("dist").groupby("NACCID").head(1)
        df[f"is_pet_index{suffix}"] = False
        df.loc[pick.index, f"is_pet_index{suffix}"] = True
        for scans, cols, prefix in [(amy, ["CENTILOIDS", "AMYLOID_STATUS", "TRACER"], "amyloid"),
                                    (tau, ["META_TEMPORAL_SUVR_z", "CTX_ENTORHINAL_SUVR_z", "TRACER"], "tau")]:
            m = df.loc[pick.index, ["NACCID", "VISITDATE"]].reset_index().merge(scans[["NACCID", "SCANDATE", *cols]], on="NACCID")
            m["gap"] = (m["SCANDATE"] - m["VISITDATE"]).dt.days          # positive = scan after visit
            m = m[m["gap"].between(*scan_ok)].set_index("index")
            for col in cols:
                df[f"{prefix}_{col.lower()}{suffix}"] = np.nan
                df.loc[m.index, f"{prefix}_{col.lower()}{suffix}"] = m[col]
            df[f"{prefix}_gap_days{suffix}"] = np.nan
            df.loc[m.index, f"{prefix}_gap_days{suffix}"] = m["gap"]
            df[f"has_{prefix}{suffix}"] = df[f"{prefix}_gap_days{suffix}"].notna().astype(float)
    df["has_pet_image"] = df["NACCID"].isin(nu.load_pet_images()["NACCID"]).astype(float)
    return df


# ---------------------------------------------------------------------------

def main() -> None:
    need = sorted(set(nu.VARIABLES) - {"NACCID", "VISITDATE"} | {"NACCSTYR", "NACCTIYR", *NPI_ITEMS, *ALL_TESTS})
    raw = nu.load_uds(need)
    df = nu.add_time(nu.clean(raw))
    log(f"{len(df):,} visits, {df.NACCID.nunique():,} participants")

    first = df.groupby("NACCID").first()
    splits = make_splits(first[["NACCADC", "NACCUDSD"]])
    train_ids = set(splits.loc[splits.split == "train", "NACCID"])
    log(f"splits: {splits.split.value_counts().to_dict()}")

    df = add_demographics(df)
    df = add_history(df, raw)
    norms = fit_cog_norms(df, train_ids)
    df = add_cognition(df, raw, norms)
    df = add_trajectory(df)
    df = add_labels(df)
    df = add_pet(df, train_ids)
    df = df.merge(splits, on="NACCID")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "visits.parquet", index=False)
    splits.to_parquet(PROCESSED / "splits.parquet", index=False)
    (PROCESSED / "cog_norms.json").write_text(json.dumps(norms, indent=1))

    # ---- aggregate summary (safe to share) --------------------------------
    y = "y_dementia_3y"
    v1 = df[df.NACCVNUM == 1]
    flow = {
        "participants": int(df.NACCID.nunique()),
        "first visit in person, UDS v1-3": int(raw_packet_ok(v1).sum()),
        "...and not demented": int(v1.eligible.sum()),
        "...and 3-year status known": int((v1.eligible & v1[y].notna()).sum()),
        "...of whom converted to dementia": int(v1.loc[v1.eligible, y].sum()),
    }
    cohorts = {"baseline (first visit)": df[(df.NACCVNUM == 1) & df.eligible],
               "longitudinal (third visit)": df[(df.NACCVNUM == 3) & df.eligible],
               "PET strict (first visit on or after scan)": df[df.is_pet_index],
               "PET nearest (visit closest to scan)": df[df.is_pet_index_nearest]}
    table = {}
    for name, c in cohorts.items():
        for target in ["y_dementia_3y", "y_dementia_2y", "y_cdrsb_rise_3y"]:
            k = c[c[target].notna()]
            table[f"{name} | {target}"] = {
                s: {"n": int((k.split == s).sum()), "events": int(k.loc[k.split == s, target].sum())}
                for s in ["train", "val", "test", "external"]}
    summary = {
        "seed": SEED, "splits_participants": splits.split.value_counts().to_dict(),
        "external_centres": int(first.loc[splits.set_index("NACCID").split == "external", "NACCADC"].nunique()),
        "all_centres": int(first.NACCADC.nunique()), "flow_baseline": flow, "cohorts": table,
        "pet_index": {"participants": int(df.is_pet_index.sum()), "with_amyloid": int(df.loc[df.is_pet_index, "has_amyloid"].sum()),
                      "with_tau": int(df.loc[df.is_pet_index, "has_tau"].sum()),
                      "with_image": int(df.loc[df.is_pet_index, "has_pet_image"].sum())},
        "pet_index_nearest": {"participants": int(df.is_pet_index_nearest.sum()),
                              "with_amyloid": int(df.loc[df.is_pet_index_nearest, "has_amyloid_nearest"].sum()),
                              "with_tau": int(df.loc[df.is_pet_index_nearest, "has_tau_nearest"].sum()),
                              "scan_after_visit_pct": float((df.loc[df.is_pet_index_nearest, "amyloid_gap_days_nearest"] > 0).mean() * 100),
                              "median_days_scan_after_visit": float(df.loc[df.is_pet_index_nearest, "amyloid_gap_days_nearest"].median())},
        "cog_norm_reference_n": {t: n["n"] for t, n in norms.items()},
    }
    text = json.dumps(summary, indent=1)
    nu.assert_no_ids(text, "cohort summary")
    (nu.REPORTS / "phase2_cohort_summary.json").write_text(text)
    log("wrote data/processed/visits.parquet and reports/phase2_cohort_summary.json")
    print(text)


if __name__ == "__main__":
    main()
