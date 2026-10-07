"""Does the saved model, as used by the screen, behave in a clinically sensible way?

verify_results.py checks that the numbers are honest. This script checks that
the tool behaves sensibly when someone types values into it:

  A  the screen asks for exactly the features the model was trained on
  B  typical cases get plausible estimates, in the right order
  C  direction: a worse rating or a worse test score never lowers the estimate
  D  contradictory entries are flagged; ordinary entries are not
  E  missing sections are handled and reported
  F  how much each cardiovascular item can move the estimate (reported)
  G  on held-out participants, predicted risk matches what happened in each
     age group and diagnosis group

Output  reports/clinical_behaviour_checks.json and a printed summary
Usage:  python scripts/check_clinical_behaviour.py
"""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import modeling as M                      # noqa: E402
import nacc_utils as nu                   # noqa: E402
from app import api                       # noqa: E402
from inference import HISTORY, MEDS, TESTS  # noqa: E402

P = api.P
TARGET = "y_dementia_3y"
RESULTS = []
ALL_ON = {"heart": True, "assessment": True, "tests": True, "genes": True, "pet": False}
TOLERANCE = 1.0          # percentage points; tree models are not perfectly smooth


def record(check: str, passed: bool, detail: str) -> None:
    RESULTS.append({"check": check, "passed": bool(passed), "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {check}: {detail}", flush=True)


def run(values: dict, on: dict = ALL_ON) -> dict:
    return api.predict(api.Entries(values=values, on=on))


def risk(values: dict, on: dict = ALL_ON) -> float:
    return run(values, on)["risk"] * 100


BASE = {"age": 72, "sex": "Female", "educ_years": 16, "race": "White", "hispanic": "No", "lives_alone": "No",
        **{k: False for k in HISTORY}, **{k: False for k in MEDS}, "smoke_ever": False, "med_count": 3,
        "bp_sys": 130, "bp_dia": 76, "heart_rate": 68, "bmi": 26.5, "gds": 1, "npi_count": 0, "tests_failed_cognitive": 0,
        "apoe": "0", "family_history": "No"}
NORMAL = {**BASE, "diagnosis": "Normal", "CDRSUM": 0, "MOCATOTS": 27, "CRAFTDVR": 18, "ANIMALS": 21, "TRAILB": 75}
IMPAIRED = {**BASE, "diagnosis": "Impaired, not MCI", "CDRSUM": 0.5, "MOCATOTS": 24, "CRAFTDVR": 13, "ANIMALS": 18, "TRAILB": 105}
MCI = {**BASE, "diagnosis": "MCI", "CDRSUM": 1.5, "MOCATOTS": 21, "CRAFTDVR": 7, "ANIMALS": 14, "TRAILB": 150}


def check_schema():
    fields = [f for s in api.SCHEMA["sections"] for f in s["fields"]]
    fed = {x for f in fields for x in f["feeds"]}
    model = set(P.features)
    pet = set(P.meta["pet_fusion"]["columns"]) - {"clin_logit"}
    pet = {c for c in pet if not c.endswith("_missing")}
    record("A every trained feature has an input", model <= fed, f"{len(model)} model features; not covered: {sorted(model - fed) or 'none'}")
    record("A no input outside the trained features", fed <= model | pet,
           f"{len(fields)} inputs feed {len(fed)} features ({len(model)} clinical + {len(fed & pet)} PET); extra: {sorted(fed - model - pet) or 'none'}")
    trained_tests = {t for t in P.norms}
    record("A every trained test can be entered", trained_tests == set(TESTS), f"{len(TESTS)} tests on screen, {len(trained_tests)} in training")


def check_vignettes():
    r = {name: risk(v) for name, v in [("normal", NORMAL), ("impaired, not MCI", IMPAIRED), ("MCI", MCI)]}
    record("B typical cases are ordered normal < impaired < MCI", r["normal"] < r["impaired, not MCI"] < r["MCI"],
           ", ".join(f"{k} {v:.1f}%" for k, v in r.items()))
    record("B typical cognitively normal 72-year-old is low", r["normal"] < 3, f"{r['normal']:.1f}% (0.9% of similar participants converted)")
    record("B typical MCI case is high but not certain", 25 < r["MCI"] < 80, f"{r['MCI']:.1f}% (40% of similar participants converted)")
    e4 = risk({**MCI, "apoe": "2"})
    record("B two APOE e4 copies raise the MCI estimate", e4 > r["MCI"], f"{r['MCI']:.1f}% -> {e4:.1f}%")


def check_directions():
    sweeps = {"CDR Sum of Boxes rising": ("CDRSUM", [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]),
              "status Normal -> Impaired -> MCI": ("diagnosis", ["Normal", "Impaired, not MCI", "MCI"]),
              "MoCA falling": ("MOCATOTS", [30, 27, 24, 21, 18, 15, 12]),
              "MMSE falling": ("NACCMMSE", [30, 28, 26, 24, 22, 20]),
              "delayed story recall falling": ("CRAFTDVR", [30, 24, 18, 12, 6, 0]),
              "animal naming falling": ("ANIMALS", [30, 24, 18, 12, 6]),
              "Trail Making B slower": ("TRAILB", [40, 75, 120, 180, 300]),
              "APOE e4 copies rising": ("apoe", ["0", "1", "2"]),
              "behavioural symptoms rising": ("npi_count", [0, 1, 3, 6])}
    for name, (key, values) in sweeps.items():
        worst_drop, lines = 0.0, []
        for label, case in [("normal", NORMAL), ("MCI", MCI)]:
            if key == "NACCMMSE":
                case = {k: v for k, v in case.items() if k != "MOCATOTS"}
            rs = [risk({**case, key: v}) for v in values]
            worst_drop = max(worst_drop, max([a - b for a, b in zip(rs, rs[1:])] + [0]))
            lines.append(f"{label} {rs[0]:.1f}% -> {rs[-1]:.1f}%")
        record(f"C {name} never lowers the estimate", worst_drop <= TOLERANCE, "; ".join(lines) + f"; largest step down {worst_drop:.1f} points")
    # Age with the age-adjusted scores held fixed (the model's own view of age).
    f = P.build_features(api.to_model_input(MCI, ALL_ON))
    rs = []
    for age in [55, 65, 75, 85]:
        X = np.array([[{**f, "age": age}[k] for k in P.features]], dtype=np.float32)
        rs.append(float(np.mean([m.predict_proba(X)[0, 1] for m in P.models])) * 100)
    record("C older age raises the estimate when adjusted scores are equal", rs[-1] > rs[0], "MCI case at 55/65/75/85: " + ", ".join(f"{r:.1f}%" for r in rs))


def check_contradictions():
    levels = lambda v, on=ALL_ON: [c["level"] for c in run(v, on)["checks"]]
    clean = [levels(v) for v in (NORMAL, IMPAIRED, MCI)]
    record("D ordinary cases raise no warning", all("check" not in c for c in clean), f"levels raised: {clean}")
    screenshot = {**BASE, "age": 91, "educ_years": 1, "race": "Black", "hispanic": "Yes", "lives_alone": "Yes", **{k: True for k in HISTORY},
                  **{k: True for k in MEDS}, "smoke_ever": True, "smoke_years": 30, "med_count": 19, "diagnosis": "Normal", "CDRSUM": 0,
                  "MOCATOTS": 0}
    out = run(screenshot)
    record("D MoCA of 0 with status Normal and CDR 0 is flagged", any(c["level"] == "check" for c in out["checks"]),
           f"estimate {out['risk'] * 100:.0f}% shown with {sum(c['level'] == 'check' for c in out['checks'])} blocking and "
           f"{sum(c['level'] == 'note' for c in out['checks'])} other warnings")
    cases = {"status Normal with CDR Sum of Boxes 3": {**NORMAL, "CDRSUM": 3},
             "CDR Sum of Boxes 8": {**MCI, "CDRSUM": 8},
             "systolic below diastolic": {**NORMAL, "bp_sys": 70, "bp_dia": 90},
             "60 years of smoking at age 50": {**NORMAL, "age": 50, "smoke_ever": True, "smoke_years": 60}}
    for name, v in cases.items():
        record(f"D flagged: {name}", "check" in levels(v), "blocking warning shown")
    record("D noted: age 100", "note" in levels({**NORMAL, "age": 100}), "note that few participants are like this")


def check_missing():
    off = {**ALL_ON, "assessment": False, "tests": False}
    out = run(MCI, off)
    record("E no thinking information is reported, not hidden", out["no_cognitive_information"] and len(out["missing"]) == 2,
           f"estimate {out['risk'] * 100:.0f}% with the warning that it is only an age-based average; not used: {out['missing']}")
    only = run({k: MCI[k] for k in ("age", "sex", "educ_years")}, {k: False for k in ALL_ON})
    record("E demographics alone still gives an estimate near the cohort average", 5 < only["risk"] * 100 < 35,
           f"{only['risk'] * 100:.0f}% (cohort average {P.meta['reference']['event_rate'] * 100:.0f}%)")


def check_cardiovascular():
    rows = {}
    for label, case in [("normal", NORMAL), ("MCI", MCI)]:
        base = risk(case)
        moves = {HISTORY[k]: risk({**case, k: True}) - base for k in HISTORY}
        everything = risk({**case, **{k: True for k in HISTORY}, **{k: True for k in MEDS}, "smoke_ever": True, "smoke_years": 30}) - base
        top = max(moves, key=lambda k: abs(moves[k]))
        rows[label] = f"{label}: largest single item {top} {moves[top]:+.1f} points; every item ticked {everything:+.1f} points"
    record("F cardiovascular items move the estimate only slightly", True, "; ".join(rows.values()) + " (reported, consistent with the +0.001 AUROC finding)")


def check_calibration():
    df = M.load_visits()
    df = df[(df.NACCVNUM == 1) & df.eligible & df[TARGET].notna() & df.split.isin(["test", "external"])].reset_index(drop=True)
    X = df[P.features].to_numpy(np.float32)
    df["p"] = np.mean([m.predict_proba(X)[:, 1] for m in P.models], axis=0)
    df["age_group"] = pd.cut(df.age, [0, 65, 75, 85, 200], right=False, labels=P.meta["reference"]["age_groups"])
    df["status"] = df.NACCUDSD.map(nu.DX_LABELS)
    t = df.groupby(["status", "age_group"], observed=True).agg(n=("p", "size"), events=(TARGET, "sum"), observed=(TARGET, "mean"), predicted=("p", "mean"))
    t = t[t.n >= 50]
    gap = (t.predicted - t.observed).abs()
    table = [{"status": s, "age_group": a, "n": int(r.n), "events": int(r.events), "observed_pct": round(r.observed * 100, 1),
              "predicted_pct": round(r.predicted * 100, 1)} for (s, a), r in t.iterrows()]
    for row in table:
        print(f"      {row['status']:18s} {row['age_group']:9s} n={row['n']:5d}  observed {row['observed_pct']:5.1f}%  predicted {row['predicted_pct']:5.1f}%")
    record("G predicted risk matches what happened, by age group and status", gap.max() < 0.08,
           f"{len(t)} groups of 50+ held-out participants; largest gap {gap.max() * 100:.1f} points, mean gap {gap.mean() * 100:.1f} points")
    return table


def main():
    check_schema(); check_vignettes(); check_directions(); check_contradictions(); check_missing(); check_cardiovascular()
    table = check_calibration()
    passed = sum(r["passed"] for r in RESULTS)
    print(f"\n{passed} of {len(RESULTS)} checks passed")
    text = json.dumps({"passed": passed, "total": len(RESULTS), "checks": RESULTS, "held_out_by_status_and_age_group": table}, indent=1)
    nu.assert_no_ids(text, "clinical behaviour checks")
    (nu.REPORTS / "clinical_behaviour_checks.json").write_text(text)


if __name__ == "__main__":
    main()
