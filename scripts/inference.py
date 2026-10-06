"""Turn what a clinician enters into the model's features, a risk and an explanation.

Used by the interface (app/app.py). It needs only the files in models/ and
never touches NACC data.

Anything the clinician did not provide is passed to the model as "missing".
The model was trained with whole groups blanked at random (see
train_final.py), so it gives a sensible, less certain estimate in that case.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shap
from xgboost import XGBClassifier

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Raw test -> (domain, label, lowest, highest, higher_is_worse)
TESTS = {
    "MOCATOTS": ("global", "MoCA total (0-30)", 0, 30, False),
    "NACCMMSE": ("global", "MMSE total (0-30)", 0, 30, False),
    "CRAFTVRS": ("memory", "Craft Story immediate, verbatim (0-44)", 0, 44, False),
    "CRAFTDVR": ("memory", "Craft Story delayed, verbatim (0-44)", 0, 44, False),
    "LOGIMEM": ("memory", "Logical Memory immediate (0-25)", 0, 25, False),
    "MEMUNITS": ("memory", "Logical Memory delayed (0-25)", 0, 25, False),
    "UDSBENTD": ("memory", "Benson figure delayed recall (0-17)", 0, 17, False),
    "DIGFORCT": ("attention", "Number span forward, trials correct (0-14)", 0, 14, False),
    "DIGBACCT": ("attention", "Number span backward, trials correct (0-14)", 0, 14, False),
    "TRAILA": ("executive", "Trail Making A, seconds (0-150)", 0, 150, True),
    "TRAILB": ("executive", "Trail Making B, seconds (0-300)", 0, 300, True),
    "ANIMALS": ("language", "Animals named in 60 s", 0, 77, False),
    "VEG": ("language", "Vegetables named in 60 s", 0, 77, False),
    "MINTTOTS": ("language", "MINT naming total (0-32)", 0, 32, False),
    "UDSVERFC": ("language", "F-words in 60 s", 0, 40, False),
    "UDSBENTC": ("visuospatial", "Benson figure copy (0-17)", 0, 17, False),
}
DOMAINS = ["memory", "attention", "executive", "language", "visuospatial", "global"]
HISTORY = {"htn_ever": "Hypertension", "chol_ever": "High cholesterol", "diabetes_ever": "Diabetes", "mi_ever": "Heart attack",
           "afib_ever": "Atrial fibrillation", "chf_ever": "Heart failure", "revasc_ever": "Angioplasty or stent", "bypass_ever": "Bypass surgery",
           "pacemaker_ever": "Pacemaker", "othercvd_ever": "Other heart disease", "stroke_ever": "Stroke", "tia_ever": "TIA"}
MEDS = {"med_bp": "Blood-pressure medication", "med_lipid": "Lipid-lowering medication", "med_diabetes": "Diabetes medication",
        "med_anticoag": "Anticoagulant or antiplatelet"}


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.meta = json.loads((models_dir / "meta.json").read_text())
        self.norms = json.loads((models_dir / "cog_norms.json").read_text())
        self.tau_norms = json.loads((models_dir / "tau_norms.json").read_text())
        self.features = self.meta["features"]
        self.models = []
        for path in sorted(models_dir.glob("clinical_xgb_*.json")):
            m = XGBClassifier()
            m.load_model(path)
            self.models.append(m)
        self.explainers = [shap.TreeExplainer(m) for m in self.models]

    # ---- features ---------------------------------------------------------
    def z_score(self, test: str, score: float, age: float, female: float, educ: float) -> float:
        n = self.norms[test]
        z = (score - (n["intercept"] + n["age"] * age + n["female"] * female + n["educ_years"] * educ)) / n["resid_sd"]
        return float(np.clip(-z if TESTS[test][4] else z, -5, 5))

    def build_features(self, x: dict) -> dict:
        """x holds only what was entered; anything absent stays missing (NaN)."""
        f = {k: np.nan for k in self.features}
        for k in ["age", "female", "educ_years", "hispanic", "lives_alone"]:
            f[k] = x.get(k, np.nan)
        if x.get("race") is not None:
            for name in ["white", "black", "asian"]:
                f[f"race_{name}"] = float(x["race"] == name)
        if "apoe_e4_count" in x:                         # APOE section ticked
            e4 = x["apoe_e4_count"]
            f["apoe_e4_count"], f["apoe_unknown"] = (np.nan, 1.0) if e4 is None else (float(e4), 0.0)
            f["family_history"] = x.get("family_history", np.nan)
        for k in [*HISTORY, *MEDS, "smoke_ever", "smoke_years", "bp_sys", "bp_dia", "heart_rate", "bmi", "med_count"]:
            if k in x:
                f[k] = float(x[k])
        if "bp_sys" in x and "bp_dia" in x:
            f["pulse_pressure"] = x["bp_sys"] - x["bp_dia"]
        risk = [x.get(k) for k in ["htn_ever", "chol_ever", "diabetes_ever", "smoke_ever"]]
        if sum(v is not None for v in risk) >= 3:
            f["vascular_risk_count"] = float(sum(v for v in risk if v is not None))
        tests = x.get("tests", {})
        if tests and all(k in x for k in ["age", "female", "educ_years"]):
            by_domain: dict = {}
            for test, score in tests.items():
                by_domain.setdefault(TESTS[test][0], []).append(self.z_score(test, score, x["age"], x["female"], x["educ_years"]))
            for dom, zs in by_domain.items():
                f[f"z_{dom}"] = float(np.mean(zs))
        for k in ["tests_failed_cognitive", "gds", "npi_count", "CDRSUM"]:
            if k in x:
                f[k] = float(x[k])
        if "diagnosis" in x:
            f["is_mci"], f["is_impaired_not_mci"] = float(x["diagnosis"] == "MCI"), float(x["diagnosis"] == "Impaired, not MCI")
        return f

    # ---- prediction -------------------------------------------------------
    def predict(self, x: dict) -> dict:
        f = self.build_features(x)
        X = np.array([[f[k] for k in self.features]], dtype=np.float32)
        p = float(np.mean([m.predict_proba(X)[0, 1] for m in self.models]))
        sv = np.mean([e.shap_values(X)[0] for e in self.explainers], axis=0)        # log-odds contributions
        group_of = {feat: g for g, feats in self.meta["feature_groups"].items() for feat in feats}
        groups: dict = {}
        for feat, v in zip(self.features, sv):
            groups[group_of[feat]] = groups.get(group_of[feat], 0.0) + float(v)
        out = {"risk_clinical": p, "features": f, "shap_by_feature": dict(zip(self.features, map(float, sv))), "shap_by_group": groups,
               "missing_groups": [g for g, feats in self.meta["feature_groups"].items() if all(np.isnan(f[k]) for k in feats)]}
        pct = self.meta["reference"]["risk_percentiles"]
        out["percentile"] = int(np.searchsorted(pct, p) * 5)
        out["risk"] = p
        if x.get("pet"):
            out["risk"] = out["risk_with_pet"] = self._fuse(p, x["pet"])
        return out

    def _fuse(self, p: float, pet: dict) -> float:
        """Combine the clinical risk with PET measures (small logistic model)."""
        fu = self.meta["pet_fusion"]
        vals = {"clin_logit": float(np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5))))}
        vals["amyloid_centiloids"] = pet.get("centiloids", fu["median"]["amyloid_centiloids"])
        status = pet.get("amyloid_positive")
        vals["amyloid_amyloid_status"] = fu["median"]["amyloid_amyloid_status"] if status is None else float(status)
        tracer = pet.get("tau_tracer")
        for col, key in [("tau_meta_temporal_suvr_z", "META_TEMPORAL_SUVR"), ("tau_ctx_entorhinal_suvr_z", "CTX_ENTORHINAL_SUVR")]:
            suvr = pet.get(key)
            if suvr is not None and tracer in self.tau_norms[key]:
                n = self.tau_norms[key][tracer]
                vals[col], vals[col + "_missing"] = (suvr - n["mean"]) / n["sd"], 0.0
            else:
                vals[col], vals[col + "_missing"] = fu["median"][col], 1.0
        v = (np.array([vals[c] for c in fu["columns"]]) - np.array(fu["mean"])) / np.array(fu["sd"])
        return float(1 / (1 + np.exp(-(fu["intercept"] + v @ np.array(fu["coef"])))))
