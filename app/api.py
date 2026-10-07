"""Thin API between the React screen (app/web) and the trained model.

Run:  python -m uvicorn app.api:app --port 8000
      then open http://localhost:8000  (after `npm run build` in app/web)

GET  /api/schema    the input form, generated from the features the model was trained on
POST /api/predict   entries -> risk, plausibility checks, comparison group, explanation
POST /api/ask       a question for the evidence library

Every input on the screen feeds one of the model's trained features; the
schema says which. Nothing is stored and nothing leaves this machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inference import DOMAINS, HISTORY, LABELS, MEDS, TESTS, Predictor  # noqa: E402

WEB = Path(__file__).resolve().parent / "web" / "dist"
P = Predictor()
REF = P.meta["reference"]
GROUP_NAMES = {"Demographics": "About the person", "Genetics / APOE": "Genes", "Cardiovascular / medical": "Heart and blood vessels",
               "Cognitive tests": "Test scores", "Clinical stage (CDR-SB, diagnosis)": "Clinician's assessment"}
DOMAIN_TITLES = {"global": "Screening", "memory": "Memory", "attention": "Attention", "executive": "Speed and planning",
                 "language": "Language", "visuospatial": "Drawing"}
YES_NO = ["Not known", "No", "Yes"]


def num(id, label, lo, hi, feeds, step=1, default=None, placeholder="not known", more=False, group=None):
    return {"id": id, "label": label, "type": "number", "min": lo, "max": hi, "step": step, "default": default,
            "placeholder": placeholder, "feeds": feeds, "more": more, "group": group}


def choice(id, label, options, feeds, default=None, more=False):
    return {"id": id, "label": label, "type": "select", "options": options, "default": default or options[0], "feeds": feeds, "more": more}


def tick(id, label, feeds, more=False, group=None):
    return {"id": id, "label": label, "type": "checkbox", "default": False, "feeds": feeds, "more": more, "group": group}


def build_schema() -> dict:
    tests = [num(code, label, lo, hi, [f"z_{domain}"], placeholder="not done", more=code not in ("MOCATOTS", "NACCMMSE"),
                 group=DOMAIN_TITLES[domain]) for code, (domain, label, lo, hi, _) in TESTS.items()]
    tests.sort(key=lambda t: list(DOMAIN_TITLES.values()).index(t["group"]))
    sections = [
        {"id": "about", "title": "About the person", "note": "Always needed.", "required": True, "model_groups": ["Demographics"], "fields": [
            num("age", "Age (years)", 40, 105, ["age"], default=72),
            choice("sex", "Sex", ["Female", "Male"], ["female"]),
            num("educ_years", "Years of education", 0, 30, ["educ_years"], default=16),
            choice("race", "Race", ["Not stated", "White", "Black", "Asian", "Other"], ["race_white", "race_black", "race_asian"], more=True),
            choice("hispanic", "Hispanic / Latino", YES_NO, ["hispanic"], more=True),
            choice("lives_alone", "Lives alone", YES_NO, ["lives_alone"], more=True)],
         "more_note": "In the research cohort, conversion rates differed between these groups, largely because of how volunteers were "
                      "recruited. They describe the cohort, not a biological effect. 'Not stated' is fine."},
        {"id": "heart", "title": "Heart and blood vessels", "note": "Tick every condition ever diagnosed. Unticked means no.",
         "default_on": True, "model_groups": ["Cardiovascular / medical"], "fields": [
            *[tick(k, label, [k] + (["vascular_risk_count"] if k in ("htn_ever", "chol_ever", "diabetes_ever") else []), group="History")
              for k, label in HISTORY.items()],
            tick("smoke_ever", "Ever smoked (100+ cigarettes)", ["smoke_ever", "vascular_risk_count"], group="Smoking"),
            num("smoke_years", "Years smoked", 0, 80, ["smoke_years"], group="Smoking"),
            *[tick(k, label, [k], more=True, group="Medication (tick only if the list is known)") for k, label in MEDS.items()],
            num("med_count", "Total number of medications", 0, 40, ["med_count"], more=True, group="Medication (tick only if the list is known)"),
            num("bp_sys", "Systolic BP (mmHg)", 70, 230, ["bp_sys", "pulse_pressure"], placeholder="not measured", more=True, group="Measurements"),
            num("bp_dia", "Diastolic BP (mmHg)", 30, 140, ["bp_dia", "pulse_pressure"], placeholder="not measured", more=True, group="Measurements"),
            num("heart_rate", "Heart rate (bpm)", 33, 160, ["heart_rate"], placeholder="not measured", more=True, group="Measurements"),
            num("bmi", "BMI", 10, 80, ["bmi"], step=0.5, placeholder="not measured", more=True, group="Measurements")]},
        {"id": "assessment", "title": "Thinking and memory: clinician's assessment",
         "note": "The model relies on this more than on anything else. It applies only to people without dementia today.",
         "default_on": True, "model_groups": ["Clinical stage (CDR-SB, diagnosis)"], "fields": [
            choice("diagnosis", "Cognitive status today", ["Not assessed", "Normal", "Impaired, not MCI", "MCI"], ["is_mci", "is_impaired_not_mci"],
                   default="Normal"),
            num("CDRSUM", "CDR Sum of Boxes (0 to 18)", 0, 18, ["CDRSUM"], step=0.5, default=0, placeholder="not rated")]},
        {"id": "tests", "title": "Thinking and memory: test scores",
         "note": "Enter only the tests that were done. Each score is compared with healthy people of the same age, sex and education.",
         "default_on": True, "model_groups": ["Cognitive tests"], "fields": [
            *tests,
            num("gds", "Geriatric Depression Scale (0-15)", 0, 15, ["gds"], more=True, group="Mood and behaviour"),
            num("npi_count", "Behavioural symptoms present (NPI-Q, 0-12)", 0, 12, ["npi_count"], more=True, group="Mood and behaviour"),
            num("tests_failed_cognitive", "Tests not completed for cognitive reasons", 0, 20, ["tests_failed_cognitive"], more=True,
                group="Mood and behaviour")]},
        {"id": "genes", "title": "Genes", "note": "APOE result and family history.", "default_on": False,
         "model_groups": ["Genetics / APOE"], "fields": [
            choice("apoe", "APOE e4 copies", ["Not tested", "0", "1", "2"], ["apoe_e4_count", "apoe_unknown"]),
            choice("family_history", "Parent or sibling with cognitive impairment", YES_NO, ["family_history"])]},
        {"id": "pet", "title": "Brain scan (PET)", "default_on": False, "model_groups": [],
         "note": "Numbers from a PET report. They adjust the estimate through a small second model; in testing they did not measurably "
                 "improve accuracy.", "fields": [
            num("centiloids", "Amyloid PET: Centiloids", -60, 250, ["amyloid_centiloids"], default=10),
            choice("amyloid_status", "Amyloid status in the report", ["Negative", "Positive"], ["amyloid_amyloid_status"]),
            choice("tau_tracer", "Tau PET tracer", ["No tau scan", "Flortaucipir", "MK-6240"], ["tau_meta_temporal_suvr_z", "tau_ctx_entorhinal_suvr_z"]),
            num("META_TEMPORAL_SUVR", "Tau: meta-temporal SUVR", 0.5, 5, ["tau_meta_temporal_suvr_z"], step=0.05, placeholder="no tau scan"),
            num("CTX_ENTORHINAL_SUVR", "Tau: entorhinal SUVR", 0.5, 5, ["tau_ctx_entorhinal_suvr_z"], step=0.05, placeholder="no tau scan")]},
    ]
    return {"sections": sections, "model_features": P.features, "pet_features": P.meta["pet_fusion"]["columns"][1:5],
            "reference": {"event_rate": REF["event_rate"], "n_train": REF["n_train"]}}


SCHEMA = build_schema()


def to_model_input(v: dict, on: dict) -> dict:
    """Screen entries -> the dictionary Predictor expects. Empty or switched-off entries are left out."""
    has = lambda k: v.get(k) is not None and v.get(k) != ""
    x = {"age": float(v["age"]), "female": float(v["sex"] == "Female"), "educ_years": float(v["educ_years"])}
    x["race"] = None if v.get("race", "Not stated") == "Not stated" else v["race"].lower()
    for k in ["hispanic", "lives_alone"]:
        if v.get(k) in ("Yes", "No"):
            x[k] = float(v[k] == "Yes")
    if on.get("heart"):
        for k in HISTORY:
            x[k] = float(bool(v.get(k)))
        x["smoke_ever"] = float(bool(v.get("smoke_ever")))
        if not x["smoke_ever"]:
            x["smoke_years"] = 0.0
        elif has("smoke_years"):
            x["smoke_years"] = float(v["smoke_years"])
        if any(v.get(k) for k in MEDS) or has("med_count"):          # medication list known
            for k in MEDS:
                x[k] = float(bool(v.get(k)))
        for k in ["med_count", "bp_sys", "bp_dia", "heart_rate", "bmi"]:
            if has(k):
                x[k] = float(v[k])
    if on.get("assessment"):
        if v.get("diagnosis") in ("Normal", "Impaired, not MCI", "MCI"):
            x["diagnosis"] = v["diagnosis"]
        if has("CDRSUM"):
            x["CDRSUM"] = float(v["CDRSUM"])
    if on.get("tests"):
        x["tests"] = {t: float(v[t]) for t in TESTS if has(t)}
        for k in ["gds", "npi_count", "tests_failed_cognitive"]:
            if has(k):
                x[k] = float(v[k])
    if on.get("genes"):
        x["apoe_e4_count"] = None if v.get("apoe", "Not tested") == "Not tested" else int(v["apoe"])
        if v.get("family_history") in ("Yes", "No"):
            x["family_history"] = float(v["family_history"] == "Yes")
    if on.get("pet") and has("centiloids"):
        pet = {"centiloids": float(v["centiloids"]), "amyloid_positive": float(v.get("amyloid_status") == "Positive")}
        if v.get("tau_tracer") in ("Flortaucipir", "MK-6240"):
            pet["tau_tracer"] = v["tau_tracer"]
            for k in ["META_TEMPORAL_SUVR", "CTX_ENTORHINAL_SUVR"]:
                if has(k):
                    pet[k] = float(v[k])
        x["pet"] = pet
    return x


class Entries(BaseModel):
    values: dict
    on: dict = {}


class Question(BaseModel):
    question: str
    context: str = ""


app = FastAPI(title="CardioCog research prototype")


@app.get("/api/schema")
def schema():
    return SCHEMA


@app.post("/api/predict")
def predict(e: Entries):
    x = to_model_input(e.values, e.on)
    res = P.predict(x)
    f = res["features"]
    items = sorted(((k, v) for k, v in res["shap_by_feature"].items() if not np.isnan(f[k]) and abs(v) > 0.02), key=lambda kv: -abs(kv[1]))[:8]
    domains = {d: round(f[f"z_{d}"], 2) for d in DOMAINS if not np.isnan(f[f"z_{d}"])}
    return {
        "risk": res["risk"], "risk_without_pet": res["risk_clinical"] if "risk_with_pet" in res else None,
        "checks": [{"level": level, "message": m} for level, m in P.check_inputs(x, f)],
        "similar": P.similar_participants(x), "cohort_rate": REF["event_rate"],
        "used": [GROUP_NAMES[g] for g in P.meta["feature_groups"] if g not in res["missing_groups"]] + (["Brain scan"] if "pet" in x else []),
        "missing": [GROUP_NAMES[g] for g in res["missing_groups"]],
        "no_cognitive_information": {"Cognitive tests", "Clinical stage (CDR-SB, diagnosis)"} <= set(res["missing_groups"]),
        "groups": [{"name": GROUP_NAMES[g], "effect": v} for g, v in res["shap_by_group"].items() if g not in res["missing_groups"]],
        "items": [{"name": LABELS.get(k, k), "value": round(float(f[k]), 2), "effect": round(v, 2)} for k, v in items],
        "domain_scores": [{"name": DOMAIN_TITLES[d], "z": z} for d, z in domains.items()],
        "typical_scores": {t: round(P.expected_score(t, x["age"], x["female"], x["educ_years"])) for t in TESTS},
        "features_filled": int(sum(not np.isnan(v) for v in f.values())), "features_total": len(f),
    }


_assistant = None


@app.post("/api/ask")
def ask(q: Question):
    global _assistant
    import rag
    if not (rag.STORE / "passages.json").exists():
        return {"answer": "The evidence library has not been built yet. Run python scripts/rag_build.py.", "mode": "unavailable", "sources": []}
    if _assistant is None:
        _assistant = rag.Assistant()
    out = _assistant.ask(q.question, model_context=q.context)
    return {"answer": out["answer"], "mode": out["mode"],
            "sources": [{k: s[k] for k in ("n", "short", "year", "section", "title", "url", "passage")} for s in out["sources"]]}


if WEB.exists():
    app.mount("/assets", StaticFiles(directory=WEB / "assets"), name="assets")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")
