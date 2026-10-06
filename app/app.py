"""CardioCog research prototype: risk estimate with a modality checklist.

Run:  streamlit run app/app.py
Needs the trained models in models/ (python scripts/train_final.py).

The clinician ticks which kinds of information are available. Only those
sections ask for input; everything else is treated as missing by the model.
No patient data is stored and nothing is sent anywhere.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inference import HISTORY, MEDS, MODELS_DIR, TESTS, Predictor  # noqa: E402

st.set_page_config(page_title="CardioCog research prototype", layout="wide")


@st.cache_resource
def load():
    return Predictor()


if not (MODELS_DIR / "meta.json").exists():
    st.error("No trained models found. Run `python scripts/train_final.py` first.")
    st.stop()
P = load()
ref = P.meta["reference"]

st.title("CardioCog: 3-year dementia risk estimate")
st.warning("**Research prototype. Not for clinical use.** Estimates describe volunteers in the NACC research cohort "
           "(median age 71, mostly White, highly educated). The tool has been tested retrospectively only. "
           "It does not diagnose and does not recommend treatment.")

# ---- modality checklist ----------------------------------------------------
st.sidebar.header("What is available for this person?")
st.sidebar.caption("Tick what you have. Unticked sections are treated as not available.")
has = {
    "history": st.sidebar.checkbox("Medical history and vitals", True),
    "tests": st.sidebar.checkbox("Cognitive test scores", True),
    "stage": st.sidebar.checkbox("CDR and clinical diagnosis", True),
    "apoe": st.sidebar.checkbox("APOE genotype", False),
    "pet": st.sidebar.checkbox("Amyloid / tau PET measures", False),
    "prior": st.sidebar.checkbox("A previous visit to compare with", False),
}
st.sidebar.divider()
st.sidebar.caption(f"Model: gradient-boosted trees trained on {ref['n_train']:,} research participants who were not demented at "
                   "their first visit. Trained with whole sections hidden at random so it can work with any combination.")

x: dict = {}
left, right = st.columns([1.15, 1])

with left:
    st.subheader("Demographics (always required)")
    c = st.columns(4)
    x["age"] = c[0].number_input("Age", 40, 105, 70)
    x["female"] = float(c[1].selectbox("Sex", ["Female", "Male"]) == "Female")
    x["educ_years"] = c[2].number_input("Years of education", 0, 30, 14)
    race = c[3].selectbox("Race", ["White", "Black", "Asian", "Other / prefer not to say"])
    x["race"] = race.lower() if race in ("White", "Black", "Asian") else "other"
    c = st.columns(4)
    x["hispanic"] = float(c[0].checkbox("Hispanic / Latino"))
    x["lives_alone"] = float(c[1].checkbox("Lives alone"))

    if has["history"]:
        with st.expander("Medical history and vitals", expanded=True):
            st.caption("Tick every condition the person has ever been diagnosed with.")
            cols = st.columns(2)
            for i, (key, label) in enumerate(HISTORY.items()):
                x[key] = float(cols[i % 2].checkbox(label, key=key))
            st.caption("Current medication")
            cols = st.columns(2)
            for i, (key, label) in enumerate(MEDS.items()):
                x[key] = float(cols[i % 2].checkbox(label, key=key))
            c = st.columns(3)
            x["smoke_ever"] = float(c[0].checkbox("Ever smoked (100+ cigarettes)"))
            x["smoke_years"] = c[1].number_input("Years smoked", 0, 80, 0) if x["smoke_ever"] else 0
            x["med_count"] = c[2].number_input("Number of medications", 0, 40, 3)
            c = st.columns(4)
            x["bp_sys"] = c[0].number_input("Systolic BP (mmHg)", 70, 230, 130)
            x["bp_dia"] = c[1].number_input("Diastolic BP (mmHg)", 30, 140, 76)
            x["heart_rate"] = c[2].number_input("Heart rate (bpm)", 33, 160, 68)
            x["bmi"] = c[3].number_input("BMI", 10.0, 80.0, 26.5, step=0.5)

    if has["tests"]:
        with st.expander("Cognitive test scores", expanded=True):
            st.caption("Enter the tests that were done; leave the rest unticked. Scores are compared with healthy people of the same "
                       "age, sex and education.")
            x["tests"] = {}
            cols = st.columns(2)
            for i, (code, (domain, label, lo, hi, _)) in enumerate(TESTS.items()):
                col = cols[i % 2]
                if col.checkbox(f"{label}", key=f"has_{code}"):
                    x["tests"][code] = col.number_input(f"Score: {label}", lo, hi, lo, key=code, label_visibility="collapsed")
            c = st.columns(3)
            x["tests_failed_cognitive"] = c[0].number_input("Tests not completed for cognitive reasons", 0, 16, 0)
            x["gds"] = c[1].number_input("Geriatric Depression Scale (0-15)", 0, 15, 1)
            x["npi_count"] = c[2].number_input("Behavioural symptoms present (NPI-Q, 0-12)", 0, 12, 0)

    if has["stage"]:
        with st.expander("CDR and clinical diagnosis", expanded=True):
            c = st.columns(2)
            x["CDRSUM"] = c[0].number_input("CDR Sum of Boxes (0-18)", 0.0, 18.0, 0.0, step=0.5)
            x["diagnosis"] = c[1].selectbox("Current cognitive status", ["Normal", "Impaired, not MCI", "MCI"])
            st.caption("The model applies to people who do not have dementia at this visit.")

    if has["apoe"]:
        with st.expander("APOE genotype", expanded=True):
            c = st.columns(2)
            x["apoe_e4_count"] = c[0].selectbox("Number of e4 alleles", [0, 1, 2])
            x["family_history"] = float(c[1].checkbox("First-degree relative with cognitive impairment"))

    if has["pet"]:
        with st.expander("PET measures", expanded=True):
            st.caption("Values as reported by a PET quantification pipeline.")
            pet = {}
            c = st.columns(2)
            pet["centiloids"] = c[0].number_input("Amyloid PET: Centiloids", -60.0, 250.0, 10.0)
            pet["amyloid_positive"] = float(c[1].selectbox("Amyloid status", ["Negative", "Positive"]) == "Positive")
            if st.checkbox("Tau PET also available"):
                c = st.columns(3)
                pet["tau_tracer"] = c[0].selectbox("Tau tracer", ["Flortaucipir", "MK-6240"])
                pet["META_TEMPORAL_SUVR"] = c[1].number_input("Meta-temporal SUVR", 0.5, 5.0, 1.2, step=0.05)
                pet["CTX_ENTORHINAL_SUVR"] = c[2].number_input("Entorhinal SUVR", 0.5, 5.0, 1.15, step=0.05)
            x["pet"] = pet

    prior = None
    if has["prior"]:
        with st.expander("Previous visit", expanded=True):
            st.caption("Used only to describe change. It does not alter the risk estimate.")
            c = st.columns(3)
            prior = {"months": c[0].number_input("Months since previous visit", 1, 120, 12),
                     "CDRSUM": c[1].number_input("CDR Sum of Boxes then", 0.0, 18.0, 0.0, step=0.5),
                     "MOCATOTS": c[2].number_input("MoCA then (leave 0 if not done)", 0, 30, 0)}

# ---- result ----------------------------------------------------------------
res = P.predict(x)
with right:
    st.subheader("Estimate")
    risk = res["risk"]
    c = st.columns(2)
    c[0].metric("3-year dementia risk", f"{risk * 100:.0f}%")
    c[1].metric("Cohort average", f"{ref['event_rate'] * 100:.0f}%",
                help="Share of all non-demented training participants who were diagnosed with dementia within 3 years.")
    st.caption("Estimated chance of a dementia diagnosis within 3 years, next to the share of all research participants who converted.")
    if "risk_with_pet" in res:
        st.caption(f"Clinical information alone: {res['risk_clinical'] * 100:.0f}%. With PET measures: {res['risk_with_pet'] * 100:.0f}%. "
                   "In testing, adding PET did not measurably improve accuracy, so treat the two as similar.")
    if "diagnosis" in x:
        rate = ref["event_rate_by_dx"].get(x["diagnosis"])
        if rate is not None:
            st.caption(f"For reference: {rate * 100:.0f}% of research participants with status '{x['diagnosis']}' converted within 3 years.")

    used = [g for g in P.meta["feature_groups"] if g not in res["missing_groups"]]
    st.markdown("**Information used:** " + ", ".join(used) + ("; PET measures" if "pet" in x else ""))
    if res["missing_groups"]:
        st.info("**Not available, so not used:** " + ", ".join(res["missing_groups"]) + ". "
                + ("Without any cognitive information the estimate is close to an age-based average and is much less certain."
                   if {"Cognitive tests", "Clinical stage (CDR-SB, diagnosis)"} <= set(res["missing_groups"]) else
                   "The estimate is less certain than with complete information."))

    st.markdown("**What moved this estimate** (by category)")
    g = pd.Series(res["shap_by_group"]).sort_values()
    g = g[[k for k in g.index if k not in res["missing_groups"]]]
    chart = pd.DataFrame({"Raises the estimate": g.clip(lower=0), "Lowers the estimate": g.clip(upper=0)})
    st.bar_chart(chart, horizontal=True, color=["#2a78d6", "#eb6834"], height=60 + 45 * len(g))
    st.caption("Bars show how much each category pushed this estimate up or down compared with an average participant. "
               "They describe what the model used. They are not causes, and changing a factor would not necessarily change the outcome.")

    with st.expander("Individual factors"):
        f = pd.DataFrame({"Value used": res["features"], "Contribution": res["shap_by_feature"]}).dropna(subset=["Value used"])
        f = f[f["Contribution"].abs() > 0.02].sort_values("Contribution", key=np.abs, ascending=False).head(10)
        st.dataframe(f.round(2), use_container_width=True)
        st.caption("Test scores appear as z-scores: 0 is typical for a healthy person of the same age, sex and education; negative is worse.")

    if prior:
        st.markdown("**Change since the previous visit**")
        years = prior["months"] / 12
        if "CDRSUM" in x:
            d = x["CDRSUM"] - prior["CDRSUM"]
            st.write(f"CDR Sum of Boxes: {prior['CDRSUM']:.1f} → {x['CDRSUM']:.1f} "
                     f"({d:+.1f} over {prior['months']} months, {d / years:+.1f} per year).")
        if prior["MOCATOTS"] and "tests" in x and "MOCATOTS" in x["tests"]:
            d = x["tests"]["MOCATOTS"] - prior["MOCATOTS"]
            st.write(f"MoCA: {prior['MOCATOTS']} → {x['tests']['MOCATOTS']} ({d:+d} points).")
        st.caption("A description of change between two visits. In testing, adding past visits changed the risk estimate very little "
                   "once the current scores were known.")

st.divider()
st.caption("How it was tested: AUROC 0.94 on held-out participants and 0.95 on nine held-out research centres (about 0.84 among people "
           "with MCI). Risks were well calibrated internally and slightly high at the held-out centres. Details are in the project reports.")
