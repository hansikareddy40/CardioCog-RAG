"""CardioCog research prototype: 3-year dementia risk estimate from five kinds of information.

Run:  streamlit run app/app.py
Needs the trained models in models/ (python scripts/train_final.py).

The screen follows the five kinds of information in the project:
  1 demographics   2 heart and blood vessels   3 thinking and memory
  4 genes          5 brain scan (PET)
Only section 1 is required. A section that is switched off, or a box that is
left empty, is passed to the model as "not available"; nothing is filled in
with a made-up value. Entries that contradict each other are flagged before
the estimate is shown. No patient data is stored and nothing is sent anywhere.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from inference import HISTORY, LABELS, MEDS, MODELS_DIR, TESTS, Predictor  # noqa: E402

st.set_page_config(page_title="CardioCog research prototype", layout="wide")


@st.cache_resource
def load():
    return Predictor()


if not (MODELS_DIR / "meta.json").exists():
    st.error("No trained models found. Run `python scripts/train_final.py` first.")
    st.stop()
P = load()
ref = P.meta["reference"]
YES_NO = ["Not known", "No", "Yes"]
MAIN_TESTS = ["MOCATOTS", "NACCMMSE"]


def yes_no(col, label, key):
    """Three-way answer; 'Not known' leaves the value out."""
    v = col.selectbox(label, YES_NO, key=key)
    return None if v == "Not known" else float(v == "Yes")


def section(number, title, note, key, default):
    """A bordered block for one kind of information, with its own on/off switch."""
    box = st.container(border=True)
    head = box.columns([3, 1.3])
    head[0].markdown(f"#### {number}. {title}")
    head[0].caption(note)
    on = head[1].toggle("I have this", default, key=key)
    return box, on


st.title("CardioCog: 3-year dementia risk estimate")
st.warning("**Research prototype. Not for clinical use.** For a person who does **not** have dementia today, it estimates the chance of a "
           "dementia diagnosis within 3 years, based on volunteers in the NACC research cohort (median age 71, mostly White, highly "
           "educated). Tested on past research data only. It does not diagnose and does not recommend treatment.")
st.caption("Fill in section 1. Switch on the other sections you have information for and leave the rest off. Empty boxes are treated as "
           "not available.")

x: dict = {}
left, right = st.columns([1.2, 1])

with left:
    # ---- 1 demographics ----------------------------------------------------
    box = st.container(border=True)
    box.markdown("#### 1. About the person")
    box.caption("Always needed.")
    c = box.columns(3)
    x["age"] = c[0].number_input("Age (years)", 40, 105, 72)
    x["female"] = float(c[1].selectbox("Sex", ["Female", "Male"]) == "Female")
    x["educ_years"] = c[2].number_input("Years of education", 0, 30, 16)
    with box.expander("More details (optional)"):
        st.caption("In the research cohort, conversion rates differed between these groups, largely because of how volunteers were "
                   "recruited at each centre. They describe the cohort, not a biological effect. Leaving them as 'Not stated' is fine.")
        c = st.columns(3)
        race = c[0].selectbox("Race", ["Not stated", "White", "Black", "Asian", "Other"])
        x["race"] = None if race == "Not stated" else race.lower()
        for col, key, label in [(c[1], "hispanic", "Hispanic / Latino"), (c[2], "lives_alone", "Lives alone")]:
            v = yes_no(col, label, key)
            if v is not None:
                x[key] = v

    # ---- 2 heart and blood vessels -----------------------------------------
    box, on = section(2, "Heart and blood vessels", "Medical history, medication and measurements.", "has_history", True)
    if on:
        box.caption("Tick every condition the person has ever been diagnosed with. Unticked means 'no'.")
        cols = box.columns(3)
        for i, (key, label) in enumerate(HISTORY.items()):
            x[key] = float(cols[i % 3].checkbox(label, key=key))
        c = box.columns(3)
        x["smoke_ever"] = float(c[0].checkbox("Ever smoked (100+ cigarettes)"))
        if x["smoke_ever"]:
            v = c[1].number_input("Years smoked", 0, 80, None, placeholder="not known")
            if v is not None:
                x["smoke_years"] = v
        else:
            x["smoke_years"] = 0
        with box.expander("Medication and measurements (optional)"):
            if st.toggle("Medication list is known", False):
                cols = st.columns(2)
                for i, (key, label) in enumerate(MEDS.items()):
                    x[key] = float(cols[i % 2].checkbox(label, key=key))
                v = st.number_input("Total number of medications", 0, 40, None, placeholder="not known")
                if v is not None:
                    x["med_count"] = v
            c = st.columns(4)
            for col, key, label, lo, hi in [(c[0], "bp_sys", "Systolic BP (mmHg)", 70, 230), (c[1], "bp_dia", "Diastolic BP (mmHg)", 30, 140),
                                            (c[2], "heart_rate", "Heart rate (bpm)", 33, 160)]:
                v = col.number_input(label, lo, hi, None, placeholder="not measured")
                if v is not None:
                    x[key] = v
            v = c[3].number_input("BMI", 10.0, 80.0, None, step=0.5, placeholder="not measured")
            if v is not None:
                x["bmi"] = v

    # ---- 3 thinking and memory ---------------------------------------------
    box, on = section(3, "Thinking and memory", "The clinician's rating and any test scores. This is what the model relies on most.",
                      "has_cognition", True)
    if on:
        box.markdown("**Clinician's assessment**")
        c = box.columns(2)
        status = c[0].selectbox("Cognitive status today", ["Not assessed", "Normal", "Impaired, not MCI", "MCI"], index=1,
                                help="The model applies only to people who do not have dementia at this visit.")
        if status != "Not assessed":
            x["diagnosis"] = status
        v = c[1].number_input("CDR Sum of Boxes (0 to 18)", 0.0, 18.0, 0.0, step=0.5, placeholder="not rated",
                              help="0 = no impairment in any of the six CDR areas. People without dementia are usually between 0 and 4.")
        if v is not None:
            x["CDRSUM"] = v

        box.markdown("**Test scores** (enter only the tests that were done)")
        x["tests"] = {}

        def test_input(col, code):
            _, label, lo, hi, _ = TESTS[code]
            typical = P.expected_score(code, x["age"], x["female"], x["educ_years"])
            v = col.number_input(label, lo, hi, None, key=code, placeholder="not done",
                                 help=f"Typical for a healthy person of this age, sex and education: about {typical:.0f}.")
            if v is not None:
                x["tests"][code] = v

        c = box.columns(2)
        for col, code in zip(c, MAIN_TESTS):
            test_input(col, code)
        with box.expander("More tests (optional)"):
            cols = st.columns(2)
            for i, code in enumerate(t for t in TESTS if t not in MAIN_TESTS):
                test_input(cols[i % 2], code)
        with box.expander("Mood and behaviour (optional)"):
            c = st.columns(3)
            for col, key, label, hi in [(c[0], "gds", "Geriatric Depression Scale (0-15)", 15),
                                        (c[1], "npi_count", "Behavioural symptoms present (NPI-Q, 0-12)", 12),
                                        (c[2], "tests_failed_cognitive", "Tests not completed for cognitive reasons", 16)]:
                v = col.number_input(label, 0, hi, None, placeholder="not known")
                if v is not None:
                    x[key] = v
        box.caption("Each score is compared with healthy people of the same age, sex and education. So the same raw score counts as "
                    "worse for a younger or more educated person. Hover over the ? next to a test to see the typical score.")

    # ---- 4 genes -----------------------------------------------------------
    box, on = section(4, "Genes", "APOE test result and family history.", "has_apoe", False)
    if on:
        c = box.columns(2)
        e4 = c[0].selectbox("APOE e4 copies", ["Not tested", 0, 1, 2])
        x["apoe_e4_count"] = None if e4 == "Not tested" else e4
        v = yes_no(c[1], "Parent or sibling with cognitive impairment", "family_history")
        if v is not None:
            x["family_history"] = v

    # ---- 5 brain scan ------------------------------------------------------
    box, on = section(5, "Brain scan (PET)", "Amyloid and tau numbers from a PET report.", "has_pet", False)
    if on:
        pet = {}
        c = box.columns(2)
        pet["centiloids"] = c[0].number_input("Amyloid PET: Centiloids", -60.0, 250.0, 10.0,
                                              help="0 is typical of young healthy adults, 100 of typical Alzheimer's dementia.")
        pet["amyloid_positive"] = float(c[1].selectbox("Amyloid status in the report", ["Negative", "Positive"]) == "Positive")
        if box.checkbox("Tau PET also available"):
            c = box.columns(3)
            pet["tau_tracer"] = c[0].selectbox("Tau tracer", ["Flortaucipir", "MK-6240"])
            pet["META_TEMPORAL_SUVR"] = c[1].number_input("Meta-temporal SUVR", 0.5, 5.0, 1.2, step=0.05)
            pet["CTX_ENTORHINAL_SUVR"] = c[2].number_input("Entorhinal SUVR", 0.5, 5.0, 1.15, step=0.05)
        x["pet"] = pet
        box.caption("In testing, adding PET numbers did not measurably improve accuracy over the other sections.")

    prior = None
    with st.expander("Compare with a previous visit (optional)"):
        st.caption("Used only to describe change. It does not alter the risk estimate.")
        c = st.columns(3)
        months = c[0].number_input("Months since previous visit", 1, 120, None, placeholder="none")
        if months:
            prior = {"months": months, "CDRSUM": c[1].number_input("CDR Sum of Boxes then", 0.0, 18.0, None, step=0.5, placeholder="not rated"),
                     "MOCATOTS": c[2].number_input("MoCA then", 0, 30, None, placeholder="not done")}

# ---- result ----------------------------------------------------------------
res = P.predict(x)
checks = P.check_inputs(x, res["features"])
stage_missing = {"Cognitive tests", "Clinical stage (CDR-SB, diagnosis)"} <= set(res["missing_groups"])
with right:
    st.subheader("Estimate")
    blocking = [m for level, m in checks if level == "check"]
    if blocking:
        st.error("**Check these entries before reading the estimate.**\n\n" + "\n\n".join(f"- {m}" for m in blocking))
    risk = res["risk"]
    c = st.columns(2)
    c[0].metric("Estimated 3-year dementia risk", f"{risk * 100:.0f}%" if risk >= 0.01 else "under 1%")
    sim = P.similar_participants(x)
    who = f"aged {sim['age_group'].lower()}" + (f", status {sim['status']}" if sim["status"] else ", any status")
    c[1].metric("What happened to similar participants", f"{sim['rate'] * 100:.0f}%" if sim["rate"] >= 0.01 else "under 1%",
                help="Share of research participants in the same age group (and with the same status, if entered) who were "
                     "diagnosed with dementia within 3 years.")
    st.caption(f"Right: of {sim['n']:,} research participants {who}, {sim['events']:,} were diagnosed with dementia within 3 years. "
               f"Across everyone without dementia it was {ref['event_rate'] * 100:.0f}%.")
    if "risk_with_pet" in res:
        st.caption(f"Without the PET numbers: {res['risk_clinical'] * 100:.0f}%. With them: {res['risk_with_pet'] * 100:.0f}%. "
                   "Treat the two as similar.")
    for level, m in checks:
        if level == "note":
            st.info(m)

    names = {"Demographics": "About the person", "Genetics / APOE": "Genes", "Cardiovascular / medical": "Heart and blood vessels",
             "Cognitive tests": "Test scores", "Clinical stage (CDR-SB, diagnosis)": "Clinician's assessment"}
    used = [names[g] for g in P.meta["feature_groups"] if g not in res["missing_groups"]]
    st.markdown("**Information used:** " + ", ".join(used) + (", brain scan" if "pet" in x else ""))
    if res["missing_groups"]:
        st.markdown("**Not available, so not used:** " + ", ".join(names[g] for g in res["missing_groups"]))
        if stage_missing:
            st.warning("Without any thinking or memory information the estimate is little more than an average for the person's age. "
                       "Do not read it as an individual risk.")

    st.markdown("**What moved this estimate**")
    g = pd.Series({names[k]: v for k, v in res["shap_by_group"].items() if k not in res["missing_groups"]}).sort_values()
    chart = pd.DataFrame({"Raises the estimate": g.clip(lower=0), "Lowers the estimate": g.clip(upper=0)})
    st.bar_chart(chart, horizontal=True, color=["#2a78d6", "#eb6834"], height=60 + 45 * len(g))
    st.caption("How much each kind of information pushed the estimate up or down compared with an average participant. "
               "This describes what the model used. It does not show causes, and changing a factor would not necessarily change the outcome.")

    with st.expander("Single items with the largest effect"):
        f = pd.DataFrame({"Value used": res["features"], "Effect": res["shap_by_feature"]}).dropna(subset=["Value used"])
        f = f[f["Effect"].abs() > 0.02].sort_values("Effect", key=np.abs, ascending=False).head(10)
        f["Direction"] = np.where(f["Effect"] > 0, "raises", "lowers")
        f.index = [LABELS.get(k, k) for k in f.index]
        st.dataframe(f[["Value used", "Direction", "Effect"]].round(2), use_container_width=True)
        st.caption("Test results are shown as z-scores: 0 is typical for a healthy person of the same age, sex and education; "
                   "negative is worse. Yes/no items are shown as 1/0.")

    if prior:
        st.markdown("**Change since the previous visit**")
        years = prior["months"] / 12
        if "CDRSUM" in x and prior["CDRSUM"] is not None:
            d = x["CDRSUM"] - prior["CDRSUM"]
            st.write(f"CDR Sum of Boxes: {prior['CDRSUM']:.1f} → {x['CDRSUM']:.1f} "
                     f"({d:+.1f} over {prior['months']} months, {d / years:+.1f} per year).")
        if prior["MOCATOTS"] is not None and "MOCATOTS" in x.get("tests", {}):
            d = x["tests"]["MOCATOTS"] - prior["MOCATOTS"]
            st.write(f"MoCA: {prior['MOCATOTS']} → {x['tests']['MOCATOTS']} ({d:+d} points).")
        st.caption("A description of change between two visits. In testing, adding past visits changed the risk estimate very little "
                   "once the current scores were known.")

# ---- evidence assistant ----------------------------------------------------
st.divider()
st.subheader("Ask the evidence library")
st.caption("Answers come only from a small library of open-access papers and are shown with their sources. "
           "This is general published information. It is not advice about this person, and it cannot recommend treatment.")


@st.cache_resource
def load_assistant():
    import rag
    return rag.Assistant() if (rag.STORE / "passages.json").exists() else None


question = st.text_input("Question", placeholder="For example: Is midlife hypertension associated with later dementia?")
if question:
    assistant = load_assistant()
    if assistant is None:
        st.info("The evidence library has not been built yet. Run `python scripts/rag_build.py`.")
    else:
        top = sorted(res["shap_by_group"].items(), key=lambda kv: -abs(kv[1]))[:2]
        context = (f"estimated 3-year dementia risk {risk * 100:.0f}%; categories with the largest influence: "
                   + ", ".join(k for k, _ in top))
        with st.spinner("Searching the library"):
            out = assistant.ask(question, model_context=context)
        st.write(out["answer"])
        if out["mode"] == "extractive":
            st.caption("Shown as direct quotations from the sources.")
        for src in out["sources"]:
            with st.expander(f"[{src['n']}] {src['short']} ({src['year']}), section: {src['section'] or 'n/a'}"):
                st.write(src["passage"])
                st.markdown(f"[{src['title']}]({src['url']})")

st.divider()
st.caption(f"Model: gradient-boosted trees trained on {ref['n_train']:,} research participants who did not have dementia at their first "
           "visit, with whole sections hidden at random during training so that it can work with any combination. "
           "How it was tested: AUROC 0.94 on held-out participants and 0.95 at nine held-out research centres for everyone together; "
           "about 0.84 among people with MCI and about 0.88 among cognitively normal people (few cases). Risks were well calibrated "
           "internally and slightly high at the held-out centres.")
