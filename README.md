# CardioCog-RAG

**Explainable Early Prediction of Cognitive Decline Using Multimodal Data**

## Overview

CardioCog-RAG is a machine learning research framework designed to predict cognitive decline and cognitive-risk progression by integrating multiple complementary data sources:

- **Cardiovascular & Medical Risk Factors** — blood pressure, hypertension, diabetes, vascular risk markers
- **Longitudinal Cognitive Trajectories** — temporal cognitive measurements across multiple visits
- **Demographic Information** — age, sex, education, race/ethnicity
- **Genetic Markers** — APOE genotype for Alzheimer's disease risk context
- **Brain PET Biomarkers** — amyloid and tau PET-derived measurements

## Key Features

- **Multimodal Integration** — combines structured clinical data with biomarker information
- **Longitudinal Modeling** — preserves temporal cognitive patterns and trajectories
- **Explainability** — uses SHAP for the tabular models and Grad-CAM for the image classifier
- **Evidence-Grounded Reporting** — leverages Retrieval-Augmented Generation (RAG) to ground predictions in scientific literature

## Data Source

Primary dataset: **NACC (National Alzheimer's Coordinating Center)** — Freeze 74, June 2026

Available datasets:
- `investigator_ftldlbd_nacc74` — clinical, cognitive, genetic, and neuropathology data
- `investigator_clariti_edc_nacc74` — imaging availability and metadata
- `investigator_scan_clariti_pet_csv_nacc74` — SCAN/CLARiTI PET analysis data
- `investigator_scan_pet_mp_csv_nacc74` — mixed-protocol PET analysis data

## Project Architecture

This section describes what was built. Items from the first plan that were not built are listed at the end of it.

### Model Pipeline
1. **Tabular Models** — logistic regression, random forest, XGBoost, an MLP and an FT-Transformer
2. **Longitudinal Component** — trajectory features and a GRU over the visit sequence
3. **Missing Modalities** — one tabular model trained with modality dropout, plus late fusion for PET measures
4. **Prediction Target** — one binary outcome: dementia diagnosis within 3 years, among participants not demented at the index visit

### Explainability Layer
- **SHAP** — feature and feature-group contributions, with a stability check
- **Grad-CAM** — for the amyloid-status image classifier (the maps were diffuse and are reported as such)

### Evidence & Reporting
- **RAG Pipeline** — hybrid retrieval over 11 open-access papers, with guardrails
- **Local language model** — answers from retrieved passages with citations, falling back to direct quotation

### Planned in the first proposal, not built
CatBoost, an LSTM or Transformer over visits, temporal-attention explanations, a regression head for continuous decline, an image model for decline (only 5 conversions among the 446 participants with images), and automatic per-patient report generation.

### Scope
The project started as "cardiovascular risk factors, cognitive trajectories, APOE and PET to predict cognitive decline". After the data exploration the question was fixed as dementia within 3 years. Whether cardiovascular information adds predictive value beyond current cognitive status became a question the project tests, and its answer is reported as a finding.

## Research Questions

1. Can cardiovascular/medical information provide predictive value beyond cognitive trajectories?
2. Do amyloid and tau biomarkers improve early cognitive-risk prediction?
3. Which baseline features are most important for predicting cognitive decline?
4. Can longitudinal cognitive patterns identify at-risk individuals earlier than cross-sectional approaches?

## Important Notes

- This is a **research framework** for decision support, not a clinically validated diagnostic system
- Predictions are based on statistical patterns in the NACC cohort
- All claims require evidence-based grounding via literature retrieval and domain expertise
- Data leakage prevention: train/validation/test splits preserve participant-level (NACCID) integrity

## Getting Started

1. Inspect NACC data files and map to Researcher's Data Dictionary
2. Build baseline cohort with available demographic and cognitive data
3. Integrate cardiovascular/medical features
4. Incorporate APOE genetic information
5. Add PET biomarker features
6. Run progressive experiments (baselines → Tier 1 → Tier 2 → image-based)
7. Generate SHAP/attention explanations
8. Build RAG evidence layer for final reporting

## Repository Layout

| Path | Contents |
|---|---|
| `docs/project_plan.md` | Phased plan, method choices and the decisions still open |
| `reports/phase1_eda_summary.md` | Summary of the data exploration |
| `reports/phase2_3_results.md` | Summary of cleaning, models and explanations |
| `reports/variable_map.md` | Every variable used: meaning, category, codes, missing codes, coverage |
| `reports/*.json` | Aggregate results written by the scripts and read by the notebooks |
| `reports/figures/` | Figures produced by the notebooks |
| `notebooks/01`-`04` | Exploration: data catalogue, longitudinal structure and outcomes, feature categories, PET |
| `notebooks/05` | Cleaning, cohort, target and splits |
| `notebooks/06` | Tabular models (logistic regression to FT-Transformer), feature groups, longitudinal models |
| `notebooks/07` | SHAP explanations and their stability |
| `notebooks/08` | Do amyloid and tau PET measures add to the clinical model? |
| `notebooks/09` | Validation: subgroups, thresholds, calibration, decision curve, survival check |
| `notebooks/10` | Missing modalities: modality dropout and late fusion |
| `notebooks/11` | PET images: conversion, registration, SUVR check against NACC, amyloid CNN and Grad-CAM |
| `notebooks/12` | Evidence assistant: retrieval, guardrails, evaluation |
| `docs/viva_notes.md` | Likely questions with evidence-based answers |
| `docs/project_explained_simply.md` | Plain-language account of the whole project |
| `docs/claims_verification.md` | Every headline number traced to the output that produced it |
| `rag/` | Evidence sources and test questions |
| `scripts/` | Pipeline code (see below) |
| `app/web/`, `app/api.py` | Clinician-facing prototype in React with a thin Python API. It asks for exactly the trained features. `app/web/dist` is the built screen; after editing `app/web/src`, run `npm install` and `npm run build` in `app/web` |
| `app/app.py` | The same prototype in Streamlit (fallback) |
| `docs/features_by_dataset.md` | Which features come from which table, what was left out, and what the model is trained on |
| `data/`, `models/` | NACC data and trained models. Not in the repository (data use agreement) |

## Running the Pipeline

Place the NACC files under `data/raw/`, then:

```
python scripts/audit_data.py          # file-level audit
python scripts/build_dataset.py       # cleaning, cohort, labels, splits
python scripts/train_tabular.py       # five models and the feature-group experiment
python scripts/train_by_baseline_group.py  # the same, separately for cognitively normal, MCI and all
python scripts/train_longitudinal.py  # trajectory features and GRU
python scripts/explain.py             # SHAP
python scripts/train_pet.py           # PET measures experiment
python scripts/validate.py            # validation analyses
python scripts/train_final.py         # missing-modality model, saved to models/
python scripts/pet_convert.py         # raw PET DICOM -> static 3D images
python scripts/pet_register.py --pass 1 && python scripts/pet_register.py --pass 2
python scripts/pet_suvr.py            # image SUVR, checked against NACC's values
python scripts/train_pet_image.py     # amyloid image classifier and Grad-CAM (GPU)
python scripts/rag_build.py           # evidence library
python scripts/rag_eval.py --llm      # evidence assistant tests
python scripts/verify_results.py      # 17 independent checks on data, labels and results
python scripts/check_clinical_behaviour.py   # 28 checks that the screen's model behaves sensibly
python -m uvicorn app.api:app --port 8000    # interface (React), then open http://localhost:8000
streamlit run app/app.py              # the same interface in Streamlit (fallback)
```

Reports and notebooks contain aggregate figures only. The interface is a research prototype and is not for clinical use.

## Main Findings So Far

Dementia within 3 years, among participants not demented at the index visit; all figures on held-out participants.

- All non-demented participants together: AUROC 0.94 on the internal test set and 0.95 on nine held-out centres.
- By diagnosis at the index visit: about 0.84 within MCI (289 and 496 events); about 0.88 within cognitively normal participants, with wide intervals (24 and 37 events, 1.6% event rate). See `reports/phase3_by_baseline_group.json`.
- The score is driven mainly by current cognitive status. Diagnosis alone gives 0.84 and CDR-SB with diagnosis 0.90 in the mixed cohort; without CDR-SB, diagnosis and cognitive tests the AUROC is 0.75.
- Logistic regression, random forest, XGBoost, an MLP and an FT-Transformer perform the same.
- Cardiovascular and medical features showed little incremental value in this cohort and setup once current cognition is known (about +0.001 AUROC). This is not evidence that vascular health is unrelated to dementia.
- Visit history gave limited incremental improvement over the current visit in this cohort and setup (+0.000 to +0.005).
- Amyloid and tau PET measures did not demonstrably improve the clinical model over 2 to 3 years, under two ways of matching scans to visits.
- A PET-only image pipeline built from raw DICOM reproduces the PET core's amyloid SUVR (correlation 0.98 over 318 scans). A 3D CNN classifies amyloid status at AUROC 0.96, below the single SUVR measurement (0.98).
- Training with modality dropout keeps predictions calibrated when whole sections of input are missing.

## References

- [NACC Official Site](https://www.naccdata.org/)
- [NACC Freeze 74 Release](https://www.naccdata.org/publish-with-nacc-data/doi-digital-object-identifier/freeze-74-june-2026/)
- [NACC Researcher's Guide](https://www.naccdata.org/the-nacc-researchers-guide)
- [NACC Imaging Data](https://www.naccdata.org/about-nacc-data/imaging-data/)

---

**Status:** Research framework in development | **Last Updated:** October 2026
