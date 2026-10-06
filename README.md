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
- **Explainability** — uses SHAP, temporal attention, and Grad-CAM for feature and decision interpretation
- **Evidence-Grounded Reporting** — leverages Retrieval-Augmented Generation (RAG) to ground predictions in scientific literature

## Data Source

Primary dataset: **NACC (National Alzheimer's Coordinating Center)** — Freeze 74, June 2026

Available datasets:
- `investigator_ftldlbd_nacc74` — clinical, cognitive, genetic, and neuropathology data
- `investigator_clariti_edc_nacc74` — imaging availability and metadata
- `investigator_scan_clariti_pet_csv_nacc74` — SCAN/CLARiTI PET analysis data
- `investigator_scan_pet_mp_csv_nacc74` — mixed-protocol PET analysis data

## Project Architecture

### Model Pipeline
1. **Baseline Models** — logistic regression, random forest, XGBoost, CatBoost
2. **Longitudinal Component** — LSTM/Transformer for cognitive trajectory sequences
3. **Multimodal Fusion** — combines clinical, cognitive, demographic, genetic, and PET representations
4. **Prediction Heads** — classification and/or regression for cognitive risk stratification

### Explainability Layer
- **SHAP** — feature importance and contribution analysis
- **Temporal Attention** — identify which visits are most informative
- **Grad-CAM** — regional importance maps for PET imaging (if 3D CNN models are used)

### Evidence & Reporting
- **RAG Pipeline** — retrieves peer-reviewed scientific literature
- **LLM Summarization** — generates readable, evidence-grounded clinical decision-support reports

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
| `scripts/` | Pipeline code (see below) |
| `app/app.py` | Clinician-facing prototype with a modality checklist |
| `data/`, `models/` | NACC data and trained models. Not in the repository (data use agreement) |

## Running the Pipeline

Place the NACC files under `data/raw/`, then:

```
python scripts/audit_data.py          # file-level audit
python scripts/build_dataset.py       # cleaning, cohort, labels, splits
python scripts/train_tabular.py       # five models and the feature-group experiment
python scripts/train_longitudinal.py  # trajectory features and GRU
python scripts/explain.py             # SHAP
python scripts/train_pet.py           # PET measures experiment
python scripts/validate.py            # validation analyses
python scripts/train_final.py         # missing-modality model, saved to models/
streamlit run app/app.py              # interface
```

Reports and notebooks contain aggregate figures only. The interface is a research prototype and is not for clinical use.

## Main Findings So Far

Dementia within 3 years, among participants not demented at the index visit; all figures on held-out participants.

- AUROC 0.94 on the internal test set and 0.95 on nine held-out centres; about 0.84 among participants with MCI.
- Logistic regression, random forest, XGBoost, an MLP and an FT-Transformer perform the same.
- Cardiovascular and medical features add no measurable predictive value once cognition is known (+0.0002 AUROC).
- Visit history adds very little beyond the current visit (+0.003 to +0.005).
- Amyloid and tau PET measures did not measurably improve the clinical model over 2 to 3 years.
- Training with modality dropout keeps predictions calibrated when whole sections of input are missing.

## References

- [NACC Official Site](https://www.naccdata.org/)
- [NACC Freeze 74 Release](https://www.naccdata.org/publish-with-nacc-data/doi-digital-object-identifier/freeze-74-june-2026/)
- [NACC Researcher's Guide](https://www.naccdata.org/the-nacc-researchers-guide)
- [NACC Imaging Data](https://www.naccdata.org/about-nacc-data/imaging-data/)

---

**Status:** Research framework in development | **Last Updated:** October 2026
