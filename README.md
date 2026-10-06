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
| `reports/phase1_eda_summary.md` | Short summary of the data exploration |
| `reports/variable_map.md` | Every variable used: meaning, category, codes, missing codes, coverage |
| `reports/data_audit_summary.md` | File-level audit of the raw data |
| `reports/figures/` | Figures produced by the notebooks |
| `notebooks/01`-`04` | Exploration notebooks: data catalogue, longitudinal structure and outcomes, feature categories, PET |
| `scripts/nacc_utils.py` | Shared loading, cleaning and plotting helpers, and the variable map |
| `scripts/audit_data.py` | Raw-data audit |
| `data/` | Raw and derived NACC data. Not in the repository (data use agreement) |

To re-run the notebooks, place the NACC files under `data/raw/`, run `python scripts/audit_data.py`, then open the notebooks in order. Reports and notebooks contain aggregate figures only.

## References

- [NACC Official Site](https://www.naccdata.org/)
- [NACC Freeze 74 Release](https://www.naccdata.org/publish-with-nacc-data/doi-digital-object-identifier/freeze-74-june-2026/)
- [NACC Researcher's Guide](https://www.naccdata.org/the-nacc-researchers-guide)
- [NACC Imaging Data](https://www.naccdata.org/about-nacc-data/imaging-data/)

---

**Status:** Research framework in development | **Last Updated:** October 2026
