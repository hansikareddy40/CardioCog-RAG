# CardioCog-RAG: phased project plan

This plan follows from the Phase 1 exploration (`reports/phase1_eda_summary.md` and the four notebooks). Each phase lists what is built, why that method fits this data, what it produces, and the decisions the team must make and be able to defend.

## Status: what was done, and where the outcome differed from the plan

| Phase | Status | Outcome | Where |
|---|---|---|---|
| 1 Exploration | Done | Data catalogue, variable map, feasibility of each modality | Notebooks 01-04 |
| 2 Cleaning, cohort, target | Done | Suggested defaults adopted; change the constants in `scripts/build_dataset.py` to revisit | Notebook 05 |
| 3 Tabular and longitudinal models | Done | All five model types equal (AUROC about 0.94); cardiovascular features and visit history add almost nothing | Notebook 06 |
| 4 PET numbers | Done | No measurable gain over the clinical model at 2 to 3 years | Notebook 08 |
| 5 PET images | Done for amyloid | Pipeline validated against NACC SUVR (r = 0.98); amyloid-status CNN slightly below the SUVR baseline; Grad-CAM diffuse. Tau image quantification not done (no standard mask) | Notebook 11 |
| 6 Fusion with missing modalities | Done | Modality dropout on the tabular model plus late fusion for PET. The intermediate-fusion deep network (approach C) was not built: the data gave no reason to expect a gain | Notebook 10 |
| 7 Explainability | Done | SHAP by category with stability check | Notebook 07 |
| 8 Validation | Done | Subgroups, thresholds, calibration, decision curve, survival check. Temporal validation and confirmed-conversion sensitivity analysis not done | Notebook 09 |
| 9 Interface | Done | Streamlit prototype with modality checklist | `app/app.py` |
| 10 Evidence assistant | Done | 11 open-access sources, hybrid retrieval, guardrails, local language model with quotation fallback | Notebook 12 |
| 11 Report and viva | Notes written | `docs/viva_notes.md`; the written report is the team's to produce | |

The decisions listed in each phase below were taken at their suggested defaults so that work could proceed. They remain the team's to confirm and justify.

## What the project is, stated honestly

**A research prototype** that predicts future cognitive decline from multimodal NACC data, explains each prediction, and presents the supporting published evidence.

| It is | It is not |
|---|---|
| A retrospective study on NACC Freeze 74 | A clinically validated tool |
| Validated on held-out participants and held-out centres | Validated prospectively on new patients |
| A risk estimate with an explanation | A diagnosis |
| Evidence-cited, non-prescriptive information ("guideline X says ... [source]") | Treatment or medication recommendations for a patient |

Why the right-hand column is out of reach: NACC is observational. It records who took which drug and what happened, but not what *would* have happened otherwise, so it cannot show that a treatment helps a given person. "Clinical validation" means testing with clinicians on new patients over time, which needs ethics approval and years. Claiming either would not survive a viva. The left-hand column is a complete and respectable project.

**The research questions, in testable form**

1. Do cardiovascular and medical features improve prediction of decline beyond demographics, APOE and cognition? (Ablation, with confidence intervals.)
2. Do amyloid and tau PET measures improve prediction among people who have PET? (Same test, on the PET subset.)
3. Does a participant's trajectory before the index visit improve prediction beyond the index visit alone?
4. Which features drive the predictions, and are the explanations stable?

A "no" or "only slightly" to any of these is a valid result.

## The numbers that constrain the design

| Fact from Phase 1 | Consequence |
|---|---|
| 57,038 participants; 39,798 with follow-up | Tabular and longitudinal models are well powered |
| About 19,600 non-demented people with known 3-year status; about 2,800 convert | Enough for gradient boosting and small neural networks |
| Cognitive battery changed in 2015 | Tests must be harmonised before modelling |
| 5,100 with amyloid PET numbers, 2,724 with tau; about 300 and 150 dementia conversions after the scan | PET enters as a few tabular features; test its value on the PET subset |
| 446 with PET images; 5 dementia conversions after the scan | No image model for decline. Images get a different, feasible target |
| 76% of the prediction pool has no PET | The fusion model must handle missing modalities |

---

## Phase 1. Data exploration (done)

Outputs: `notebooks/01`-`04`, `reports/variable_map.md`, `reports/phase1_eda_summary.md`, `reports/figures/`.

## Phase 2. Cleaning, cohort and target

**2a. Cleaning** (`scripts/` + a notebook that shows before and after)

| Step | Method | Why |
|---|---|---|
| Missing codes | Per-variable codes from the variable map, converted to NaN | 88 is a valid age but "not applicable" for smoking years |
| Cardiovascular history | One "ever" flag per condition, from A5 where present and D2 otherwise, carried forward in time | A5 is not collected at version 3 follow-ups; the two forms agree 92-99% |
| Blood pressure, BMI | Range checks; keep per-visit values; add mean of prior visits | Single readings are noisy |
| Cognitive tests | z-scores against cognitively normal participants adjusted for age, sex, education; averaged into domains (memory, attention, executive/speed, language, visuospatial) | Lets version 1-2 and version 3 tests feed the same feature |
| "Could not complete" | Separate indicator per test | It signals severity; imputing the mean would hide that |
| APOE | e4 count with an explicit "not genotyped" level | 20% are not genotyped, and not at random |
| PET | Centiloids, amyloid status, tau meta-temporal and entorhinal SUVR (tau z-scored within tracer); first scan matched to nearest visit within 12 months | Centiloids are comparable across tracers; SUVRs are not |
| Imputation and scaling | Fitted on training folds only | Fitting on all data leaks test information |

Output: one cleaned visit-level table and one participant-level PET table under `data/processed/` (gitignored), plus a data-flow diagram with counts at every exclusion step.

**2b. Cohort and target**

Suggested default, to be confirmed by the team:

* **Cohort:** participants not demented at the index visit, with at least one later visit.
* **Index visit:** the first visit for the tabular baseline; for the longitudinal model, the latest visit that still leaves the outcome window observable, so that earlier visits form the history.
* **Main target:** dementia diagnosis within 3 years of the index visit (yes/no). People whose status at 3 years is unknown are excluded from this label and counted.
* **Secondary targets:** CDR-SB increase of 1 point or more within 3 years; time to dementia (survival analysis) as a robustness check.
* **PET experiments:** same definitions with a 2-year horizon, because follow-up after PET is short.

**2c. Splits (fix once, before any modelling)**

* Split by `NACCID`, never by row.
* Hold out whole centres as an "external-style" test set (for example 20% of centres). The rest is split into training and internal test by participant, stratified by outcome and starting diagnosis.
* Save the split as a file under `data/` and never change it.

**Decisions for the team**

| # | Decision | Options | Suggested |
|---|---|---|---|
| 1 | Cohort | Non-demented; normal only; MCI only | Non-demented, with results also reported for normal and MCI separately |
| 2 | Main target | Dementia conversion; MCI or dementia; CDR-SB rise | Dementia conversion |
| 3 | Horizon | 2, 3 or 5 years | 3 years (2 for PET) |
| 4 | Censored participants | Exclude from label; survival model | Exclude, plus survival model as a check |
| 5 | Confirmed conversion | Single visit; sustained at next visit | Single visit, with sustained as a sensitivity analysis |
| 6 | Cognitive harmonisation | Common tests; domain z-scores; crosswalk; version 3 only | Domain z-scores |
| 7 | Proximal variables (FAQ, independence, memory complaint, AD medication) | In; out; both | Report both |
| 8 | UDS version 4 visits | Include with remapping; exclude | Exclude for now (3% of visits) |

## Phase 3. Tabular and longitudinal models

**3a. Tabular baselines (index-visit features)**

| Model | Why it is here |
|---|---|
| Logistic regression (regularised) | Transparent reference; if nothing beats it, that is worth knowing |
| Random forest | Non-linear reference |
| XGBoost or LightGBM, and CatBoost | Usually the strongest on mixed clinical tables; handle missing values natively |

Tuning by cross-validation inside the training set only. Class imbalance handled by class weights, not by resampling the test set.

**3b. The feature-group experiment (research question 1)**

Train the best tabular model on: demographics; + APOE; + cardiovascular; + cognition; full. Then remove one group at a time from the full model. Report AUROC and AUPRC differences with bootstrap confidence intervals on the same test participants.

**3c. Longitudinal models (research question 3)**

In order of complexity; stop when the next step does not help:

1. **Trajectory features into the tabular model:** for each cognitive domain and CDR-SB, the value at index, the slope over prior visits, the change since the previous visit, and the number of prior visits. Simple, strong, and SHAP still works.
2. **GRU or LSTM** over the sequence of prior visits, with time since the previous visit as an input and masking for different sequence lengths.
3. **Small Transformer** only as an optional comparison.

Compare all three with the index-visit-only model on the same participants. About half of the cohort has three or more visits, so sequences are short; the engineered-slope approach may well win. If so, say so.

**Decision for the team:** minimum number of prior visits for the longitudinal analysis (2 or 3), and whether participants with shorter history are padded or analysed separately.

## Phase 4. PET numbers (tabular PET branch)

* Cohort: the Phase 2 cohort restricted to people with a PET scan within 12 months of the index visit (about 2,750 with amyloid, 1,440 with tau, 1,270 with both, among non-demented people with follow-up).
* Features: Centiloids, amyloid status, tau meta-temporal SUVR, entorhinal SUVR, tracer, source (SCAN or mixed protocol), scan-to-visit gap.
* Experiment (research question 2): clinical model versus clinical + amyloid versus clinical + amyloid + tau, on the same PET-subset test participants, with bootstrap intervals. Report separately for people who were cognitively normal at index, where PET is most likely to add.
* Regularise firmly: about 150 to 300 events, so keep the PET feature list under ten.

## Phase 5. PET images (realistic scope for 446 participants)

**5a. Preprocessing pipeline** (established tools: `dcm2niix`, `nibabel`, `ANTsPy` or `SimpleITK`, `MONAI` transforms)

1. Convert each DICOM series to NIfTI.
2. Align the time frames to each other and average them into one static 3D image.
3. Register to a tracer-appropriate PET template in standard (MNI) space. There is no MRI on disk, so this is PET-only registration.
4. Scale intensity by a reference region (whole cerebellum for amyloid, inferior cerebellum for tau) to get an SUVR image.
5. Resample to one grid (for example 2 mm) and crop to the brain.
6. Visual QC of every image with a saved overlay; record failures.

**5b. Check the pipeline against the PET core.** Compute a cortical summary SUVR from each processed amyloid image and compare with the tabulated value for the same scan (292 of the 318 amyloid images have a Centiloid value). Report the correlation and a Bland-Altman plot. This is the evidence that the preprocessing is correct.

**5c. An image model with a target that has enough labels.** Predict amyloid positive versus negative from the image: 208 negative and 84 positive labelled scans.

* Start with a transparent baseline: mean SUVR in a few atlas regions into logistic regression.
* Then a small 3D CNN (or a pretrained 3D ResNet, for example MedicalNet weights through MONAI, with the early layers frozen), heavy augmentation, 5-fold cross-validation grouped by participant and stratified by tracer.
* Grad-CAM on the CNN, checked against where amyloid is expected (frontal, precuneus, posterior cingulate).
* Report honestly that this task is easier than predicting decline and that the simple regional baseline may match the CNN.

**5d. Exploratory embedding.** Take the CNN's penultimate-layer vector, reduce it to a few dimensions, and offer it to the fusion model as an optional modality. State in advance that with 5 dementia conversions its effect on decline prediction cannot be measured reliably.

**Not in scope, and why:** an end-to-end 3D CNN that predicts cognitive decline from images. The events do not exist in this dataset.

**Decisions for the team:** which tools to standardise on; whether to process tau images as well as amyloid (232 scans, two tracers); whether to do 5c and 5d or stop at 5b if time is short.

## Phase 6. Multimodal fusion with missing modalities

Requirement: a prediction for any participant, whichever modalities they have.

| Approach | How it handles missing modalities | When to use |
|---|---|---|
| **A. Late fusion (recommended main model)** | One model per modality block (clinical, PET numbers, image embedding). A small meta-model combines the available block predictions plus "modality present" flags | Simple, robust, each block trains on everyone who has it |
| **B. Single gradient-boosted model on all features** | Tree models treat missing values natively; add presence flags | Strong baseline; easiest for SHAP |
| **C. Intermediate fusion network** | Each modality has an encoder; missing ones are replaced by a learned "missing" vector; modality dropout during training so the network learns to cope | The "deep" version; include if time allows, compare with A and B |

Evaluation specific to this phase:

* Performance by modality pattern (clinical only; + PET numbers; + image).
* **Simulated missingness:** take test participants who have PET, hide it, and confirm the prediction falls back sensibly towards the clinical-only model.
* Check that the model has not simply learned "has PET means healthier" (the presence flags make this possible). Compare with and without the flags.

**Decision for the team:** which of A, B, C is the headline model. Suggested: B as the headline for explainability, A for the interface, C as an extension.

## Phase 7. Explainability

* **SHAP** on the tree model: global importance, per-person explanations, and importance summed by feature group. Because tests are correlated, report group-level and domain-level importance as the main result and individual features as supporting detail.
* **Stability check:** recompute SHAP rankings over cross-validation folds and report how much the top features move.
* **Longitudinal model:** if a GRU is used, per-visit importance by occlusion (remove a visit, measure the change). Attention weights alone are not a reliable explanation and should not be presented as one.
* **Images:** Grad-CAM for the amyloid classifier only.
* Wording rule for the whole report: SHAP says what the *model* used. It does not say what *causes* decline.

## Phase 8. Validation

| Check | Method |
|---|---|
| Discrimination | AUROC and AUPRC with bootstrap 95% intervals |
| At a decision threshold | Sensitivity, specificity, PPV, NPV; threshold chosen on validation data |
| Calibration | Calibration plot, Brier score; recalibrate on validation data if needed |
| Internal validation | Held-out participants from the training centres |
| External-style validation | Held-out centres, never seen in training |
| Temporal validation (optional) | Train on index visits before a cut-off year, test after |
| Subgroups | Sex, age band, race, education, starting diagnosis, APOE |
| Robustness | Other horizons, confirmed conversion, without proximal variables, survival model |
| Usefulness | Decision-curve analysis against "refer everyone" and "refer no one" |
| Reporting | Follow the TRIPOD+AI checklist; include a participant flow diagram |

The held-out centres are the closest thing to external validation available here. A true external test would need a different cohort such as ADNI, which is future work unless the team obtains access.

## Phase 9. Clinician-facing interface (prototype)

* **Stack:** Streamlit or Gradio. Runs locally.
* **Screen 1, modality checklist:** tick what is available for this person (demographics, medical history and vitals, cognitive tests, prior visits, APOE, amyloid PET, tau PET). Only the ticked sections ask for input.
* **Screen 2, result:** estimated risk within the horizon with an uncertainty band, how it compares with similar people in the cohort, the top contributing factors (SHAP), and which modalities were and were not used.
* **Always visible:** "Research prototype. Not for clinical use. Estimates describe NACC research participants."
* **Data:** the demo uses made-up example patients, never NACC rows. No real patient data is entered or stored.

**Decision for the team:** Streamlit or Gradio; which fusion model sits behind it.

## Phase 10. Evidence-grounded RAG assistant

* **Corpus (curated, citable, legally usable):** open-access reviews and guidelines, for example the Lancet Commission on dementia prevention, AHA/ASA statements on vascular contributions to cognitive impairment, WHO risk-reduction guidelines, NIA-AA and Alzheimer's Association criteria, and open-access PubMed Central papers on APOE, amyloid and tau PET. Keep a list of sources with licences.
* **Pipeline:** split documents into passages; embed them; store in a local vector index (FAISS or Chroma); retrieve with hybrid keyword + embedding search; re-rank; generate an answer that may only use the retrieved passages and must cite each claim.
* **Inputs to the assistant:** the question, the model's output and top factors for the example patient, and the retrieved passages.
* **Guardrails:**
  * answers only from retrieved text, and says "I could not find evidence for that" otherwise;
  * no treatment, medication or dosing advice for an individual; such questions get general guideline information with a citation and a pointer to a clinician;
  * always separates "what the model estimated" from "what the literature says";
  * no NACC participant data is ever sent to an external LLM service. Use made-up example patients, or a locally hosted model.
* **Evaluation:** a set of 30 to 50 test questions with expected sources; measure retrieval hit rate, whether each cited passage supports the claim (checked by hand on a sample), and whether refusal works on out-of-scope and prescriptive questions.

**Decisions for the team:** which LLM (hosted API or local open model) and the data-handling consequences; the final source list.

## Phase 11. Report and viva

Write up following TRIPOD+AI. Limitations section to include: volunteer cohort, selection through follow-up, no mid-life exposure, short follow-up after PET, image sample size, no true external cohort, no prospective or clinical validation.

---

## Suggested order and effort

| Order | Phase | Depends on | Rough share of effort |
|---|---|---|---|
| 1 | 2 Cleaning, cohort, target, splits | Decisions 1-8 | 20% |
| 2 | 3 Tabular baselines and feature-group experiment | 2 | 15% |
| 3 | 3 Longitudinal models | 2 | 10% |
| 4 | 4 PET numbers | 2 | 10% |
| 5 | 6-8 Fusion, SHAP, validation | 3, 4 | 20% |
| 6 | 9-10 Interface and RAG | 6 | 15% |
| 7 | 5 PET images | independent; can run in parallel | 10%, cut first if time is short |

If time runs short, the core that still answers the research questions is phases 2, 3, 4, 7 and 8.

## Working rules

* Raw files under `data/raw` are read-only. Derived data go under `data/interim` or `data/processed`, both gitignored.
* Nothing participant-level is committed, pasted into chat, or sent to an external service. Reports and notebooks contain aggregates only.
* Every model result is reported on the fixed test split, with an interval.
* Every decision in the tables above gets one or two sentences of justification in the report, written by the team.
* Each team member should be able to explain any notebook cell and any line of this plan without help.
