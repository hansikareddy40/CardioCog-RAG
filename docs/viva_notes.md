# Viva preparation notes

Questions a guide or examiner is likely to ask, with the answer the project's own evidence supports and where to find it. Read the notebook named in each answer before the viva; the answer should be in your own words.

## The project in four sentences

We used the NACC Uniform Data Set (57,038 participants, 217,598 visits) to predict whether a person who does not have dementia will be diagnosed with it within 3 years. We compared classical, boosted-tree and deep-learning models, tested what each kind of information adds, explained the predictions with SHAP, and validated on held-out participants and nine held-out centres. We built a PET image pipeline from raw DICOM and checked it against NACC's own values. A prototype interface lets a clinician tick which information is available, and an evidence assistant answers background questions from cited papers.

## Data and design

**Why NACC?** Large, longitudinal, standardised forms across about 46 centres, with diagnosis and CDR at every visit, APOE, and PET for a subset. (Notebook 01)

**Why dementia within 3 years as the target?** Diagnosis and CDR are the only outcomes recorded identically in every form version since 2005. Three years balances the number of events (about 2,800) against the number of people whose outcome is still unknown. (Notebooks 02, 05)

**What happens to people followed for less than 3 years?** Their outcome is unknown, so they are left out of the yes/no label and counted (about half of those eligible). A survival analysis that includes them gave the same conclusions. (Notebooks 05, 09)

**How did you avoid data leakage?** Split by participant, never by visit; nine whole centres held out; every fitted step (cognitive norms, tau scaling, imputation) learned from training participants only; features use only the current and earlier visits; hyperparameters chosen on a validation split; test sets scored once. (Notebook 05, section 5)

**How do you know that is true and not just intended?** `scripts/verify_results.py` tests it. It deletes every visit after a chosen one, rebuilds the features and checks they are unchanged; recomputes the labels with a separate implementation; trains on shuffled labels and confirms performance falls to chance (0.5); recomputes every AUROC from saved predictions; and retrains with a new seed. All 17 checks pass.

**Did the checks find anything?** Yes, two things, both fixed before the final numbers. Education and race missing at an early visit had been filled from a later visit (about 0.1% of participants). And PET scans had been attached to the nearest visit, which in about four of five cases was before the scan. After the fixes the headline results changed only in the third decimal place.

**The cognitive tests changed in 2015. How did you handle that?** Each score was converted to a z-score against cognitively normal participants of the same age, sex and education, and z-scores were averaged within a domain. Healthy participants the norms never saw average within 0.1 of zero in every version. (Notebook 05, section 2.2)

**Why not replace all 88s and 99s with missing?** NACC missing codes differ by variable: 88 is a real age and a real diastolic pressure, but "not applicable" for years smoked. Codes are listed per variable in the variable map. (Notebook 01, section 3.3)

## Models

**Which model is best?** None is measurably better. Logistic regression 0.936, random forest 0.932, XGBoost 0.938, MLP 0.939, FT-Transformer 0.939 (test AUROC), with overlapping intervals; about 0.945 on the held-out centres. (Notebook 06)

**Then why include deep learning?** To test it fairly. Published benchmarks find the same on tabular data of this size. The deep-learning idea that did help was modality dropout. (Notebooks 06, 10)

**Is 0.94 realistic?** It is inflated by mixing cognitively normal people (1.6% convert) with people who have MCI (41% convert). Within MCI the AUROC is about 0.84, which is the fairer figure. (Notebook 06, section 1.1)

**Do cardiovascular features help?** Almost not at all once cognition is known: about +0.001 AUROC (test-set interval includes zero); the same in the survival analysis. Limits: NACC enrols at a median age of 71, so mid-life exposure is not observed, and vascular damage may already show up in the cognitive scores. This is not evidence that vascular health is unrelated to dementia. (Notebooks 06, 09)

**Does longitudinal history help?** Very little (+0.003 to +0.005 AUROC), with engineered slopes or a GRU. The current visit already reflects past decline. (Notebook 06, section 3)

**Does PET help?** No gain was demonstrated over 2 to 3 years, under two designs: a strict one where the scan is on or before the index visit (clean, about 30 events, wide intervals) and a larger one using the nearest visit (tight intervals around zero, but the scan often falls shortly after the visit). On its own amyloid is a strong marker (about 34% versus 7% conversion within 2 years among people with MCI), but the clinical model has already captured most of that through memory scores and CDR-SB. A gain of 0.02 to 0.03 AUROC cannot be ruled out, and cognitively normal people cannot be assessed (0 or 1 conversion). (Notebook 08)

## Explainability

**What does SHAP tell you?** What the model relied on. About three quarters is current cognitive state (clinical stage and test scores), about 16% demographics (mainly age), 5 to 10% cardiovascular, under 5% APOE. (Notebook 07)

**Does SHAP show causes?** No. Example: higher BMI goes with lower predicted risk, most plausibly because people lose weight before diagnosis.

**How stable are the explanations?** Category shares are stable across eight retrained models. The order of individual mid-ranked features is not (age moves between 4th and 8th), so only category-level statements are made.

## PET images

**What did you do with the images?** Converted 599 raw dynamic DICOM series into static 3D images, registered them to a common space in two passes, and computed amyloid SUVR with the Centiloid project's standard masks. (Notebook 11)

**How do you know the pipeline is correct?** Our SUVR correlates 0.98 with the values NACC's PET core computed for the same 318 scans using a different, MRI-based pipeline, and separates amyloid-positive from amyloid-negative scans with AUROC 0.98.

**Why not predict decline from the images with a 3D CNN?** Only 5 of the 446 people with an image went on to dementia. The image group is mostly cognitively normal and was scanned recently. No architecture can learn from 5 events.

**What did the image CNN do, then?** Classified amyloid positive versus negative, where labels exist for about 290 scans. It reached about 0.96 AUROC, slightly below the single SUVR number (0.98). Grad-CAM maps were diffuse and did not concentrate on the cortical target region. The simple, standard measure was both more accurate and easier to interpret.

## Missing data, interface, assistant

**What if a patient lacks some assessments?** A standard model breaks when a whole section is missing (Brier score 0.44 with no cognitive assessment, worse than guessing the base rate). Training with whole sections blanked at random keeps it calibrated (0.12, close to a purpose-built model). (Notebook 10)

**What does the interface show?** A 3-year risk, the cohort average for comparison, which information was and was not used, which categories pushed the estimate up or down, and change since a previous visit. It carries a "research prototype, not for clinical use" notice.

**Why does it not recommend treatment?** NACC is observational. It records who took what and what happened, not what would have happened otherwise, so it cannot show that a treatment helps a given person. A tool that implied otherwise would be unsafe.

**How does the evidence assistant avoid making things up?** It retrieves passages from 11 open-access papers (hybrid keyword and embedding search), answers only from them with citations, checks every citation refers to a retrieved passage, falls back to direct quotation when the language model does not cite properly, says "not found" below a similarity threshold, and flags requests for individual advice. Everything runs locally. (Notebook 12)

**How well does it work?** On a 52-question test set written by the team the retrieval and guardrail checks all passed, but that test is easy and not independent. The small local language model produced a properly cited answer for only about a fifth of questions; the rest were answered by quotation.

## Validation and limits

**Is this clinically validated?** No. It is retrospective validation on research volunteers: an internal test set and nine held-out centres, with subgroup, calibration, decision-curve and survival analyses. Clinical validation would need a prospective study with clinicians and new patients.

**Main limitations**

1. Volunteer, memory-clinic cohort: median 16 years of education, about 76% White, median age 71.
2. Half of eligible participants have unknown 3-year status.
3. Mid-life cardiovascular exposure is not observed.
4. Follow-up after PET is short; very few conversions among cognitively normal participants.
5. Performance is lower at age 85 and over; risks are slightly over-estimated at unseen centres.
6. No truly external cohort (for example ADNI) and no prospective use.
7. PET-only image registration (no MRI); tau quantification from images not completed.
8. Small evidence library; assistant evaluation written by the developers.

**What would you do next?** Test on an external cohort; a longer horizon and a cognitively-normal cohort, where cardiovascular features and PET have more room to matter; confirmed (sustained) conversion as the outcome; MRI-based PET processing; an independently written evaluation for the assistant.

## Numbers worth knowing by heart

| Quantity | Value |
|---|---|
| Participants / visits | 57,038 / 217,598 |
| First-visit cohort with known 3-year status | 19,470, of whom 2,818 converted |
| Train / validation / test / external participants (cohort) | 10,879 / 2,241 / 2,286 / 4,064 |
| Test AUROC, XGBoost | 0.938 (0.925-0.948) |
| External-centre AUROC | 0.946 (0.938-0.954) |
| AUROC within MCI | about 0.84 |
| Gain from cardiovascular group | about +0.001 |
| Calibration, external | predicted 15.0%, observed 13.4% |
| PET image scans processed | 599 |
| Image SUVR vs NACC SUVR | r = 0.98 (318 scans) |
