# Project audit: does every part make sense, and where can it be attacked?

Audit date: 22 October 2026. Every part of the project was checked from the problem statement to the screen: does it make clinical and scientific sense, does the evidence support the claim, and what could an examiner point at. Each weak point has a status: **fixed**, **mitigated** (a check exists, say so), or **limitation** (cannot be fixed with this data; state it yourself before you are asked).

## Verdict in one paragraph

The project is sound. The question is clearly defined, the data are used correctly, the split prevents the usual leakage, the results are reproducible, and 17 independent checks pass. Its real weaknesses are not errors but **limits of the question and the data**: the model leans on the clinician's current rating, the cohort is a volunteer research cohort, the "external" test uses unseen centres from the same study rather than a different study, and the cognitively normal subgroup has few cases. A guide who knows the field will raise these. You are safe if **you raise them first** and show the check that answers each one.

---

## 1. Problem statement and objectives

| Point | Status | What to know |
|---|---|---|
| The title still says "Early Prediction of Cognitive Decline Using Cardiovascular Risk Factors..." but the work predicts **dementia within 3 years** | **Fix in the thesis** | An examiner will notice the mismatch. Use a title that says what was done (suggestions in `docs/thesis_outline.md`) and explain the scope change in section 1.2 of the thesis, as one of the reference theses does ("How the research question developed") |
| "Early prediction" | **Limitation, word carefully** | Most of the signal comes from current cognition and the CDR rating, so the model mostly estimates progression **given the current stage**. Call it "short-term risk of progression to dementia", not "early detection before symptoms" |
| The cardiovascular hypothesis was not confirmed | Not a weakness | A negative result measured properly (ablation, confidence interval, held-out centres, a survival check) is a valid finding. Present it as research question 1 and its answer |

## 2. Data

| Point | Status | What to know |
|---|---|---|
| NACC is a volunteer cohort from memory centres: mostly White, highly educated, median age 71 | **Limitation** | Results describe NACC-like participants. Subgroup results by sex, race and education are in notebook 09 |
| About half of eligible people were not followed for 3 years and were left out | **Mitigated** | A survival model that includes everyone (`scripts/validate.py`) gave the same conclusions. Also, people with dementia at the first visit return less often (61% against 77% of normal people), so the followed group is slightly healthier |
| Missing-value codes differ by column (88 is a real age) | **Handled** | Column-by-column cleaning (`scripts/nacc_utils.py`, `reports/variable_map.md`) |
| Test battery changed in 2015 | **Handled** | Z-scores against healthy training participants of the same age, sex and education; fitted on training data only |
| Form version 4 visits (3%) excluded | Stated | New column names; little data |
| NACC requires a specific acknowledgement in any publication or thesis that uses its data | **Must do in the thesis** | Copy the official acknowledgement paragraph and grant text from the NACC website (naccdata.org) into the acknowledgements. Do not paraphrase it and do not invent the grant numbers |

## 3. Cohort, outcome and leakage

| Point | Status | What to know |
|---|---|---|
| Same person in training and test | **Checked** | Split by person; verified 0 overlaps (check B) |
| Information from later visits leaking into features | **Fixed and checked** | Two real cases were found and fixed (education filled from a later visit; PET scans after the visit). Check C rebuilds features with later visits deleted: identical |
| The outcome is one clinician's diagnosis, which can be reversed | **Fixed today** | New sensitivity analysis (`scripts/sensitivity_confirmed.py`): 217 of 2,818 converters (7.7%) were later recorded as not demented. Removing them gives AUROC 0.947 (test) and 0.950 (unseen centres); MCI 0.839 and 0.851. The conclusion does not depend on reversals. 907 converters had no later visit, so their diagnosis cannot be confirmed either way |
| CDR and current diagnosis used as inputs, and the diagnosis is partly made from the same tests | **Limitation, already reported** | Not leakage (measured at the first visit, outcome later), but it is why 0.94 is optimistic. Report 0.84 within MCI and the ablation (0.75 without current cognition). Results by group in `reports/phase3_by_baseline_group.json` |
| The healthy reference for z-scores is defined by the clinician's "normal" label, which uses the same tests | **Limitation, minor** | This is the standard approach in the NACC literature. Say so if asked |
| Daily-function questions excluded as "too close to the diagnosis" | Stated | Tested separately: +0.005 |

## 4. Models and training

| Point | Status | What to know |
|---|---|---|
| Why five models | Justified | To answer whether modern models beat simple ones. They do not: intervals overlap (logistic regression 0.924 to 0.949, FT-Transformer 0.927 to 0.949) |
| Settings chosen on the validation group, test used once per experiment | **Handled** | Early stopping and model selection use validation only |
| The same test group was used for many experiments (models, ablations, subgroups) | **Limitation, minor** | This slightly inflates confidence in the best numbers. The unseen-centre group gives a second, independent check, and the results agree |
| Little hyperparameter tuning | Stated | Deliberate: the comparison is fair across models, and the models already agree |
| Results depend on one random split | **Mitigated** | Retraining with a new seed reproduces the result (check G); three seeds averaged |
| No package versions recorded | **Fixed today** | `requirements.txt` lists the exact versions used |

## 5. Results and claims

| Claim | Status | What to know |
|---|---|---|
| AUROC 0.94 overall | Supported | Always say 0.84 within MCI in the same breath |
| 0.88 within cognitively normal | **Weak** | Only 24 test and 37 external cases; interval about 0.82 to 0.94. With confirmed conversions only 16 and 33 cases. Present it as "promising but uncertain" |
| "Unseen hospitals" = external validation | **Word carefully** | They are 9 centres held out from the **same** study, using the same forms. That is site-level validation, not external validation on another cohort. Call it "held-out centre validation" |
| Cardiovascular adds +0.001 | Supported | Test interval includes zero; external interval just above zero. Say "little incremental value in this cohort and setup" |
| History adds +0.000 to +0.005 | Supported | Same wording |
| PET numbers do not clearly help | Supported | The fused model scored +0.020 on 243 held-out scanned people, but the interval is -0.009 to +0.066 (includes zero). Say "not shown", never "no effect" |
| Calibration | Supported, with a known gap | Good internally; slightly high at held-out centres (15.0% vs 13.4%); about 8 points too high for MCI under 65 (35% vs 27%) |

## 6. PET images

| Point | Status | What to know |
|---|---|---|
| Image SUVR agrees with NACC's (r = 0.98) | Supported | A real, independent check of the pipeline |
| SUVR (0.98) beats the 3D CNN (0.96) at amyloid status | **Expected, explain why** | The amyloid status label itself comes from the PET core's own quantification, which is essentially a threshold on a cortical SUVR or Centiloid value. A single SUVR number is therefore close to the label's definition, and a CNN learning from 292 scans is unlikely to beat it. This is a feasibility study of the image pipeline, not a new diagnostic method |
| Registration without MRI (PET-only, affine) | **Limitation** | Standard pipelines use the person's MRI. State it |
| Tau images not quantified | Stated | No standard tau mask was available |
| Grad-CAM was diffuse | Reported honestly | Do not claim the CNN "looks at" the right regions |

## 7. The doctor's screen

| Point | Status | What to know |
|---|---|---|
| Asks for exactly the trained features | **Checked** | `scripts/check_clinical_behaviour.py`, check A |
| Worse rating or test score never lowers the risk | **Checked** | 9 directions, both a normal and an MCI case |
| Contradictory entries | **Flagged** | For example a MoCA of 0 with status Normal |
| Race, ethnicity and living alone are inputs | **Open decision** | They improve calibration for Black participants, but the same scores can give 61% or 35% by race. Default on the screen is "Not stated". Decide whether to keep them, and be ready to justify the choice |
| Not for clinical use | Stated on the screen | Keep the banner |

## 8. Evidence chatbot (RAG)

| Point | Status | What to know |
|---|---|---|
| Library is small: 11 papers (12 listed, 1 could not be downloaded) | **Limitation** | Say 11 |
| Test questions written by the team | **Limitation** | Retrieval 32 of 32 is an easy test; say so |
| The small local language model wrote a proper cited answer only about 1 time in 5 | Reported | The fallback is direct quotation, which is safer |
| Not connected to the prediction as an automatic report | Stated | It answers questions, with the prediction as context |

## 9. Reproducibility and integrity

| Point | Status |
|---|---|
| 17 independent checks | Re-run 22 October 2026: 17 of 17 passed |
| Every headline number traced to its output | `docs/claims_verification.md` |
| Seeds fixed; data never modified; participant IDs never in outputs | Checked |
| Package versions | `requirements.txt` (added today) |

## 10. Questions an examiner is most likely to ask, with the evidence

1. "Isn't 0.94 inflated?" Yes, and we show why: MCI 0.84, status alone 0.84, without current cognition 0.75.
2. "What if the diagnosis was wrong?" Confirmed-conversion analysis: 0.947 / 0.950.
3. "How do you know there is no leakage?" Checks B, C, D, E (person split, future deletion, label recomputation, shuffled labels 0.51).
4. "Is this external validation?" Held-out centres from the same study; a different cohort is future work.
5. "Why did cardiovascular not help?" Late-life measurements, mid-life exposure unseen, effect already in cognition. Not "useless".
6. "Why is the CNN worse than one number?" The label is defined from that number.
7. "What about people not followed for 3 years?" Survival model including them: same conclusions.
8. "Why include race?" Calibration; open decision; default not stated.
