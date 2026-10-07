# Which features came from which dataset, and what the model is finally trained on

Checked against the code on 7 October 2026 (`scripts/nacc_utils.py`, `scripts/build_dataset.py`, `scripts/modeling.py`). Counts only; no participant data.

## The short answer

* We received **10 NACC tables**. Only **5 of them** supply inputs to a model; the other 5 were examined and not used as inputs.
* The big clinical table has **2,649 columns**. **787** have data in at least 10% of visits. We mapped **114** of those by hand (`reports/variable_map.md`). The final model uses **about 75 raw columns**, which become **48 model features** in 5 groups.
* PET adds **4 more numbers** through a small second model, and only when a scan is entered.
* The screen asks for exactly these and nothing else. `scripts/check_clinical_behaviour.py` checks this automatically (check A).

## 1. The ten tables

| Table | Rows | What it holds | Used for |
|---|---|---|---|
| `investigator_ftldlbd_nacc74.csv` (main clinical table) | 217,598 visits, 57,038 people | Demographics, medical history, blood pressure, medication, thinking tests, CDR, diagnosis, APOE | **All 48 model features and the outcome** |
| `scan_clariti_amyloidpetgaain` | 3,933 scans | Amyloid PET summary: Centiloids, amyloid status | **PET numbers** (Centiloids, status) |
| `scan_mp_amyloidpetgaain` | 2,235 scans | The same for older scans | **PET numbers** |
| `scan_clariti_taupetnpdka` | 2,653 scans | Tau PET: about 170 regional values | **PET numbers** (2 regions only) |
| `scan_mp_taupetnpdka` | 671 scans | The same for older scans | **PET numbers** (2 regions only) |
| `scan_clariti_amyloidpetnpdka` | 3,933 scans | Amyloid PET by brain region (about 170 values) | Only to check our own image SUVR (notebook 11). Not a model input |
| `scan_mp_amyloidpetnpdka` | 2,235 scans | The same for older scans | Same |
| `scan_clariti_fdgpetnpdka` | 885 scans | FDG PET (brain energy use) | Explored only. Too few people (600) |
| `scan_clariti_petqc` | 8,759 images | Image list: tracer, scanner, pass/fail | Quality filter (pass only) and linking the 599 image folders |
| `clariti_edc` | 857 rows | Study tracking dates | Not a feature source |

The 599 raw PET image folders were used for the separate image experiment (notebook 11). No image feature goes into the dementia model, because only 5 of the 446 people with images developed dementia.

## 2. How 2,649 columns became 48 features

| Step | Columns left | What was removed and why |
|---|---|---|
| Main clinical table as received | 2,649 | |
| Has data in at least 10% of visits | 787 | 1,862 columns are almost empty: forms for rare conditions (FTLD and Lewy body modules), free-text fields, and questions only asked in one form version |
| Mapped by hand with meaning, codes and missing codes | 114 | The rest are detailed sub-questions outside our five kinds of information (for example neurological exam findings, individual medication names, co-participant details) |
| Allowed as an input ("predictor") | 71 | 12 are outcomes, 15 are too close to the diagnosis ("proximal", see below), 16 are IDs, dates and bookkeeping |
| Actually used, as raw columns | about 75 | Includes the 12 behaviour items and 2 stroke-year columns, which are used but summarised |
| **Final model features** | **48** | Raw columns are combined: 20 tests become 6 scores, two history forms become one yes/no, and so on |

## 3. The 48 features the model is trained on

### Group 1: Demographics (8 features)

| Feature | From NACC column | Note |
|---|---|---|
| Age | `NACCAGE` | |
| Sex | `NACCSEX` | |
| Years of education | `EDUC` | |
| Race: White / Black / Asian (3 yes/no features) | `NACCNIHR` | See the note on race below |
| Hispanic / Latino | `NACCHISP` | |
| Lives alone | `NACCLIVS` | |

### Group 2: Heart and blood vessels (25 features)

| Features | From NACC columns | Note |
|---|---|---|
| Hypertension, high cholesterol, diabetes, heart attack, atrial fibrillation, heart failure, angioplasty or stent, bypass, pacemaker, other heart disease (10) | Form A5 (`HYPERTEN`, `HYPERCHO`, `DIABETES`, `CVHATT`, `CVAFIB`, `CVCHF`, `CVANGIO`, `CVBYPASS`, `CVPACE`, `CVOTHR`) and, where A5 is blank, Form D2 (`HYPERT`, `HYPCHOL`, `DIABET`, `MYOINF`, `AFIBRILL`, `CONGHRT`, `ANGIOPCI`, `PACEMAKE`) | "Ever had it": once reported, it stays yes at later visits |
| Stroke, TIA (2) | `CBSTROKE`, `CBTIA`, `NACCSTYR`, `NACCTIYR` | |
| Ever smoked, years smoked (2) | `TOBAC100`, `SMOKYRS` | |
| Systolic BP, diastolic BP, heart rate, BMI (4) | `BPSYS`, `BPDIAS`, `HRATE`, `NACCBMI` | |
| Pulse pressure (1) | calculated | systolic minus diastolic |
| Number of vascular risk factors (1) | calculated | hypertension + cholesterol + diabetes + smoking |
| Blood-pressure, lipid, diabetes, anticoagulant medication; number of medications (5) | `NACCAHTN`, `NACCLIPL`, `NACCDBMD`, `NACCAC`, `NACCAMD` | |

### Group 3: Thinking tests (9 features)

| Feature | From NACC columns | Note |
|---|---|---|
| Screening score | `MOCATOTS`, `NACCMMSE` | Each raw test is first compared with healthy people of the same age, sex and education (a z-score). The tests in one area are then averaged. This is how tests from before and after 2015 share one scale |
| Memory score | `LOGIMEM`, `MEMUNITS`, `CRAFTVRS`, `CRAFTDVR`, `UDSBENTD` | |
| Attention score | `DIGIF`, `DIGIB`, `DIGFORCT`, `DIGBACCT` | |
| Speed and planning score | `TRAILA`, `TRAILB`, `WAIS` | |
| Language score | `ANIMALS`, `VEG`, `BOSTON`, `MINTTOTS`, `UDSVERFC` | |
| Drawing score | `UDSBENTC` | |
| Tests not completed for cognitive reasons | code 96 / 996 on any test | |
| Depression scale | `NACCGDS` | |
| Number of behavioural symptoms | 12 NPI-Q items (`DEL`, `HALL`, `AGIT`, ...) | count of symptoms present |

### Group 4: Clinician's assessment (3 features)

| Feature | From NACC column |
|---|---|
| CDR Sum of Boxes | `CDRSUM` |
| Status is MCI | `NACCUDSD` = 3 |
| Status is impaired, not MCI | `NACCUDSD` = 2 |

### Group 5: Genes (3 features)

| Feature | From NACC column |
|---|---|
| Number of APOE e4 copies | `NACCNE4S` |
| APOE not tested | `NACCNE4S` missing |
| Parent or sibling with cognitive impairment | `NACCFAM` |

### PET numbers (4, used only through the small second model)

| Feature | From |
|---|---|
| Amyloid Centiloids | amyloid summary tables, `CENTILOIDS` |
| Amyloid positive / negative | amyloid summary tables, `AMYLOID_STATUS` |
| Tau, meta-temporal region | tau tables, `META_TEMPORAL_SUVR`, scaled within tracer |
| Tau, entorhinal region | tau tables, `CTX_ENTORHINAL_SUVR`, scaled within tracer |

Only scans that passed quality control are used, and a scan is only used at a visit on or after the scan date.

## 4. What was considered and left out

| Left out | Why |
|---|---|
| **Close-to-diagnosis items**: 10 daily-function questions (FAQ), level of independence, memory complaint by the person and by the informant, Alzheimer's medication | These are almost the diagnosis itself. Tested separately: they add only +0.005 to the score, so the main model does without them |
| **Outcomes**: global CDR, the six CDR boxes, cause of impairment, MCI subtype | They describe the diagnosis. The CDR boxes are already summed in CDR Sum of Boxes |
| Age at first visit, education level in bands, APOE full genotype, height and weight | Duplicates of something already used (age, years of education, e4 count, BMI) |
| Mother / father with impairment, marital status, type of residence, handedness | Covered by "parent or sibling" and "lives alone", or not relevant |
| Smoked in the last 30 days | Recorded in only 27% of visits |
| Hachinski score | Only on the old forms (44% of visits) |
| White-matter changes on imaging | Recorded in 7% of visits |
| Alcohol abuse, heart valve surgery, sleep apnoea | Mapped but not included in the cardiovascular group. They could be added; given that the whole group adds about +0.001, we did not expect them to matter |
| Changes over earlier visits (14 "trajectory" features) | Tested in the history experiment (+0.000 to +0.005). Not in the final model, which uses one visit |
| About 170 regional amyloid values and 170 regional tau values per scan | Too many numbers for too few people with follow-up; we kept the standard summary values |
| FDG PET | 600 people |
| PET image features | Only 5 conversions among 446 people with images |

## 5. A note on race, ethnicity and living alone

These three are in the model because they improved it slightly in testing (+0.003 to +0.004 on the score). The reason is visible in the data: among people with MCI, Black participants converted to dementia less often than White participants (about 22% to 28% against about 40% on held-out data). Without race as an input the model predicts about 35% to 38% for Black participants with MCI, which is too high; with it, 27% to 29%.

This most likely reflects **how volunteers were recruited** at the research centres, not biology. It means the estimate for the same test scores changes noticeably with race (in one test case from 61% to 35%). The screen therefore treats these three as optional ("Not stated" is the default) and says why they are asked. **Whether a tool for clinicians should use them at all is a decision for the team and the mentor**; removing them costs about 0.003 on the score and makes estimates for Black participants too high.

## 6. Is the screen clinically sensible?

`scripts/check_clinical_behaviour.py` runs 28 checks on the saved model through the same code the screen uses. Results on 7 October 2026: **27 of 28 passed**.

| What was checked | Result |
|---|---|
| The screen has an input for every trained feature and no others | Pass: 48 clinical features and 4 PET numbers, nothing extra; all 20 trained tests can be entered |
| Typical cases: normal 72-year-old 0.6%, impaired-not-MCI 4%, typical MCI 61% | Pass: right order and plausible size |
| A worse CDR, worse status, lower test score, more APOE e4 or more behavioural symptoms never lowers the estimate | Pass for all 9 items, in both a normal and an MCI case |
| Contradictory entries are flagged (for example MoCA 0 with status Normal; status Normal with CDR 3; systolic below diastolic) | Pass |
| Missing sections are reported, and the estimate without thinking information is labelled as an age-based average | Pass |
| Ticking every heart condition changes the estimate by | +0.1 points (normal case), +1.3 points (MCI case) |
| On held-out people, predicted risk matches what happened in each age and status group | **Not passed.** 10 of 11 groups are within 7 points (average gap 2.6). For MCI under 65 the model says 35% and 27% actually converted |

Two things that look odd but are correct:

* **The same raw test score counts as worse for a younger or more educated person**, because scores are compared with healthy people of the same age, sex and education. So raising the age with the raw scores unchanged can lower the estimate.
* **The model trusts the clinician's rating more than a single test.** A very low test score with status "Normal" and CDR 0 still gives a low estimate. The screen now flags that combination instead of quietly showing a low number.
