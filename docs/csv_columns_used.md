# The exact CSV columns we use, and what happens to each one

Every column name below is copied from the header row of the CSV files and was checked against the files on 12 October 2026. The steps are the ones in `scripts/build_dataset.py`.

**About the form codes (A1, B4, ...).** Those codes are **not in the CSV**. They are the names of the paper questionnaires the columns come from, and they are only useful for looking a column up in the NACC data dictionary. In the CSV you only see column names such as `NACCAGE` or `CDRSUM`. The "Form" column below is just a pointer.

**How to read the tables.** "CSV column" is what is in the file. "Model feature" is the name the model sees after our processing. One CSV column does not always equal one feature: sometimes several columns are combined into one feature.

Main file: `investigator_ftldlbd_nacc74.csv` (2,649 columns, 217,598 rows). We read **about 97 of its columns**: 7 to organise the data (section A), 76 to build the 48 features (sections B to F), and 14 that are only looked at separately (section H).

---

## A. Columns used to organise the data (not features)

| CSV column | What it holds | How we use it |
|---|---|---|
| `NACCID` | The person's ID | Groups all visits of one person; links PET scans to the person; keeps a person entirely in training or entirely in testing |
| `NACCADC` | The research centre | 9 whole centres are kept aside as the "unseen hospitals" test |
| `VISITDATE` | Date of the visit | Orders the visits and measures the 3 years for the outcome |
| `NACCVNUM` | Visit number (1 = first visit) | The main model uses visit 1 |
| `PACKET` | Visit type (I initial, F follow-up, T telephone ...) | Only in-person visits (I and F) are kept |
| `FORMVER` | Form version (1, 2, 3, 4) | Only versions 1 to 3 are kept |
| `NACCUDSD` | Cognitive status at the visit: 1 normal, 2 impaired-not-MCI, 3 MCI, 4 dementia | Decides who is eligible (1, 2 or 3 at the visit) and builds the outcome (see section G). It is also a feature (section E) |

---

## B. Demographics: 6 CSV columns become 8 features

| CSV column | What it holds | What we do | Model feature |
|---|---|---|---|
| `NACCAGE` | Age at the visit, in years | Used as it is | `age` |
| `NACCSEX` | 1 male, 2 female | Turned into 1 = female, 0 = male | `female` |
| `EDUC` | Years of education (99 = unknown) | 99 becomes missing; carried forward from the visit where it was first recorded | `educ_years` |
| `NACCNIHR` | Race: 1 White, 2 Black, 3 American Indian, 4 Pacific Islander, 5 Asian, 6 multiracial | Split into three yes/no features | `race_white`, `race_black`, `race_asian` |
| `NACCHISP` | Hispanic / Latino: 0 no, 1 yes | Used as it is | `hispanic` |
| `NACCLIVS` | Living situation (1 = lives alone) | Turned into 1 = lives alone, 0 = anything else | `lives_alone` |

---

## C. Heart and blood vessels: 33 CSV columns become 25 features

### Conditions (yes / no)

In the CSV each condition is coded 0 = absent, 1 = recent or active, 2 = in the past. We turn 1 and 2 into "yes". When the first column is blank, we take the clinician's version (second column). Once a condition is "yes" at one visit, it stays "yes" at every later visit.

| CSV column (first choice) | CSV column (back-up) | Condition | Model feature |
|---|---|---|---|
| `HYPERTEN` | `HYPERT` | Hypertension | `htn_ever` |
| `HYPERCHO` | `HYPCHOL` | High cholesterol | `chol_ever` |
| `DIABETES` | `DIABET` | Diabetes | `diabetes_ever` |
| `CVHATT` | `MYOINF` | Heart attack | `mi_ever` |
| `CVAFIB` | `AFIBRILL` | Atrial fibrillation | `afib_ever` |
| `CVCHF` | `CONGHRT` | Heart failure | `chf_ever` |
| `CVANGIO` | `ANGIOPCI` | Angioplasty or stent | `revasc_ever` |
| `CVBYPASS` | (none) | Bypass surgery | `bypass_ever` |
| `CVPACE` | `PACEMAKE` | Pacemaker | `pacemaker_ever` |
| `CVOTHR` | (none) | Other heart disease | `othercvd_ever` |
| `NACCSTYR` (year of last stroke) | `CBSTROKE` | Stroke | `stroke_ever` |
| `NACCTIYR` (year of last TIA) | `CBTIA` | TIA (mini-stroke) | `tia_ever` |

### Smoking, measurements and medicines

| CSV column | What it holds | What we do | Model feature |
|---|---|---|---|
| `TOBAC100` | Smoked more than 100 cigarettes in life: 0 no, 1 yes | Once yes, stays yes | `smoke_ever` |
| `SMOKYRS` | Years smoked (88 = not applicable, 99 = unknown) | Codes become missing; set to 0 for never-smokers | `smoke_years` |
| `BPSYS` | Systolic blood pressure, mmHg (888 = not measured) | Latest measured value up to this visit | `bp_sys` |
| `BPDIAS` | Diastolic blood pressure, mmHg | Same | `bp_dia` |
| `HRATE` | Heart rate, beats per minute | Same | `heart_rate` |
| `NACCBMI` | Body mass index (888.8 = not measured) | Same | `bmi` |
| `NACCAHTN` | Takes a blood-pressure medicine: 0 no, 1 yes | Used as it is | `med_bp` |
| `NACCLIPL` | Takes a cholesterol medicine | Used as it is | `med_lipid` |
| `NACCDBMD` | Takes a diabetes medicine | Used as it is | `med_diabetes` |
| `NACCAC` | Takes a blood thinner | Used as it is | `med_anticoag` |
| `NACCAMD` | Total number of medicines | Used as it is | `med_count` |

### Calculated from the above (no CSV column of their own)

| Model feature | How it is calculated |
|---|---|
| `pulse_pressure` | `bp_sys` minus `bp_dia` |
| `vascular_risk_count` | How many of these four are "yes": hypertension, high cholesterol, diabetes, ever smoked |

---

## D. Thinking tests: 33 CSV columns become 9 features

### The 20 test columns become 6 area scores

For every test: the raw score is compared with what a healthy person of the same age, sex and education scores (that gives a z-score, 0 = typical, negative = worse). The z-scores in one area are then averaged. Codes 88 and 95 to 98 (995 to 998 for timed tests) mean the test was not done and become missing.

| CSV column | Test | Area score it goes into (model feature) |
|---|---|---|
| `MOCATOTS` | MoCA total, 0 to 30 | `z_global` |
| `NACCMMSE` | MMSE total, 0 to 30 | `z_global` |
| `CRAFTVRS` | Craft Story, told back at once | `z_memory` |
| `CRAFTDVR` | Craft Story, told back after a delay | `z_memory` |
| `LOGIMEM` | Logical Memory, at once | `z_memory` |
| `MEMUNITS` | Logical Memory, after a delay | `z_memory` |
| `UDSBENTD` | Benson figure, drawn from memory | `z_memory` |
| `DIGFORCT` | Number Span forward | `z_attention` |
| `DIGBACCT` | Number Span backward | `z_attention` |
| `DIGIF` | Digit Span forward (older form) | `z_attention` |
| `DIGIB` | Digit Span backward (older form) | `z_attention` |
| `TRAILA` | Trail Making A, seconds (higher is worse, so we flip the sign) | `z_executive` |
| `TRAILB` | Trail Making B, seconds (higher is worse, so we flip the sign) | `z_executive` |
| `WAIS` | Digit Symbol (older form) | `z_executive` |
| `ANIMALS` | Animals named in 60 seconds | `z_language` |
| `VEG` | Vegetables named in 60 seconds | `z_language` |
| `UDSVERFC` | F-words in 60 seconds | `z_language` |
| `BOSTON` | Boston Naming Test (older form) | `z_language` |
| `MINTTOTS` | MINT naming test | `z_language` |
| `UDSBENTC` | Benson figure, copied | `z_visuospatial` |

### The other three features in this group

| CSV column(s) | What it holds | What we do | Model feature |
|---|---|---|---|
| The same 20 test columns | Code 96 (996 for timed tests) = could not do the test because of a thinking or behaviour problem | Count how many tests have that code | `tests_failed_cognitive` |
| `NACCGDS` | Depression score, 0 to 15 (88 = not done) | Used as it is | `gds` |
| `DEL`, `HALL`, `AGIT`, `DEPD`, `ANX`, `ELAT`, `APA`, `DISN`, `IRR`, `MOT`, `NITE`, `APP` | Twelve behaviour symptoms, each 0 = no, 1 = yes: delusions, hallucinations, agitation, depression, anxiety, elation, apathy, disinhibition, irritability, restless movement, night-time behaviour, appetite change | Count how many are "yes" | `npi_count` |

---

## E. Clinician's assessment: 2 CSV columns become 3 features

| CSV column | What it holds | What we do | Model feature |
|---|---|---|---|
| `CDRSUM` | CDR Sum of Boxes, 0 to 18 | Used as it is | `CDRSUM` |
| `NACCUDSD` | Cognitive status: 1 normal, 2 impaired-not-MCI, 3 MCI, 4 dementia | Split into two yes/no features. (Both "no" means normal. Status 4 never appears as an input, because people with dementia are excluded.) | `is_mci`, `is_impaired_not_mci` |

---

## F. Genes: 2 CSV columns become 3 features

| CSV column | What it holds | What we do | Model feature |
|---|---|---|---|
| `NACCNE4S` | Number of APOE e4 copies: 0, 1 or 2 (9 = not tested) | Used as it is; 9 becomes missing | `apoe_e4_count` |
| `NACCNE4S` | (same column) | 1 if the person was not tested, else 0 | `apoe_unknown` |
| `NACCFAM` | Parent, sibling or child with cognitive impairment: 0 no, 1 yes (9 = unknown) | Once yes, stays yes | `family_history` |

---

## G. The outcome (what the model predicts)

| CSV column | How it becomes the answer |
|---|---|
| `NACCUDSD` and `VISITDATE` | For each first visit: if `NACCUDSD` = 4 (dementia) at any later visit within 3 years, the answer is **1**. If the person was seen 3 years or more later with no dementia in that time, the answer is **0**. If we cannot tell (not followed long enough), the person is left out |

Model name: `y_dementia_3y`.

---

## H. Columns read from the main file but NOT in the main model

| CSV column | What it holds | Why it is only looked at separately |
|---|---|---|
| `BILLS`, `TAXES`, `SHOPPING`, `GAMES`, `STOVE`, `MEALPREP`, `EVENTS`, `PAYATTN`, `REMDATES`, `TRAVEL` | Ten everyday tasks, each rated 0 to 3 | Added up as `faq_total`. Too close to the diagnosis; tested as an extra, +0.005 on the score |
| `INDEPEND` | Level of independence | Same |
| `DECSUB`, `DECIN` | Memory decline reported by the person / by the companion | Same |
| `NACCADMD` | Takes an Alzheimer's medicine | Same |

---

## I. PET files: 4 features

These features do not go into the main model directly. A small second model combines the main model's risk with these four numbers, and only for people who have a scan.

| File | CSV column | What it holds | What we do | Model feature |
|---|---|---|---|---|
| `investigator_scan_clariti_amyloidpetgaain_nacc74.csv` | `CENTILOIDS` | Amount of amyloid | Used as it is | `amyloid_centiloids` |
| same | `AMYLOIDSTATUS` | 1 positive, 0 negative | Used as it is | `amyloid_amyloid_status` |
| `investigator_scan_mp_amyloidpetgaain_nacc74.csv` | `CENTILOIDS` | Same, older scans | Stacked under the file above | `amyloid_centiloids` |
| same | `AMYLOID_STATUS` | 1 positive, 0 negative, 9 undetermined | 9 becomes missing; stacked | `amyloid_amyloid_status` |
| `investigator_scan_clariti_taupetnpdka_nacc74.csv` | `METATEMPORALSUVR` | Tau in the meta-temporal area | Rescaled within tracer against amyloid-negative people | `tau_meta_temporal_suvr_z` |
| same | `CTXENTORHINALSUVR` | Tau in the entorhinal cortex | Same | `tau_ctx_entorhinal_suvr_z` |
| `investigator_scan_mp_taupetnpdka_nacc74.csv` | `META_TEMPORAL_SUVR` | Same, older scans | Stacked under the file above | `tau_meta_temporal_suvr_z` |
| same | `CTX_ENTORHINAL_SUVR` | Same, older scans | Stacked | `tau_ctx_entorhinal_suvr_z` |

Helper columns in the PET files: `NACCID` (link to the person), `SCANDATE` (the scan must be on or before the visit, within one year), `QCSTATUS` or `QC_IMAGE` (must be 1 = passed), `TRACER` (needed to rescale tau). One person's first passed scan is used.

---

## J. Count

| Group | CSV columns used | Model features |
|---|---|---|
| Demographics | 6 | 8 |
| Heart and blood vessels | 33 | 25 |
| Thinking tests | 33 | 9 |
| Clinician's assessment | 2 | 3 |
| Genes | 2 | 3 |
| **Main model total** | **76** | **48** |
| PET (second model) | 2 per amyloid file, 2 per tau file | 4 |

Everything else in the 10 files, about 2,560 columns of the main file and all regional PET values, is not used.
