# Every column we take, by category: what it holds, in what form, and why

For each of the five categories this note lists the exact CSV column names, what each one holds, the form it has in the file (the values you will actually see if you open the CSV), why we take it, and what it becomes for the model.

The values in the "Form in the CSV" column were read from the files on 12 October 2026.

**Codes you will see in almost every column.** They mean "no real answer" and we turn them into "missing":

| Code | Meaning |
|---|---|
| -4 | The question was not asked at this visit (for example it is on a different form version) |
| 8, 88, 888, 888.8, 8888 | Not applicable / not measured |
| 9, 99, 9999 | Unknown |
| 95, 96, 97, 98 (995 to 998 for timed tests) | The test could not be done. 96 / 996 = because of a thinking or behaviour problem |

---

## 1. Demographics

**File:** `investigator_ftldlbd_nacc74.csv`. **6 columns become 8 features.**

**Why this category:** age is the strongest single risk factor for dementia. The others describe the person and are also needed to judge test scores fairly (a score must be compared with people of the same age, sex and education).

| CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|
| `NACCAGE` | Age at the visit | Whole number of years, 18 to 111 | Dementia risk rises steeply with age | `age` |
| `NACCSEX` | Sex | 1 = male, 2 = female | Risk and test scores differ by sex | `female` (1 or 0) |
| `EDUC` | Years of education | Whole number 0 to 30; 99 = unknown | More education is linked to later onset; also needed to judge test scores | `educ_years` |
| `NACCNIHR` | Race | 1 White, 2 Black, 3 American Indian, 4 Pacific Islander, 5 Asian, 6 multiracial; 99 = unknown | Conversion rates differed between groups in this cohort (mostly a recruitment effect; see `features_by_dataset.md`, section 5) | `race_white`, `race_black`, `race_asian` (each 1 or 0) |
| `NACCHISP` | Hispanic / Latino | 0 = no, 1 = yes; 9 = unknown | Same reason as race | `hispanic` |
| `NACCLIVS` | Who the person lives with | 1 = alone, 2 to 5 = with others or in care; 9 = unknown | Living alone is a marker of independence and social contact | `lives_alone` (1 or 0) |

---

## 2. Cardiovascular

**File:** `investigator_ftldlbd_nacc74.csv`. **33 columns become 25 features.**

**Why this category:** this was the starting idea of the project. Damage to blood vessels reduces blood supply to the brain, and conditions like hypertension, diabetes and stroke are known risk factors for dementia. We wanted to test how much they add to the prediction.

### 2a. Conditions the person has ever had (12 features)

All of these have the same form. The person reports each condition as 0 = never, 1 = recent or active, 2 = in the past (9 = unknown). We turn 1 and 2 into "yes". If the person's answer is blank we use the clinician's own record (the back-up column, where 0 = no, 1 = yes, 8 = not assessed). Once "yes", it stays "yes" at later visits.

| CSV column | Back-up column | Condition | Why we take it | Becomes |
|---|---|---|---|---|
| `HYPERTEN` | `HYPERT` | High blood pressure | The best-known vascular risk factor for dementia | `htn_ever` |
| `HYPERCHO` | `HYPCHOL` | High cholesterol | Vascular risk factor | `chol_ever` |
| `DIABETES` | `DIABET` | Diabetes | Damages small blood vessels, including in the brain | `diabetes_ever` |
| `CVHATT` | `MYOINF` | Heart attack | Sign of serious artery disease | `mi_ever` |
| `CVAFIB` | `AFIBRILL` | Atrial fibrillation (irregular heartbeat) | Causes small clots that can reach the brain | `afib_ever` |
| `CVCHF` | `CONGHRT` | Heart failure | Less blood pumped to the brain | `chf_ever` |
| `CVANGIO` | `ANGIOPCI` | Angioplasty or stent | Sign of treated artery disease | `revasc_ever` |
| `CVBYPASS` | (none) | Bypass surgery | Same | `bypass_ever` |
| `CVPACE` | `PACEMAKE` | Pacemaker | Sign of heart rhythm disease | `pacemaker_ever` |
| `CVOTHR` | (none) | Other heart disease | Catches anything not listed above | `othercvd_ever` |
| `NACCSTYR` | `CBSTROKE` | Stroke. `NACCSTYR` holds the year of the most recent stroke (8888 = never had one, 9999 = unknown year) | Stroke directly damages brain tissue | `stroke_ever` |
| `NACCTIYR` | `CBTIA` | TIA (mini-stroke), same form as above | Warning sign of blood-vessel disease in the brain | `tia_ever` |

### 2b. Smoking (2 features)

| CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|
| `TOBAC100` | Ever smoked more than 100 cigarettes | 0 = no, 1 = yes; 9 = unknown | Smoking damages blood vessels | `smoke_ever` |
| `SMOKYRS` | Total years smoked | Whole number of years; 88 = never smoked, 99 = unknown | How much exposure, not just yes/no | `smoke_years` (0 for never-smokers) |

### 2c. Measurements at the visit (4 features, plus 1 calculated)

| CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|
| `BPSYS` | Systolic blood pressure | mmHg, for example 130; 888 = not measured | The actual pressure today, not just "has hypertension" | `bp_sys` |
| `BPDIAS` | Diastolic blood pressure | mmHg, for example 76; 888 = not measured | Same | `bp_dia` |
| `HRATE` | Resting heart rate | Beats per minute; 888 = not measured | General heart health | `heart_rate` |
| `NACCBMI` | Body mass index | Decimal number, for example 25.8; 888.8 = not measured | Obesity is a vascular risk factor; weight loss often comes before dementia | `bmi` |
| (calculated) | Systolic minus diastolic | | A sign of stiff arteries | `pulse_pressure` |

### 2d. Medicines (5 features, plus 1 calculated)

| CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|
| `NACCAHTN` | Takes a blood-pressure medicine | 0 = no, 1 = yes | Shows the condition is being treated | `med_bp` |
| `NACCLIPL` | Takes a cholesterol medicine | 0 = no, 1 = yes | Same | `med_lipid` |
| `NACCDBMD` | Takes a diabetes medicine | 0 = no, 1 = yes | Same | `med_diabetes` |
| `NACCAC` | Takes a blood thinner | 0 = no, 1 = yes | Marks clot risk (for example atrial fibrillation) | `med_anticoag` |
| `NACCAMD` | Total number of medicines | Whole number 0 to 40 | A rough measure of overall illness | `med_count` |
| (calculated) | How many of hypertension, high cholesterol, diabetes and smoking are "yes" | 0 to 4 | One summary of vascular risk | `vascular_risk_count` |

---

## 3. Cognitive

**File:** `investigator_ftldlbd_nacc74.csv`. **35 columns become 12 features.**

**Why this category:** how the brain is working today is by far the best guide to the next 3 years. In our results this category carries most of the prediction.

### 3a. Thinking tests (20 columns become 6 area scores)

All 20 have the same form: **a whole-number score**, with 88 and 95 to 98 (995 to 998 for the two timed tests) meaning the test was not done. Higher is better, except the two Trail Making tests where the score is seconds and higher is worse.

Each raw score is compared with healthy people of the same age, sex and education, giving a z-score (0 = typical, negative = worse). The z-scores in one area are averaged into one feature.

| CSV column | Test | Score range | Why we take it | Goes into |
|---|---|---|---|---|
| `MOCATOTS` | MoCA, a 10-minute overall check (from 2015) | 0 to 30 | Quick overall measure of thinking | `z_global` |
| `NACCMMSE` | MMSE, a similar overall check (before 2015) | 0 to 30 | Same, for older visits | `z_global` |
| `CRAFTVRS` | Craft Story, retold at once | 0 to 44 | Memory for new information | `z_memory` |
| `CRAFTDVR` | Craft Story, retold after a delay | 0 to 44 | Delayed recall is the earliest thing to fail in Alzheimer's | `z_memory` |
| `LOGIMEM` | Logical Memory story, at once (before 2015) | 0 to 25 | Same as Craft Story, older form | `z_memory` |
| `MEMUNITS` | Logical Memory story, after a delay (before 2015) | 0 to 25 | Same | `z_memory` |
| `UDSBENTD` | Benson figure drawn from memory | 0 to 17 | Memory for pictures, not words | `z_memory` |
| `DIGFORCT` | Number Span forward | 0 to 14 | Attention span | `z_attention` |
| `DIGBACCT` | Number Span backward | 0 to 14 | Holding and reworking information | `z_attention` |
| `DIGIF` | Digit Span forward (before 2015) | 0 to 12 | Same, older form | `z_attention` |
| `DIGIB` | Digit Span backward (before 2015) | 0 to 12 | Same, older form | `z_attention` |
| `TRAILA` | Trail Making A: connect numbers in order | Seconds, up to 150 | Thinking speed | `z_executive` |
| `TRAILB` | Trail Making B: alternate numbers and letters | Seconds, up to 300 | Planning and switching between tasks | `z_executive` |
| `WAIS` | Digit Symbol (before 2015) | 0 to 93 | Thinking speed, older form | `z_executive` |
| `ANIMALS` | Animals named in 60 seconds | Count | Finding words quickly | `z_language` |
| `VEG` | Vegetables named in 60 seconds | Count | Same | `z_language` |
| `UDSVERFC` | Words starting with F in 60 seconds | Count | Same, by letter | `z_language` |
| `BOSTON` | Boston Naming Test: name 30 drawings (before 2015) | 0 to 30 | Naming objects | `z_language` |
| `MINTTOTS` | MINT: name 32 pictures (from 2015) | 0 to 32 | Naming objects | `z_language` |
| `UDSBENTC` | Benson figure copied | 0 to 17 | Seeing and drawing shapes | `z_visuospatial` |

One more feature from the same 20 columns: `tests_failed_cognitive` counts how many tests have code 96 / 996 ("could not do it because of a thinking problem"). That fact is information in itself, so we keep it instead of hiding it as missing.

### 3b. Mood and behaviour (13 columns become 2 features)

| CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|
| `NACCGDS` | Geriatric Depression Scale | Whole number 0 to 15, higher = more depressed; 88 = not done | Depression often goes with, or comes before, dementia | `gds` |
| `DEL`, `HALL`, `AGIT`, `DEPD`, `ANX`, `ELAT`, `APA`, `DISN`, `IRR`, `MOT`, `NITE`, `APP` | Twelve behaviour symptoms: delusions, hallucinations, agitation, depression, anxiety, elation, apathy, disinhibition, irritability, restless movement, night-time behaviour, appetite change | Each is 0 = no, 1 = yes; 9 = unknown | Behaviour changes are an early sign of brain disease | `npi_count` (how many are "yes", 0 to 12) |

### 3c. The clinician's assessment (2 columns become 3 features)

In our code and results this part is reported as its own group, "Clinical stage".

| CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|
| `CDRSUM` | CDR Sum of Boxes: six areas of daily life, each rated 0 to 3, added together | 0 to 18 in steps of 0.5; 0 = no problems | The clinician's overall rating of how much daily life is affected | `CDRSUM` |
| `NACCUDSD` | The clinician's status for the person | 1 = normal, 2 = impaired but not MCI, 3 = MCI, 4 = dementia | Having MCI is the single strongest sign of coming dementia (41.5% convert, against 1.6% of normal) | `is_mci`, `is_impaired_not_mci` (each 1 or 0) |

---

## 4. Genetics

**File:** `investigator_ftldlbd_nacc74.csv`. **2 columns become 3 features.**

**Why this category:** APOE is the main known gene for late-onset Alzheimer's disease. Each copy of its e4 version raises the risk.

| CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|
| `NACCNE4S` | Number of APOE e4 copies | 0, 1 or 2; 9 = not tested | 1 copy raises Alzheimer's risk, 2 copies raise it much more | `apoe_e4_count` |
| `NACCNE4S` | (same column) | | About 1 in 5 people were not tested, and the model needs to know that instead of guessing | `apoe_unknown` (1 = not tested) |
| `NACCFAM` | A parent, sibling or child had cognitive impairment | 0 = no, 1 = yes; 9 = unknown | Family history captures inherited risk beyond APOE | `family_history` |

---

## 5. PET

**Files:** four PET files. **4 columns become 4 features.**

**Why this category:** PET shows the two Alzheimer's proteins, amyloid and tau, building up in the brain, often years before symptoms. We wanted to test whether knowing this improves the prediction.

| File | CSV column | What it holds | Form in the CSV | Why we take it | Becomes |
|---|---|---|---|---|---|
| `investigator_scan_clariti_amyloidpetgaain_nacc74.csv` | `CENTILOIDS` | Amount of amyloid | Number from about -56 to 219. 0 = typical young healthy adult, 100 = typical Alzheimer's dementia | The only amyloid number that can be compared across the four tracers | `amyloid_centiloids` |
| same | `AMYLOIDSTATUS` | Amyloid positive or negative | 0 = negative, 1 = positive; blank if not determined | A simple yes/no that clinicians use | `amyloid_amyloid_status` |
| `investigator_scan_mp_amyloidpetgaain_nacc74.csv` | `CENTILOIDS` | The same, for older scans | Same form | Adds older scans, so more people have a value | `amyloid_centiloids` |
| same | `AMYLOID_STATUS` | The same, for older scans | 0, 1; 9 = undetermined | Same | `amyloid_amyloid_status` |
| `investigator_scan_clariti_taupetnpdka_nacc74.csv` | `METATEMPORALSUVR` | Tau in the meta-temporal area | Decimal number, about 0.8 to 4.4; higher = more tau | The standard summary of early tau build-up | `tau_meta_temporal_suvr_z` |
| same | `CTXENTORHINALSUVR` | Tau in the entorhinal cortex | Decimal number, about 0.6 to 3.6 | The place where tau appears first | `tau_ctx_entorhinal_suvr_z` |
| `investigator_scan_mp_taupetnpdka_nacc74.csv` | `META_TEMPORAL_SUVR` | The same, for older scans | Same form | Adds older scans | `tau_meta_temporal_suvr_z` |
| same | `CTX_ENTORHINAL_SUVR` | The same, for older scans | Same form | Adds older scans | `tau_ctx_entorhinal_suvr_z` |

The tau values are rescaled within each tracer, because the two tau tracers give different numbers for the same amount of tau.

---

## 6. Columns we need that are not in any category

| CSV column | File | What it holds | Form in the CSV | Why we take it |
|---|---|---|---|---|
| `NACCID` | all files | The person's ID | Text code | Groups a person's visits, links scans to the person, and keeps each person on one side of the train/test split |
| `NACCADC` | main file | The research centre | Number code | 9 whole centres are kept aside as "unseen hospitals" |
| `VISITDATE` | main file | Date of the visit | Date | Orders visits and measures the 3 years |
| `NACCVNUM` | main file | Visit number | 1 to 21 | The main model uses visit 1 |
| `PACKET` | main file | Visit type | I = initial, F = follow-up, T and IT = telephone, I4 = new form | Only in-person visits (I, F) have the tests |
| `FORMVER` | main file | Form version | 1, 2, 3, 3.2, 4 | Only versions 1 to 3 are used |
| `NACCUDSD` and `VISITDATE` | main file | Status at later visits | | **The outcome:** 1 if status 4 (dementia) appears within 3 years of the first visit |
| `SCANDATE` | PET files | Date of the scan | Date | A scan is only used at a visit on or after it |
| `QCSTATUS` / `QC_IMAGE` | PET files | Quality check | 1 = passed, 0 = failed | Failed scans are dropped |
| `TRACER` | PET files | Which tracer was used | Number code 1 to 7 | Needed to rescale tau |

---

## 7. Totals

| Category | CSV columns | Features |
|---|---|---|
| Demographics | 6 | 8 |
| Cardiovascular | 33 | 25 |
| Cognitive (tests, mood and behaviour, clinician's assessment) | 35 | 12 |
| Genetics | 2 | 3 |
| PET | 4 (in each of 2 amyloid and 2 tau files) | 4 |
| **Total** | | **52** |
