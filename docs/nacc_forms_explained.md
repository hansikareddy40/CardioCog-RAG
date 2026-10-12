# The NACC forms, explained simply

This note explains what the "forms" in the main clinical file are, what each one asks, and which ones our project uses. It goes with `docs/features_by_dataset.md`.

## 1. What is a "form"?

When a volunteer visits a research centre, the staff do not write free notes. They fill in a fixed set of **paper-style questionnaires**, the same at every centre. Each questionnaire is called a **form** and has a short code such as A1 or B4.

* The whole set is called the **UDS** (Uniform Data Set). "Uniform" means every centre uses the same forms, so the data can be combined.
* The letter tells you who or what the form is about:
  * **A forms**: facts about the person and their history (asked in an interview).
  * **B forms**: things the clinician examines or rates.
  * **C forms**: the thinking and memory tests.
  * **D forms**: the clinician's diagnosis.
* In the CSV file, every question on every form is **one column**. One visit is one row. That is why the file has 2,649 columns.

**Form versions.** The forms have been revised over the years: version 1 (2005), version 2 (2008), version 3 (2015) and version 4 (recent). The biggest change was in 2015, when the thinking tests were replaced by a new set. Our project uses in-person visits on versions 1 to 3.

**Columns that start with `NACC`.** These are not typed in by staff. NACC calculates them afterwards so that one column means the same thing across all form versions. Example: `NACCAGE` (age at the visit) and `NACCUDSD` (cognitive status). Where one exists, we prefer it.

## 2. Every form, and whether we use it

| Form | Name | What it asks, in plain words | Do we use it? |
|---|---|---|---|
| Header | Visit information | Who, which centre, visit date, visit number, in person or by telephone, form version | Yes, to organise the data. Not a feature |
| **A1** | Participant demographics | Age, sex, years of education, race, ethnicity, marital status, who they live with | **Yes: demographics (8 features)** |
| A2 | Co-participant demographics | Details of the relative or friend who comes along and answers questions about the person | No. It is about the companion, not the person |
| **A3** | Family history | Whether parents, brothers, sisters or children had memory problems or dementia | **Yes: genes (1 feature, "parent or sibling affected")** |
| **A4** | Medications | Every medicine the person takes | **Yes: heart and blood vessels (5 features: four medicine types and the total count)** |
| **A5** | Health history | Reported by the person: heart attack, atrial fibrillation, stroke, hypertension, diabetes, cholesterol, smoking, alcohol, and more | **Yes: heart and blood vessels (main source of the 12 conditions and smoking)** |
| **B1** | Physical evaluation | Height, weight, blood pressure, heart rate | **Yes: heart and blood vessels (blood pressure, heart rate, BMI)** |
| B2 | Hachinski score | A score for how likely the problems are caused by blood-vessel disease. Only on versions 1 and 2 | No. Missing for more than half of visits |
| B3 | Parkinson's movement scale (UPDRS) | Tremor, stiffness, slowness. Only on versions 1 and 2 | No. Not one of our five kinds of information |
| **B4** | CDR (Clinical Dementia Rating) | The clinician rates six areas of daily life from 0 to 3: memory, orientation, judgment, community affairs, home and hobbies, personal care | **Yes: clinician's assessment (the six ratings added up = CDR Sum of Boxes)** |
| **B5** | NPI-Q (behaviour questionnaire) | Twelve behaviour symptoms, yes or no: delusions, hallucinations, agitation, depression, anxiety, elation, apathy, disinhibition, irritability, restless movement, night-time behaviour, appetite change | **Yes: thinking tests group (1 feature: how many are present)** |
| **B6** | GDS (Geriatric Depression Scale) | Fifteen yes/no questions about mood, giving a score from 0 to 15 | **Yes: thinking tests group (1 feature)** |
| B7 | FAQ (Functional Activities Questionnaire) | Ten everyday tasks rated 0 to 3: paying bills, shopping, cooking, remembering appointments, travelling | No. Too close to the diagnosis itself. Tested separately |
| B8 | Neurological examination | Findings such as weakness, reflexes, walking | No. Not one of our five kinds of information |
| B9 | Clinician's judgment of symptoms | Whether the person or their companion reports memory decline, and when it started | No. Too close to the diagnosis itself. Tested separately |
| **C1** | Thinking tests, versions 1 and 2 (before 2015) | MMSE, Logical Memory, Digit Span, Trail Making, Digit Symbol, animal and vegetable naming, Boston Naming | **Yes: thinking tests** |
| **C2** | Thinking tests, version 3 (from 2015) | MoCA, Craft Story, Benson figure, Number Span, Trail Making, animal and vegetable naming, MINT naming, F-words | **Yes: thinking tests** |
| **D1** | Clinician's diagnosis | Cognitive status (normal, impaired-not-MCI, MCI, dementia), the MCI type, and the suspected cause | **Yes: clinician's assessment (status today), and the outcome (dementia at a later visit)** |
| **D2** | Clinician-assessed medical conditions | The clinician's own record of the same illnesses as A5. Filled in for about half of visits (the newer forms) | **Yes: back-up for A5 when A5 is blank** |
| Genetics | APOE | The APOE gene result, attached by NACC | **Yes: genes (2 features)** |
| FTLD module | Extra forms for frontotemporal dementia | Detailed language and behaviour tests | No. Filled in for very few people |
| LBD module | Extra forms for Lewy body dementia | Sleep, movement and fluctuation questions | No. Filled in for very few people |
| Neuropathology | Autopsy findings | What was found in the brain after death | No. Only for people who died and donated; it is also information from the future |

The file name `ftldlbd` only means that the FTLD and LBD modules are included in the download. The file is the full clinical data set, not just those modules.

## 3. The tests and scales, one line each

### Rated by the clinician

| Name | What it is | How to read it |
|---|---|---|
| **CDR Sum of Boxes** | The six CDR ratings added together | 0 to 18. 0 = no problems. People without dementia are usually 0 to 4 |
| **Cognitive status** (`NACCUDSD`) | The clinician's overall call | Normal, impaired-not-MCI, MCI, or dementia |
| **MCI** | Mild cognitive impairment | A measurable problem with memory or thinking, but daily life is still managed independently |

### Short screening tests

| Name | What the person does | Score |
|---|---|---|
| **MoCA** | A 10-minute mix: draw a clock, remember five words, name animals, subtract sevens | 0 to 30, higher is better. Used from 2015 |
| **MMSE** | A similar short mix of questions | 0 to 30, higher is better. Used before 2015 |

### Memory

| Name | What the person does |
|---|---|
| **Craft Story** (immediate and delayed) | Listens to a short story and retells it at once, then again about 20 minutes later. From 2015 |
| **Logical Memory** (immediate and delayed) | The same idea with a different story. Before 2015 |
| **Benson figure, delayed** | Draws a complex figure from memory 10 to 15 minutes after copying it |

### Attention

| Name | What the person does |
|---|---|
| **Number Span / Digit Span, forward** | Repeats a growing string of numbers |
| **Number Span / Digit Span, backward** | Repeats the numbers in reverse order |

### Speed and planning

| Name | What the person does |
|---|---|
| **Trail Making A** | Connects numbered circles in order as fast as possible. Scored in seconds, so **higher is worse** |
| **Trail Making B** | Alternates numbers and letters (1, A, 2, B ...). Scored in seconds, so **higher is worse** |
| **Digit Symbol** (WAIS) | Fills in symbols that match numbers using a key, as many as possible in 90 seconds. Before 2015 |

### Language

| Name | What the person does |
|---|---|
| **Animals / Vegetables** | Names as many as possible in 60 seconds |
| **F-words** | Says as many words starting with F as possible in 60 seconds |
| **Boston Naming Test** | Names 30 line drawings. Before 2015 |
| **MINT** | Names 32 pictures. From 2015 |

### Drawing

| Name | What the person does |
|---|---|
| **Benson figure, copy** | Copies a complex figure while looking at it |

### Mood and behaviour

| Name | What it is |
|---|---|
| **GDS** | Depression score, 0 to 15, higher means more depressive symptoms |
| **NPI-Q count** | How many of the 12 behaviour symptoms are present |

## 4. How the tests become six scores

A raw score cannot be compared directly: 20 animals is excellent at 90 and ordinary at 60, and the tests changed in 2015. So for every test we ask: **how does this score compare with a healthy person of the same age, sex and education?** The answer is a z-score: 0 is typical, negative is worse than expected.

Then the z-scores of the tests in one area are averaged:

| Area score | Tests averaged |
|---|---|
| Screening | MoCA, MMSE |
| Memory | Craft Story (2), Logical Memory (2), Benson delayed |
| Attention | Number Span (2), Digit Span (2) |
| Speed and planning | Trail Making A and B, Digit Symbol |
| Language | Animals, Vegetables, F-words, Boston, MINT |
| Drawing | Benson copy |

A person has whichever tests their form version included, and the area score is built from those. This is how old and new visits end up on one scale.

## 5. Special codes in the columns

The forms use numbers to mean "no answer", and the meaning changes from column to column.

| Code | Usual meaning |
|---|---|
| -4 | The form or question was not given at this visit |
| 8, 88, 888 | Not applicable |
| 9, 99, 999 | Unknown |
| 95 to 98 (995 to 998 for timed tests) | The test could not be done; 96 / 996 means because of a thinking or behaviour problem |

These are not always missing values: 88 is a real age and a real blood pressure. The correct codes for every column we use are in `reports/variable_map.md`, and we count code 96 as information ("could not do the test") instead of hiding it.

## 6. Summary

| Our factor | Forms it comes from |
|---|---|
| Demographics | A1 |
| Heart and blood vessels | A4, A5, B1, D2 |
| Thinking tests | C1, C2, B5, B6 |
| Clinician's assessment | B4, D1 |
| Genes | A3, genetics |
| Outcome (dementia within 3 years) | D1 at later visits |
| Not used | A2, B2, B3, B7, B8, B9, FTLD and LBD modules, neuropathology |
