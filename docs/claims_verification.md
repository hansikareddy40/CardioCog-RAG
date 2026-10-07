# Claims check: is every headline number backed by a real output?

Checked on 7 October 2026 against the files in this repository and the NACC data on the project machine. Every claim in `docs/project_explained_simply.md` (the 390-line project knowledge note) was traced to the script that produces it and the result file it writes. `scripts/verify_results.py` was re-run the same day: **17 of 17 checks passed**.

**Verdict.** The experiments described in the note were run and their outputs are in the repository. No headline result was invented. Four statements did not match their outputs exactly and have been corrected (marked "Corrected" below), and the first-plan items that were never built are now labelled as planned.

A note on file names: the root file `Knowledge.txt` is the original planning brief written before any analysis. It contains no results. The note that reports results is `docs/project_explained_simply.md`. `Knowledge.txt` now carries a status note saying so.

## Headline claims

| # | Claim in the note | Produced by | Result file and value | Status |
|---|---|---|---|---|
| 1 | 57,038 people, 217,598 visits | `scripts/audit_data.py` | `reports/data_audit_summary.md`: 217,598 rows, 57,038 participants; verify check H | Match |
| 2 | 39,798 people came back at least once | Notebook 02 | `reports/phase1_eda_summary.md` | Match |
| 3 | 19,470 eligible people | `scripts/build_dataset.py` | `reports/phase2_cohort_summary.json`, `flow_baseline`; verify check H | Match |
| 4 | 2,818 developed dementia within 3 years | `scripts/build_dataset.py` | Same file; labels recomputed independently in verify check D (0 disagreements in 11,403 visits) | Match |
| 5 | Train 10,879 / validation 2,241 / test 2,286 / external 4,064 | `scripts/build_dataset.py` | `phase2_cohort_summary.json`; verify check F | Match |
| 6 | Five models trained | `scripts/train_tabular.py` | `reports/phase3_tabular_results.json`, `models` | Match |
| 7 | Test AUROC 0.936 to 0.939 | `scripts/train_tabular.py` | LR 0.936, RF 0.932, XGBoost 0.938, MLP 0.938, FT-Transformer 0.939; recomputed from saved predictions in verify check F | Match. Random forest (0.932) is just below the quoted range; the note's table lists it correctly |
| 8 | About 0.945 on 9 held-out centres | `scripts/train_tabular.py` | External AUROC 0.940 to 0.946; 9 external centres, 0 shared with training (verify check B) | Match |
| 9 | About 0.84 inside MCI | `scripts/train_tabular.py`, `scripts/train_by_baseline_group.py` | `mci_at_index`: 0.827 to 0.844 test, 0.833 to 0.846 external | Match |
| 10 | Conversion 41% from MCI, 1.6% from normal | `scripts/train_by_baseline_group.py` | `reports/phase3_by_baseline_group.json`: 41.5% and 1.6% | Match |
| 11 | History (GRU) adds +0.003 to +0.005 | `scripts/train_longitudinal.py` | `reports/phase3_longitudinal_results.json`: trajectory features +0.003 on both sets; GRU +0.000 test, +0.005 external | **Corrected**: the range is +0.000 to +0.005 |
| 12 | Cardiovascular group adds about +0.001 | `scripts/train_tabular.py` | `groups_removed`: +0.0011 test, +0.0010 external; +0.0016 with a fresh seed in verify check G | Match |
| 13 | SHAP shares 41% / 35% / 15% / 5 to 10% / under 5% | `scripts/explain.py` | `reports/phase7_shap_results.json`: 40.8%, 35.0%, 15.5%, 5.1%, 3.5%; ranges over 8 refits agree | Match |
| 14 | About 5,100 people with PET numbers | Notebook 04 | `reports/phase1_eda_summary.md`: 5,100 amyloid, 2,724 tau | Match |
| 15 | 599 raw scans, about 100 GB, about 20 scanner models | `scripts/audit_data.py` | `reports/data_audit_summary.md`: 599 scan folders, 100.7 GB extracted; verify check A | Match |
| 16 | 446 people with images, 5 of whom later developed dementia | `scripts/audit_data.py`, Notebook 04 | `data_audit_summary.md`: 446; `phase1_eda_summary.md`: 5 | Match |
| 17 | Amyloid: 34% against 7% conversion in MCI within 2 years | `scripts/train_pet.py` | Notebook 08 (nearest-visit design); 36% against 9% in the strict design | Match |
| 18 | PET numbers did not clearly improve prediction | `scripts/train_pet.py` | `reports/phase4_pet_results.json`: intervals overlap under both designs | Match |
| 19 | Image SUVR against NACC SUVR: r = 0.98 over 318 scans | `scripts/pet_convert.py`, `pet_register.py`, `pet_suvr.py` | `reports/phase5_pet_image_results.json`: Pearson r 0.977, n 318 | Match |
| 20 | 3D CNN AUROC 0.96, single SUVR number 0.98 | `scripts/train_pet_image.py` | `reports/phase5_pet_cnn_results.json`: 0.960 and 0.982 on 292 scans with a known amyloid status | Match. The target is amyloid status, not dementia |
| 21 | Grad-CAM map was blurry | `scripts/train_pet_image.py` | Same file: 1.7% of the strongest activation inside the cortical region against 3.8% expected by chance | Match |
| 22 | Modality dropout: error 0.44 / 0.12 / 0.11 | `scripts/train_final.py` | `reports/phase6_fusion_results.json`: 0.441, 0.115, 0.111 | **Corrected**: these are for "no thinking tests, no CDR, no diagnosis". The note said "no thinking tests", where the standard model's error is 0.08 |
| 23 | Similar across sex, race, education; weaker at 85 and over | `scripts/validate.py` | `reports/phase8_validation_results.json`: 0.92 to 0.96 across those groups; 0.85 (test) and 0.89 (external) at 85+ | Match; wording made more precise |
| 24 | External calibration: 15.0% predicted, 13.4% observed | `scripts/validate.py` | Same file: 0.1498 and 0.1341 | Match |
| 25 | 17 verification checks | `scripts/verify_results.py` | `reports/verification_report.json`: 17 of 17; re-run 7 October 2026 with the same result | Match |
| 26 | Scrambled-answers test scored 0.51 | `scripts/verify_results.py` | Check E: 0.508 | Match |
| 27 | Two future-information mistakes found and fixed | `scripts/verify_results.py`, `scripts/build_dataset.py` | Checks C (visit 1, visit 3, PET dates) now pass; earlier outputs kept under `data/processed/before_fix/` | Match |
| 28 | Evidence assistant with 11 papers | `scripts/rag_build.py`, `rag.py`, `rag_eval.py` | `reports/phase10_rag_results.json`: 11 sources, 447 passages; right paper found for 32 of 32 questions | Match |
| 29 | The local language model wrote a cited answer about 1 time in 5 | `scripts/rag_eval.py --llm` | Same file: 21.9% generative, 75% fell back to quoting | Match |
| 30 | A working doctor's screen | `app/app.py`, `scripts/inference.py` | Started headless on 7 October 2026: loads with no errors, shows the risk estimate, the checklist and the "not for clinical use" notice | Match |

## Reconciling the two cohort counts

Two pairs of numbers appear in the project. Both are real; they answer slightly different questions.

| Count | Where | Definition |
|---|---|---|
| 19,611 people, 2,833 converted | Notebook 03, `reports/phase1_eda_summary.md` | All first visits, not demented, 3-year status known |
| 19,470 people, 2,818 converted | `reports/phase2_cohort_summary.json`, all models | The same, restricted to in-person visits on UDS versions 1 to 3 (`raw_packet_ok` in `scripts/build_dataset.py`), because telephone visits and the newest form do not have the same tests |

The cohort summary file has shown 19,470 and 2,818 in every committed version. Use these for anything about the models.

## New on 7 October 2026: results by baseline group

`scripts/train_by_baseline_group.py` writes `reports/phase3_by_baseline_group.json`. XGBoost, three seeds, the same splits as every other model.

| Group at the first visit | People | Events | Test AUROC (95% interval) | Test events | External AUROC (95% interval) | External events |
|---|---|---|---|---|---|---|
| All non-demented | 19,470 | 2,818 (14.5%) | 0.937 (0.925 to 0.948) | 332 | 0.946 (0.938 to 0.954) | 545 |
| Cognitively normal | 12,377 | 198 (1.6%) | 0.884 (0.819 to 0.936) | 24 | 0.876 (0.823 to 0.918) | 37 |
| Impaired, not MCI | 1,100 | 135 (12.3%) | 0.887 (0.812 to 0.950) | 19 | 0.950 (0.893 to 0.989) | 12 |
| MCI | 5,993 | 2,485 (41.5%) | 0.835 (0.807 to 0.862) | 289 | 0.846 (0.824 to 0.867) | 496 |

Models trained inside one group only give the same picture: cognitively normal 0.876 test and 0.881 external; MCI 0.829 and 0.838.

In the cognitively normal group the average precision is 0.15 (test) and 0.19 (external) against an event rate of 1.6%. The ranking is useful, but most people flagged would not convert.

### How much CDR-SB and current diagnosis drive the score (test AUROC)

| Features used | All | Cognitively normal | MCI |
|---|---|---|---|
| Diagnosis only | 0.840 | 0.500 | 0.500 |
| CDR-SB only | 0.876 | 0.615 | 0.749 |
| CDR-SB and diagnosis | 0.897 | 0.614 | 0.749 |
| Cognitive tests only | 0.896 | 0.779 | 0.774 |
| Everything except CDR-SB and diagnosis | 0.922 | 0.873 | 0.808 |
| Everything except CDR-SB, diagnosis and cognitive tests | 0.746 | 0.783 | 0.638 |
| Everything | 0.937 | 0.884 | 0.835 |

Paired gain from adding a group to a model that already has everything else (95% bootstrap interval):

| Group added | Set | All | Cognitively normal | MCI |
|---|---|---|---|---|
| CDR-SB and diagnosis | Test | +0.015 (+0.007 to +0.023) | +0.011 (-0.009 to +0.031) | +0.027 (+0.011 to +0.044) |
| CDR-SB and diagnosis | External | +0.021 (+0.015 to +0.027) | +0.015 (-0.009 to +0.040) | +0.036 (+0.022 to +0.049) |
| CDR-SB, diagnosis and cognitive tests | Test | +0.191 (+0.163 to +0.222) | +0.100 (+0.024 to +0.188) | +0.197 (+0.157 to +0.240) |
| CDR-SB, diagnosis and cognitive tests | External | +0.202 (+0.180 to +0.222) | +0.090 (+0.024 to +0.160) | +0.204 (+0.172 to +0.239) |
| Cardiovascular / medical | Test | +0.001 (-0.000 to +0.003) | +0.008 (-0.002 to +0.020) | +0.001 (-0.004 to +0.005) |
| Cardiovascular / medical | External | +0.001 (+0.000 to +0.002) | +0.003 (-0.005 to +0.012) | +0.004 (+0.000 to +0.007) |

Reading: CDR-SB and diagnosis are measured at the index visit and the outcome comes later, so this is not leakage. But current cognitive status carries most of the signal, and the diagnosis alone explains much of the mixed-cohort 0.94. The cardiovascular gain is small in every group; in the cognitively normal group its interval is too wide to rule a modest benefit in or out.

## Wording changed

| Before | After | Why |
|---|---|---|
| Heart and blood-vessel information adds "almost nothing" | "Little extra predictive value in this cohort and setup, once current cognitive status is known" | The experiment supports limited incremental value here, not that the information is useless |
| "History adds very little" | "In this cohort and setup, adding past visits gave only a small improvement", with the caveat that a longer horizon or earlier stage could differ | Same reason |
| "Amyloid is a harmful protein that builds up" | "Amyloid-beta is a protein that can build up into plaques in the brain; a major biological marker of Alzheimer's disease" | Scientific precision |
| PET: "pictures showing harmful proteins" | "Scans that use a tracer to give information about amyloid or tau build-up" | Scientific precision |
| "0.84 is the fairer number" only | A three-row table: all, cognitively normal, MCI, with event counts | The target population must be explicit |

## Planned, not run

These appear in the original brief (`Knowledge.txt`) or the first README but have no output behind them. They are now labelled as planned wherever they are mentioned: CatBoost; an LSTM or Transformer over visits; temporal-attention explanations; a regression head for continuous decline; reported models for the CDR-SB rise outcome; a dementia model from PET images; tau image quantification; MRI; automatic per-patient report generation; temporal validation; any prospective or clinical test.
