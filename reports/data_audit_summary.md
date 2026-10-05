# NACC data audit (aggregate only)

Generated 2026-10-05 by `scripts/audit_data.py`. Counts, ranges and percentages only; no participant-level values. Dates are shown to the month.

## Tables

| File | Rows | Columns | Unique NACCIDs | Rows per participant (median / mean / max) | Date range | Cells missing |
|---|---|---|---|---|---|---|
| `investigator_clariti_edc_nacc74.csv` | 857 | 56 | 776 | 1 / 1.1 / 4 | SCREENDT 2024-08 to 2026-05; LOCSCREENDT 2024-08 to 2026-05; CNSTDT1 2024-08 to 2026-05 | 53.42% |
| `investigator_ftldlbd_nacc74.csv` | 217,598 | 2,649 | 57,038 | 3 / 3.81 / 21 | VISITDATE 2005-06 to 2026-05 | 81.16% |
| `investigator_scan_clariti_amyloidpetgaain_nacc74.csv` | 3,933 | 18 | 3,566 | 1 / 1.1 / 4 | SCANDATE 2018-09 to 2026-04; PROCESSDATE 2023-05 to 2026-05 | 7.39% |
| `investigator_scan_clariti_amyloidpetnpdka_nacc74.csv` | 3,933 | 175 | 3,566 | 1 / 1.1 / 4 | SCANDATE 2018-09 to 2026-04; PROCESSDATE 2023-05 to 2026-05 | 0.59% |
| `investigator_scan_clariti_fdgpetnpdka_nacc74.csv` | 885 | 172 | 639 | 1 / 1.38 / 4 | SCANDATE 2021-01 to 2026-04; PROCESSDATE 2024-03 to 2026-05 | 0.52% |
| `investigator_scan_clariti_petqc_nacc74.csv` | 8,759 | 14 | 4,585 | 2 / 1.91 / 15 | SCANDATE 2018-09 to 2026-05 | 20.63% |
| `investigator_scan_clariti_taupetnpdka_nacc74.csv` | 2,653 | 174 | 2,405 | 1 / 1.1 / 4 | SCANDATE 2020-07 to 2026-04; PROCESSDATE 2023-05 to 2026-05 | 0.58% |
| `investigator_scan_mp_amyloidpetgaain_nacc74.csv` | 2,235 | 27 | 2,073 | 1 / 1.08 / 3 | SCANDATE 2005-10 to 2023-11; PROCESSDATE 2023-03 to 2025-05 | 14.72% |
| `investigator_scan_mp_amyloidpetnpdka_nacc74.csv` | 2,235 | 175 | 2,073 | 1 / 1.08 / 3 | SCANDATE 2005-10 to 2023-11 | 10.12% |
| `investigator_scan_mp_taupetnpdka_nacc74.csv` | 671 | 182 | 616 | 1 / 1.09 / 3 | SCANDATE 2015-03 to 2020-12; PROCESSDATE 2025-09 to 2025-09 | 4.07% |

Cells missing = blank cells plus NACC not-available codes (-4, -4.4).

### Missingness by column

| File | Blank cells | Complete columns | Columns >50% missing | Columns >90% missing | Columns 100% missing | Median column missing |
|---|---|---|---|---|---|---|
| `investigator_clariti_edc_nacc74.csv` | 5.33% | 4 | 20 | 13 | 6 | 43.41% |
| `investigator_ftldlbd_nacc74.csv` | 11.66% | 143 | 2272 | 1862 | 308 | 97.27% |
| `investigator_scan_clariti_amyloidpetgaain_nacc74.csv` | 7.28% | 10 | 1 | 1 | 0 | 0.0% |
| `investigator_scan_clariti_amyloidpetnpdka_nacc74.csv` | 0.59% | 10 | 1 | 1 | 0 | 0.08% |
| `investigator_scan_clariti_fdgpetnpdka_nacc74.csv` | 0.52% | 171 | 1 | 1 | 0 | 0.0% |
| `investigator_scan_clariti_petqc_nacc74.csv` | 20.63% | 11 | 3 | 2 | 2 | 0.0% |
| `investigator_scan_clariti_taupetnpdka_nacc74.csv` | 0.58% | 10 | 1 | 0 | 0 | 0.08% |
| `investigator_scan_mp_amyloidpetgaain_nacc74.csv` | 14.62% | 11 | 3 | 3 | 0 | 0.04% |
| `investigator_scan_mp_amyloidpetnpdka_nacc74.csv` | 10.12% | 6 | 1 | 1 | 0 | 9.98% |
| `investigator_scan_mp_taupetnpdka_nacc74.csv` | 4.07% | 15 | 2 | 1 | 0 | 2.98% |

Per-column figures are in `data_audit_aggregate.json`.

### Rows per participant and key categories

**`investigator_clariti_edc_nacc74.csv`**
- Rows per participant: 1: 704, 2: 65, 3: 5, 4: 2, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 30

**`investigator_ftldlbd_nacc74.csv`**
- Rows per participant: 1: 17,240, 2: 10,443, 3: 7,471, 4: 5,217, 5: 3,662, 6-10: 9,511, 11+: 3,494
- Centers (ADCs): 46
- PACKET: F (130695), I (56516), T (24555), I4 (5307), IT (525)
- NACCUDSD: 1 (108107), 4 (61326), 3 (38366), 2 (9769), 8 (30)

**`investigator_scan_clariti_amyloidpetgaain_nacc74.csv`**
- Rows per participant: 1: 3,233, 2: 302, 3: 28, 4: 3, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 35
- TRACER: 2 (1751), 4 (1399), 3 (553), 5 (230)
- PROJECT: SCAN (3309), CLARITI (377), PENDING (247)
- VISIT: (blank) (3556), BL (377)
- QCSTATUS: 1 (3930), 0 (3)
- AMYLOIDSTATUS: 0 (1767), 1 (1375), (blank) (791)

**`investigator_scan_clariti_amyloidpetnpdka_nacc74.csv`**
- Rows per participant: 1: 3,233, 2: 302, 3: 28, 4: 3, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 35
- TRACER: 2 (1751), 4 (1399), 3 (553), 5 (230)
- PROJECT: SCAN (3309), CLARITI (377), PENDING (247)
- VISIT: (blank) (3556), BL (377)
- QCSTATUS: 1 (3930), 0 (3)

**`investigator_scan_clariti_fdgpetnpdka_nacc74.csv`**
- Rows per participant: 1: 465, 2: 112, 3: 52, 4: 10, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 10
- TRACER: 1 (885)
- PROJECT: SCAN (775), CLARITI (87), PENDING (23)
- VISIT: (blank) (798), BL (87)
- QCSTATUS: 1 (885)

**`investigator_scan_clariti_petqc_nacc74.csv`**
- Rows per participant: 1: 1,898, 2: 2,032, 3: 303, 4: 186, 5: 61, 6-10: 91, 11+: 14
- Centers (ADCs): 36
- RADIOTRACER: 6 (2109), 2 (1967), 4 (1448), 1 (1176), 7 (840), 3 (596), 8 (361), 5 (240), 10 (22)
- PROJECT: SCAN (7293), CLARITI (970), PENDING (496)
- PASSFAIL: 1 (8759)

**`investigator_scan_clariti_taupetnpdka_nacc74.csv`**
- Rows per participant: 1: 2,188, 2: 190, 3: 23, 4: 4, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 31
- TRACER: 6 (1856), 7 (797)
- PROJECT: SCAN (2200), CLARITI (311), PENDING (142)
- VISIT: (blank) (2342), BL (309), FU1 (2)
- QCSTATUS: 1 (2651), 0 (2)

**`investigator_scan_mp_amyloidpetgaain_nacc74.csv`**
- Rows per participant: 1: 1,920, 2: 144, 3: 9, 4: 0, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 12
- TRACER: 2 (938), 3 (816), 4 (480), (blank) (1)
- SCAN_PROJECT: mixed-protocol (2235)
- QC_IMAGE: 1 (2206), 0 (29)
- AMYLOID_STATUS: 0 (1197), 1 (985), 9 (53)
- DYNAMIC: 2 (1974), 1 (261)

**`investigator_scan_mp_amyloidpetnpdka_nacc74.csv`**
- Rows per participant: 1: 1,920, 2: 144, 3: 9, 4: 0, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 12
- TRACER: 2 (938), 3 (816), 4 (480), (blank) (1)
- SCAN_PROJECT: mixed-protocol (2235)

**`investigator_scan_mp_taupetnpdka_nacc74.csv`**
- Rows per participant: 1: 563, 2: 51, 3: 2, 4: 0, 5: 0, 6-10: 0, 11+: 0
- Centers (ADCs): 6
- TRACER: 6 (568), 7 (103)
- SCAN_PROJECT: mixed-protocol (671)
- QC_IMAGE: 1 (660), 0 (11)
- DYNAMIC: 2 (671)

## PET images

599 scan folders in total.

| Folder | Scans | Participants | Format | Files | Extracted GB | Zip GB | Header tracer class agrees with folder |
|---|---|---|---|---|---|---|---|
| Amyloid | 318 | 318 | DICOM (318) | 266,827 | 64.5 | 13.5 | 308/318 |
| FDG | 49 | 49 | DICOM (49) | 16,564 | 1.9 | 0.8 | 49/49 |
| Tau | 232 | 232 | DICOM (232) | 147,193 | 34.3 | 7.8 | 230/232 |

### Amyloid

- Tracers: PiB (154), Florbetaben (82), Florbetapir (44), NAV4694 (28), unknown (10)
- Matrix (rows x cols x slices): 192x192x47 (80), 360x360x109 (68), 344x344x127 (45), 440x440x159 (31), 192x192x89 (30), 400x400x109 (28), 128x128x90 (20), 256x256x164 (9), +4 more
- Voxel spacing mm (x by y by slice): 1.33x1.33x3.27 (80), 1.03x1.03x2.03 (68), 1.04x1.04x2.03 (45), 0.82x0.82x1.65 (31), 1.02x1.02x2.03 (29), 2.00x2.00x2.00 (20), +6 more
- In-plane spacing: 0.82 to 2.00 mm (median 1.04); slice thickness: 1.00 to 3.27 mm (median 2.03)
- Frames per scan: 4 (151), 6 (77), 17 (46), 8 (20), 26 (14), 14 (9), 12 (1)
- Files per scan: median 436, range 188 to 4134; file count equals slices x frames in 317/318
- Series type: DYNAMIC/IMAGE (318); units: BQML (318); attenuation corrected: 318/318
- Scanners: SIEMENS (177), GE MEDICAL SYSTEMS (112), Philips (29); models: Discovery MI DR (52), Biograph Horizon (45), Biograph_mMR (41), Biograph128_Vision 600 Edge (23), Biograph16_Horizon 4R (23), Biograph40_mCT (20), +14 more
- Participants with more than one scan: 0
- De-identification: PatientName de-identified in 318/318, birth date present in 0

### FDG

- Tracers: FDG (49)
- Matrix (rows x cols x slices): 192x192x47 (28), 256x256x71 (11), 192x192x89 (8), 440x440x159 (2)
- Voxel spacing mm (x by y by slice): 1.33x1.33x3.27 (28), 1.17x1.17x2.79 (11), 1.33x1.33x2.80 (8), 0.72x0.72x1.65 (2)
- In-plane spacing: 0.72 to 1.33 mm (median 1.33); slice thickness: 1.65 to 3.27 mm (median 3.27)
- Frames per scan: 6 (36), 4 (13)
- Files per scan: median 282, range 282 to 636; file count equals slices x frames in 49/49
- Series type: DYNAMIC/IMAGE (49); units: BQML (49); attenuation corrected: 49/49
- Scanners: GE MEDICAL SYSTEMS (47), SIEMENS (2); models: Discovery MI DR (28), Discovery MI (19), Biograph64_Vision 600 (2)
- Participants with more than one scan: 0
- De-identification: PatientName de-identified in 49/49, birth date present in 0

### Tau

- Tracers: MK-6240 (125), Flortaucipir (105), Florbetaben (1), unknown (1)
- Matrix (rows x cols x slices): 360x360x109 (76), 192x192x47 (66), 440x440x159 (30), 128x128x90 (23), 192x192x89 (13), 400x400x109 (7), 256x256x164 (5), 344x344x127 (5), +4 more
- Voxel spacing mm (x by y by slice): 1.03x1.03x2.03 (76), 1.33x1.33x3.27 (66), 0.82x0.82x1.65 (30), 2.00x2.00x2.00 (24), 1.33x1.33x2.80 (12), 1.02x1.02x2.03 (8), +5 more
- In-plane spacing: 0.82 to 2.00 mm (median 1.03); slice thickness: 1.00 to 3.27 mm (median 2.03)
- Frames per scan: 6 (128), 8 (76), 4 (27), 22 (1)
- Files per scan: median 655, range 188 to 3608; file count equals slices x frames in 231/232
- Series type: DYNAMIC/IMAGE (232); units: BQML (232); attenuation corrected: 232/232
- Scanners: SIEMENS (123), GE MEDICAL SYSTEMS (80), Philips (29); models: Biograph Horizon (46), Discovery MI DR (41), Biograph16_Horizon 4R (30), Biograph128_Vision 600 Edge (28), Ingenuity TF PET/CT (24), Discovery 710 (18), +8 more
- Participants with more than one scan: 0
- De-identification: PatientName de-identified in 232/232, birth date present in 0

## PET images matched to tables

- Participants with any PET image: **446** (amyloid 318, tau 232, FDG 49)
- Amyloid and tau both: **149**; all three: 1; amyloid only: 169; tau only: 80

Participants with a local image who also appear in each table:

| Table | Any PET | Amyloid | Tau | FDG | Amyloid and tau |
|---|---|---|---|---|---|
| `investigator_clariti_edc_nacc74.csv` | 30 | 22 | 18 | 2 | 12 |
| `investigator_ftldlbd_nacc74.csv` | 446 | 318 | 232 | 49 | 149 |
| `investigator_scan_clariti_amyloidpetgaain_nacc74.csv` | 363 | 318 | 189 | 6 | 149 |
| `investigator_scan_clariti_amyloidpetnpdka_nacc74.csv` | 363 | 318 | 189 | 6 | 149 |
| `investigator_scan_clariti_fdgpetnpdka_nacc74.csv` | 52 | 4 | 5 | 49 | 2 |
| `investigator_scan_clariti_petqc_nacc74.csv` | 446 | 318 | 232 | 49 | 149 |
| `investigator_scan_clariti_taupetnpdka_nacc74.csv` | 285 | 194 | 232 | 12 | 149 |
| `investigator_scan_mp_amyloidpetgaain_nacc74.csv` | 60 | 38 | 28 | 13 | 19 |
| `investigator_scan_mp_amyloidpetnpdka_nacc74.csv` | 60 | 38 | 28 | 13 | 19 |
| `investigator_scan_mp_taupetnpdka_nacc74.csv` | 29 | 24 | 22 | 0 | 17 |

- Image IDs found in `investigator_scan_clariti_petqc_nacc74.csv`: 599/599
- QC result of local images: 1 (599)
- Project of local images: SCAN (537), PENDING (62)
- Scan dates of local images: 2021-01 to 2026-01
- Tracer code decoded from DICOM headers: 1 = FDG (49); 2 = PiB (154); 3 = Florbetapir (44), unknown (2), Florbetaben (1); 4 = Florbetaben (81), unknown (8); 5 = NAV4694 (28); 6 = Flortaucipir (105), Florbetaben (1), unknown (1); 7 = MK-6240 (125)

Local images that have a SUVR row for the same participant and scan date:

| Table | Amyloid | FDG | Tau |
|---|---|---|---|
| `investigator_scan_clariti_amyloidpetgaain_nacc74.csv` | 318 | 0 | 65 |
| `investigator_scan_clariti_amyloidpetnpdka_nacc74.csv` | 318 | 0 | 65 |
| `investigator_scan_clariti_fdgpetnpdka_nacc74.csv` | 0 | 44 | 0 |
| `investigator_scan_clariti_taupetnpdka_nacc74.csv` | 67 | 0 | 232 |
| `investigator_scan_mp_amyloidpetgaain_nacc74.csv` | 0 | 0 | 0 |
| `investigator_scan_mp_amyloidpetnpdka_nacc74.csv` | 0 | 0 | 0 |
| `investigator_scan_mp_taupetnpdka_nacc74.csv` | 0 | 0 | 0 |

Nearest clinical visit in `investigator_ftldlbd_nacc74.csv` to each scan:

- Images with any visit for that participant: 599
- Gap in days: median 113, p75 173, max 358
- Within 90 / 180 / 365 days: 241 / 470 / 599
- Participants with a visit within 365 days of the scan: Amyloid 318, FDG 49, Tau 232
- Participants with amyloid and tau scans each within 365 days of a visit: **149**

- Clinical visits per PET participant: median 6, range 2 to 20

