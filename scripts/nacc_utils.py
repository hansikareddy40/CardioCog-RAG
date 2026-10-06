"""Shared helpers for the CardioCog-RAG exploration notebooks.

What lives here
---------------
VARIABLES      the project's variable map: every NACC variable we plan to look
               at, its meaning, its category, and which codes mean "missing".
load_uds()     the main clinical table, read from a small cached copy.
clean()        turn NACC missing codes into NaN, one variable at a time.
load_pet_*()   the PET analysis tables with one common set of column names.
savefig()      save a figure under reports/figures.
assert_no_ids() refuse to let a NACCID reach a report or notebook output.

Rules this module follows
-------------------------
* The raw files under data/raw are only ever read, never written.
* Anything participant-level that we derive goes under data/interim, which is
  gitignored.
* Variable meanings come from the NACC UDS Researcher's Data Dictionary (RDD)
  and the SCAN PET documentation. The code frequencies were checked against the
  Freeze 74 file itself, but each entry should still be confirmed against the
  Freeze 74 RDD before it is cited in the report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
INTERIM = REPO / "data" / "interim"
REPORTS = REPO / "reports"
FIGURES = REPORTS / "figures"

UDS_CSV = RAW / "investigator_ftldlbd_nacc74.csv"
UDS_CACHE = INTERIM / "uds_core.parquet"
SCAN_DIR = RAW / "investigator_scan_clariti_pet_csv_nacc74"
MP_DIR = RAW / "investigator_scan_pet_mp_csv_nacc74"
EDC_CSV = RAW / "investigator_clariti_edc_nacc74.csv"
PET_INVENTORY = INTERIM / "pet_scan_inventory.csv"
AUDIT_JSON = REPORTS / "data_audit_aggregate.json"

NACCID_RE = re.compile(r"NACC\d{4,}")

# ---------------------------------------------------------------------------
# Variable map
# ---------------------------------------------------------------------------
# NACC does not use one missing code. "88" is a real age and a real diastolic
# pressure, but it means "not applicable" for smoking years. So the missing
# codes are listed per variable, never applied globally.
#
# role:
#   id         identifier, never a model input
#   structure  describes the visit (date, number, form version)
#   predictor  candidate model input
#   outcome    what we may predict; using a *future* value as input is leakage
#   proximal   a clinician judgement that is almost the diagnosis itself; legal
#              at the index visit, but it can make the model look better than
#              it is, so the team must decide whether to keep it
#   context    useful for description, not planned as an input

M4 = [-4]                       # not available: form or item not collected
YN9 = [-4, 9]                   # plus 9 = unknown
ARU = [-4, 9]                   # absent / recent / remote scale: 9 = unknown
TEST = [-4, 88, 95, 96, 97, 98]  # 95-98 = not done (physical, cognitive, other, refused)
TRAIL = [-4, 88, 995, 996, 997, 998]


def _v(cat, sub, meaning, form, role="predictor", kind="categorical",
       missing=None, codes=""):
    return dict(category=cat, subgroup=sub, meaning=meaning, form=form,
                role=role, kind=kind, missing=missing or M4, codes=codes)


CV, COG, DEM, GEN, PETC = ("Cardiovascular / medical", "Cognitive",
                           "Demographics", "Genetics / APOE", "PET")
LONG, OUT = "Longitudinal structure", "Outcomes"

VARIABLES: dict[str, dict] = {
    # ---- longitudinal structure -------------------------------------------
    "NACCID": _v(LONG, "keys", "Participant ID; links every table", "header", "id", "id"),
    "NACCADC": _v(LONG, "keys", "Alzheimer's Disease Research Center (site)", "header", "structure"),
    "VISITDATE": _v(LONG, "time", "Date of this visit", "header", "structure", "date"),
    "NACCVNUM": _v(LONG, "time", "Visit number in order (1 = first)", "derived", "structure", "count"),
    "NACCAVST": _v(LONG, "time", "Total number of visits this participant has", "derived", "structure", "count"),
    "NACCFDYS": _v(LONG, "time", "Days from first visit to this visit", "derived", "structure", "continuous"),
    "PACKET": _v(LONG, "form", "Visit type", "header", "structure",
                 codes="I initial, F follow-up, T telephone follow-up, IT initial by telephone, I4 initial under UDS v4"),
    "FORMVER": _v(LONG, "form", "UDS form version (1, 2, 3, 3.2, 4)", "header", "structure"),
    "NACCDIED": _v(LONG, "status", "Participant known to have died", "derived", "context", codes="0 no, 1 yes"),
    "NACCACTV": _v(LONG, "status", "Follow-up status at the centre", "derived", "context"),

    # ---- outcomes ----------------------------------------------------------
    "NACCUDSD": _v(OUT, "diagnosis", "Cognitive status at this visit", "D1", "outcome",
                   missing=[-4, 8], codes="1 normal, 2 impaired-not-MCI, 3 MCI, 4 dementia"),
    "CDRGLOB": _v(OUT, "CDR", "Global CDR score", "B4", "outcome", "ordinal",
                  missing=[-4, 99], codes="0, 0.5, 1, 2, 3"),
    "CDRSUM": _v(OUT, "CDR", "CDR Sum of Boxes (0-18): sum of the six domains", "B4",
                 "outcome", "continuous", missing=[-4, 99]),
    "NACCETPR": _v(OUT, "aetiology", "Primary cause of the impairment", "D1", "outcome",
                   missing=[-4, 99], codes="1 Alzheimer's, 2 Lewy body, 7 FTLD, 8 vascular, 88 not impaired, others"),
    "NACCALZD": _v(OUT, "aetiology", "Alzheimer's disease is a presumed cause", "D1", "outcome",
                   codes="0 no, 1 yes, 8 not impaired"),
    "NACCTMCI": _v(OUT, "diagnosis", "MCI subtype", "D1", "outcome",
                   codes="1 amnestic single, 2 amnestic multi, 3 non-amnestic single, 4 non-amnestic multi, 8 no MCI"),
    "MEMORY": _v(OUT, "CDR", "CDR memory box", "B4", "outcome", "ordinal", [-4, 99]),
    "ORIENT": _v(OUT, "CDR", "CDR orientation box", "B4", "outcome", "ordinal", [-4, 99]),
    "JUDGMENT": _v(OUT, "CDR", "CDR judgment and problem solving box", "B4", "outcome", "ordinal", [-4, 99]),
    "COMMUN": _v(OUT, "CDR", "CDR community affairs box", "B4", "outcome", "ordinal", [-4, 99]),
    "HOMEHOBB": _v(OUT, "CDR", "CDR home and hobbies box", "B4", "outcome", "ordinal", [-4, 99]),
    "PERSCARE": _v(OUT, "CDR", "CDR personal care box", "B4", "outcome", "ordinal", [-4, 99]),

    # ---- demographics ------------------------------------------------------
    "NACCAGE": _v(DEM, "age", "Age at this visit (years)", "A1", kind="continuous"),
    "NACCAGEB": _v(DEM, "age", "Age at first visit (years)", "A1", kind="continuous"),
    "NACCSEX": _v(DEM, "sex", "Sex", "A1", missing=[-4, 8, 9], codes="1 male, 2 female"),
    "EDUC": _v(DEM, "education", "Years of education", "A1", kind="continuous", missing=[-4, 99]),
    "NACCEDULVL": _v(DEM, "education", "Education level (grouped)", "A1", "predictor", "ordinal", [-4, 9]),
    "NACCNIHR": _v(DEM, "race / ethnicity", "Race (NIH categories, derived)", "A1", missing=[-4, 99],
                   codes="1 White, 2 Black, 3 American Indian/Alaska Native, 4 Pacific Islander, 5 Asian, 6 multiracial"),
    "NACCHISP": _v(DEM, "race / ethnicity", "Hispanic/Latino ethnicity", "A1", missing=YN9, codes="0 no, 1 yes"),
    "MARISTAT": _v(DEM, "social", "Marital status", "A1", missing=[-4, 9]),
    "NACCLIVS": _v(DEM, "social", "Living situation", "A1", missing=[-4, 9]),
    "HANDED": _v(DEM, "other", "Handedness", "A1", "context", missing=[-4, 9]),
    "INDEPEND": _v(DEM, "function", "Level of independence", "A1", "proximal", "ordinal", [-4, 9],
                   "1 independent ... 4 completely dependent"),
    "RESIDENC": _v(DEM, "social", "Type of residence", "A1", "proximal", missing=[-4, 9]),

    # ---- genetics / APOE and family history -------------------------------
    "NACCAPOE": _v(GEN, "APOE", "APOE genotype", "genetics", missing=[-4, 9],
                   codes="1 e3/e3, 2 e3/e4, 3 e3/e2, 4 e4/e4, 5 e4/e2, 6 e2/e2"),
    "NACCNE4S": _v(GEN, "APOE", "Number of APOE e4 alleles", "genetics", kind="ordinal",
                   missing=[-4, 9], codes="0, 1, 2"),
    "NACCFAM": _v(GEN, "family history", "First-degree relative with cognitive impairment", "A3",
                  missing=YN9, codes="0 no, 1 yes"),
    "NACCMOM": _v(GEN, "family history", "Mother with cognitive impairment", "A3", missing=YN9),
    "NACCDAD": _v(GEN, "family history", "Father with cognitive impairment", "A3", missing=YN9),
    "NACCADMU": _v(GEN, "mutation", "Known dominant AD mutation in the family", "A3", "context"),
    "NACCFTDM": _v(GEN, "mutation", "Known FTLD mutation in the family", "A3", "context"),

    # ---- cardiovascular / medical: measured at the visit (form B1) ---------
    "BPSYS": _v(CV, "blood pressure", "Systolic blood pressure (mmHg)", "B1", kind="continuous",
                missing=[-4, 777, 888]),
    "BPDIAS": _v(CV, "blood pressure", "Diastolic blood pressure (mmHg)", "B1", kind="continuous",
                 missing=[-4, 777, 888]),
    "HRATE": _v(CV, "vitals", "Resting heart rate (beats/min)", "B1", kind="continuous", missing=[-4, 888]),
    "NACCBMI": _v(CV, "anthropometrics", "Body mass index", "B1", kind="continuous", missing=[-4, 888.8]),
    "HEIGHT": _v(CV, "anthropometrics", "Height (inches)", "B1", "context", "continuous", [-4, 88.8]),
    "WEIGHT": _v(CV, "anthropometrics", "Weight (pounds)", "B1", "context", "continuous", [-4, 888]),
    # ---- self/informant-reported history (form A5; not asked at v3 follow-ups)
    "HYPERTEN": _v(CV, "vascular risk", "Hypertension", "A5", missing=ARU, codes="0 absent, 1 recent/active, 2 remote/inactive"),
    "HYPERCHO": _v(CV, "vascular risk", "Hypercholesterolaemia", "A5", missing=ARU, codes="0, 1, 2"),
    "DIABETES": _v(CV, "vascular risk", "Diabetes", "A5", missing=ARU, codes="0, 1, 2"),
    "CVHATT": _v(CV, "heart disease", "Heart attack / cardiac arrest", "A5", missing=ARU, codes="0, 1, 2"),
    "CVAFIB": _v(CV, "heart disease", "Atrial fibrillation", "A5", missing=ARU, codes="0, 1, 2"),
    "CVANGIO": _v(CV, "heart disease", "Angioplasty / endarterectomy / stent", "A5", missing=ARU, codes="0, 1, 2"),
    "CVBYPASS": _v(CV, "heart disease", "Cardiac bypass", "A5", missing=ARU, codes="0, 1, 2"),
    "CVPACE": _v(CV, "heart disease", "Pacemaker", "A5", missing=ARU, codes="0, 1, 2"),
    "CVCHF": _v(CV, "heart disease", "Congestive heart failure", "A5", missing=ARU, codes="0, 1, 2"),
    "CVOTHR": _v(CV, "heart disease", "Other cardiovascular disease", "A5", missing=ARU, codes="0, 1, 2"),
    "CBSTROKE": _v(CV, "cerebrovascular", "Stroke", "A5", missing=ARU, codes="0, 1, 2"),
    "CBTIA": _v(CV, "cerebrovascular", "Transient ischaemic attack", "A5", missing=ARU, codes="0, 1, 2"),
    "TOBAC100": _v(CV, "lifestyle", "Smoked more than 100 cigarettes in life", "A5", missing=YN9),
    "TOBAC30": _v(CV, "lifestyle", "Smoked in the last 30 days", "A5", missing=[-4, 8, 9]),
    "SMOKYRS": _v(CV, "lifestyle", "Total years smoked", "A5", kind="continuous", missing=[-4, 88, 99]),
    "ALCOHOL": _v(CV, "lifestyle", "Alcohol abuse", "A5", missing=ARU, codes="0, 1, 2"),
    # ---- clinician-assessed conditions (form D2; UDS v3 only) --------------
    "HYPERT": _v(CV, "vascular risk", "Hypertension (clinician-assessed)", "D2", missing=[-4, 8], codes="0 no, 1 yes"),
    "HYPCHOL": _v(CV, "vascular risk", "Hypercholesterolaemia (clinician-assessed)", "D2", missing=[-4, 8]),
    "DIABET": _v(CV, "vascular risk", "Diabetes (clinician-assessed)", "D2", missing=[-4, 9],
                 codes="0 no, 1 type 1, 2 type 2, 3 other"),
    "MYOINF": _v(CV, "heart disease", "Myocardial infarct", "D2", missing=[-4, 8]),
    "AFIBRILL": _v(CV, "heart disease", "Atrial fibrillation", "D2", missing=[-4, 8]),
    "CONGHRT": _v(CV, "heart disease", "Congestive heart failure", "D2", missing=[-4, 8]),
    "ANGIOPCI": _v(CV, "heart disease", "Angioplasty / stent", "D2", missing=[-4, 8]),
    "PACEMAKE": _v(CV, "heart disease", "Pacemaker / defibrillator", "D2", missing=[-4, 8]),
    "HVALVE": _v(CV, "heart disease", "Heart valve replacement or repair", "D2", missing=[-4, 8]),
    "SLEEPAP": _v(CV, "other medical", "Sleep apnoea", "D2", missing=[-4, 8]),
    # ---- medications (form A4, derived drug classes; all visit types) ------
    "NACCAHTN": _v(CV, "medication", "Takes any blood-pressure medication", "A4", codes="0 no, 1 yes"),
    "NACCLIPL": _v(CV, "medication", "Takes a lipid-lowering medication", "A4"),
    "NACCDBMD": _v(CV, "medication", "Takes a diabetes medication", "A4"),
    "NACCAC": _v(CV, "medication", "Takes an anticoagulant or antiplatelet", "A4"),
    "NACCAMD": _v(CV, "medication", "Total number of medications", "A4", kind="count"),
    "NACCADMD": _v(COG, "medication", "Takes an Alzheimer's symptomatic drug", "A4", "proximal"),
    # ---- vascular contribution judged by the clinician ---------------------
    "HACHIN": _v(CV, "cerebrovascular", "Hachinski ischaemic score (UDS v1-2 only)", "B2", kind="continuous"),
    "NACCWMHSEV": _v(CV, "cerebrovascular", "White-matter hyperintensity severity on imaging", "D1",
                     "context", missing=[-4, 7, 8, 9]),

    # ---- cognitive: screening ---------------------------------------------
    "NACCMMSE": _v(COG, "global screen", "MMSE total (0-30); UDS v1-2", "C1", kind="continuous", missing=TEST),
    "MOCATOTS": _v(COG, "global screen", "MoCA total, raw (0-30); UDS v3 onward", "C2", kind="continuous", missing=[-4, 88]),
    # ---- memory ------------------------------------------------------------
    "LOGIMEM": _v(COG, "memory", "Logical Memory immediate recall; v1-2", "C1", kind="continuous", missing=TEST),
    "MEMUNITS": _v(COG, "memory", "Logical Memory delayed recall; v1-2", "C1", kind="continuous", missing=TEST),
    "CRAFTVRS": _v(COG, "memory", "Craft Story immediate recall, verbatim; v3", "C2", kind="continuous", missing=TEST),
    "CRAFTDVR": _v(COG, "memory", "Craft Story delayed recall, verbatim; v3", "C2", kind="continuous", missing=TEST),
    "UDSBENTD": _v(COG, "memory", "Benson figure delayed recall; v3", "C2", kind="continuous", missing=TEST),
    # ---- attention / executive --------------------------------------------
    "DIGIF": _v(COG, "attention", "Digit span forward, trials correct; v1-2", "C1", kind="continuous", missing=TEST),
    "DIGIB": _v(COG, "attention", "Digit span backward, trials correct; v1-2", "C1", kind="continuous", missing=TEST),
    "DIGFORCT": _v(COG, "attention", "Number span forward, trials correct; v3", "C2", kind="continuous", missing=TEST),
    "DIGBACCT": _v(COG, "attention", "Number span backward, trials correct; v3", "C2", kind="continuous", missing=TEST),
    "TRAILA": _v(COG, "processing speed", "Trail Making A, seconds (higher = worse)", "C1/C2", kind="continuous", missing=TRAIL),
    "TRAILB": _v(COG, "executive", "Trail Making B, seconds (higher = worse)", "C1/C2", kind="continuous", missing=TRAIL),
    "WAIS": _v(COG, "processing speed", "WAIS-R Digit Symbol; v1-2", "C1", kind="continuous", missing=TEST),
    # ---- language ----------------------------------------------------------
    "ANIMALS": _v(COG, "language", "Category fluency: animals in 60 s", "C1/C2", kind="continuous", missing=TEST),
    "VEG": _v(COG, "language", "Category fluency: vegetables in 60 s", "C1/C2", kind="continuous", missing=TEST),
    "BOSTON": _v(COG, "language", "Boston Naming Test (30 items); v1-2", "C1", kind="continuous", missing=TEST),
    "MINTTOTS": _v(COG, "language", "Multilingual Naming Test total; v3", "C2", kind="continuous", missing=TEST),
    "UDSVERFC": _v(COG, "language", "Letter fluency, F words; v3", "C2", kind="continuous", missing=TEST),
    # ---- visuospatial ------------------------------------------------------
    "UDSBENTC": _v(COG, "visuospatial", "Benson figure copy; v3", "C2", kind="continuous", missing=TEST),
    # ---- function, mood, behaviour ----------------------------------------
    "NACCGDS": _v(COG, "mood", "Geriatric Depression Scale total (0-15)", "B6", kind="continuous", missing=[-4, 88]),
    "BILLS": _v(COG, "function (FAQ)", "FAQ: paying bills (0-3)", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "TAXES": _v(COG, "function (FAQ)", "FAQ: taxes and business affairs", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "SHOPPING": _v(COG, "function (FAQ)", "FAQ: shopping alone", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "GAMES": _v(COG, "function (FAQ)", "FAQ: games or hobbies", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "STOVE": _v(COG, "function (FAQ)", "FAQ: using the stove", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "MEALPREP": _v(COG, "function (FAQ)", "FAQ: preparing a meal", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "EVENTS": _v(COG, "function (FAQ)", "FAQ: keeping track of current events", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "PAYATTN": _v(COG, "function (FAQ)", "FAQ: paying attention to a programme or book", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "REMDATES": _v(COG, "function (FAQ)", "FAQ: remembering dates and appointments", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "TRAVEL": _v(COG, "function (FAQ)", "FAQ: travelling out of the neighbourhood", "B7", "proximal", "ordinal", [-4, 8, 9]),
    "DECSUB": _v(COG, "complaint", "Participant reports memory decline", "B9", "proximal", missing=[-4, 8, 9]),
    "DECIN": _v(COG, "complaint", "Informant reports memory decline", "B9", "proximal", missing=[-4, 8, 9]),
}

FAQ_ITEMS = ["BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE", "MEALPREP",
             "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"]
CDR_BOXES = ["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"]
DX_LABELS = {1: "Normal", 2: "Impaired, not MCI", 3: "MCI", 4: "Dementia"}

# PET tracer codes. Codes 1-7 were confirmed by decoding the DICOM headers of
# the local images in the data audit.
TRACERS = {1: "FDG", 2: "PiB", 3: "Florbetapir", 4: "Florbetaben", 5: "NAV4694",
           6: "Flortaucipir", 7: "MK-6240"}

# PET variables use different spellings in the SCAN and mixed-protocol files,
# so they are described here once under a common name.
PET_VARIABLES = {
    "CENTILOIDS": "Amyloid burden on the Centiloid scale: 0 = young controls, 100 = typical Alzheimer's dementia. The only amyloid number that is comparable across tracers.",
    "AMYLOID_STATUS": "Amyloid positive (1) or negative (0), set by the PET core from a tracer-specific cut-off.",
    "SUMMARY_SUVR": "Cortical summary SUVR (uptake relative to a reference region). Tracer-specific, do not pool across tracers.",
    "META_TEMPORAL_SUVR": "Tau PET: SUVR in the meta-temporal region (entorhinal, amygdala, fusiform, inferior and middle temporal), the standard early-tau summary.",
    "CTX_ENTORHINAL_SUVR": "Tau PET: entorhinal cortex SUVR, where tau appears first.",
    "FDGMETAROISUVR": "FDG PET: SUVR in the Alzheimer's 'meta-ROI' (low = hypometabolism).",
    "regional *_SUVR": "About 160 FreeSurfer regions (bilateral, left, right) per scan.",
    "TRACER": "Radiotracer code: " + ", ".join(f"{k} {v}" for k, v in TRACERS.items()),
    "QC": "Quality-control result of the scan (1 = pass).",
}

# ---------------------------------------------------------------------------
# Privacy guard
# ---------------------------------------------------------------------------

def assert_no_ids(text: str, where: str = "output") -> None:
    """Stop if text that is about to be shared contains a NACCID."""
    if NACCID_RE.search(text):
        raise ValueError(f"{where} contains something shaped like a NACCID; "
                         "only aggregates may leave data/.")


# ---------------------------------------------------------------------------
# Main clinical table
# ---------------------------------------------------------------------------

def build_uds_cache(max_missing_pct: float = 90.0) -> Path:
    """Copy the usable columns of the 1.7 GB CSV into a small parquet file.

    We keep every column that has data in at least 10% of rows (787 of 2,649).
    Values are copied exactly as they are, missing codes included; cleaning is
    a separate, visible step (see clean()).
    """
    import pyarrow as pa
    import pyarrow.csv as pc
    import pyarrow.parquet as pq

    per_col = next(t for t in json.loads(AUDIT_JSON.read_text())["tables"]
                   if "ftldlbd" in t["file"])["missingness"]["per_column"]
    header = list(per_col)
    keep = [c for c in header if per_col[c]["blank_or_coded_pct"] <= max_missing_pct]
    types: dict = {}
    while True:  # a column that turns out to hold text is re-read as text
        try:
            table = pc.read_csv(
                UDS_CSV, read_options=pc.ReadOptions(block_size=64 << 20),
                convert_options=pc.ConvertOptions(include_columns=keep, column_types=types,
                                                  strings_can_be_null=True))
            break
        except pa.ArrowInvalid as err:
            col = header[int(re.search(r"column #(\d+)", str(err)).group(1))]
            types[col] = pa.string()
    INTERIM.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, UDS_CACHE, compression="zstd")
    return UDS_CACHE


def load_uds(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the clinical table (one row per visit), sorted by person and date."""
    if not UDS_CACHE.exists():
        build_uds_cache()
    if columns is not None:
        columns = list(dict.fromkeys(["NACCID", "VISITDATE", *columns]))
    df = pd.read_parquet(UDS_CACHE, columns=columns)
    df["VISITDATE"] = pd.to_datetime(df["VISITDATE"])
    return df.sort_values(["NACCID", "VISITDATE"]).reset_index(drop=True)


def clean(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Replace each variable's own missing codes with NaN. Returns a copy."""
    out = df.copy()
    for col in columns or [c for c in df.columns if c in VARIABLES]:
        spec = VARIABLES.get(col)
        if spec is None or spec["kind"] in ("id", "date") or col not in out:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        out[col] = s.mask(s.isin(spec["missing"]))
    return out


def add_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add years since each participant's first visit."""
    out = df.copy()
    first = out.groupby("NACCID")["VISITDATE"].transform("min")
    out["years"] = (out["VISITDATE"] - first).dt.days / 365.25
    return out


# ---------------------------------------------------------------------------
# PET tables
# ---------------------------------------------------------------------------

def _std(df: pd.DataFrame) -> pd.DataFrame:
    """Give the SCAN and mixed-protocol files the same column spellings."""
    ren = {"AMYLOIDSTATUS": "AMYLOID_STATUS", "GAAINSUMMARYSUVR": "SUMMARY_SUVR",
           "GAAIN_SUMMARY_SUVR": "SUMMARY_SUVR", "METATEMPORALSUVR": "META_TEMPORAL_SUVR",
           "CTXENTORHINALSUVR": "CTX_ENTORHINAL_SUVR", "QCSTATUS": "QC", "QC_IMAGE": "QC"}
    df = df.rename(columns=ren)
    df["SCANDATE"] = pd.to_datetime(df["SCANDATE"])
    return df


def load_pet_amyloid() -> pd.DataFrame:
    """Every amyloid PET scan with a Centiloid / summary SUVR row."""
    keep = ["NACCID", "NACCADC", "SCANDATE", "TRACER", "QC", "AMYLOID_STATUS",
            "CENTILOIDS", "SUMMARY_SUVR", "source"]
    a = _std(pd.read_csv(SCAN_DIR / "investigator_scan_clariti_amyloidpetgaain_nacc74.csv")).assign(source="SCAN")
    b = _std(pd.read_csv(MP_DIR / "investigator_scan_mp_amyloidpetgaain_nacc74.csv")).assign(source="Mixed protocol")
    out = pd.concat([a[keep], b[keep]], ignore_index=True)
    out["AMYLOID_STATUS"] = out["AMYLOID_STATUS"].where(out["AMYLOID_STATUS"].isin([0, 1]))
    return out


def load_pet_tau() -> pd.DataFrame:
    """Every tau PET scan with a meta-temporal SUVR row."""
    keep = ["NACCID", "NACCADC", "SCANDATE", "TRACER", "QC", "META_TEMPORAL_SUVR",
            "CTX_ENTORHINAL_SUVR", "source"]
    a = _std(pd.read_csv(SCAN_DIR / "investigator_scan_clariti_taupetnpdka_nacc74.csv")).assign(source="SCAN")
    b = _std(pd.read_csv(MP_DIR / "investigator_scan_mp_taupetnpdka_nacc74.csv")).assign(source="Mixed protocol")
    return pd.concat([a[keep], b[keep]], ignore_index=True)


def load_pet_fdg() -> pd.DataFrame:
    df = _std(pd.read_csv(SCAN_DIR / "investigator_scan_clariti_fdgpetnpdka_nacc74.csv"))
    return df[["NACCID", "NACCADC", "SCANDATE", "TRACER", "QC", "FDGMETAROISUVR"]].assign(source="SCAN")


def load_pet_images() -> pd.DataFrame:
    """One row per local PET image folder (from the data audit)."""
    df = pd.read_csv(PET_INVENTORY, parse_dates=["SCANDATE"])
    return df.rename(columns={"naccid": "NACCID"})


def first_scan(scans: pd.DataFrame) -> pd.DataFrame:
    """Earliest scan per participant."""
    return scans.sort_values(["NACCID", "SCANDATE"]).groupby("NACCID", as_index=False).first()


def index_visit(scans: pd.DataFrame, uds: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """Attach to each participant's first scan the clinic visit closest in time.

    Returns one row per participant with the scan columns, the index visit's
    columns (prefixed "i_") and the gap in days.
    """
    scan = first_scan(scans)
    # the centre is taken from the clinical table, not the scan table
    scan = scan.drop(columns=[c for c in scan.columns if c in uds.columns and c != "NACCID"])
    m = scan.merge(uds, on="NACCID")
    m["gap_days"] = (m["VISITDATE"] - m["SCANDATE"]).dt.days.abs()
    m = m[m["gap_days"] <= window_days].sort_values("gap_days").groupby("NACCID", as_index=False).first()
    return m.rename(columns={c: f"i_{c}" for c in uds.columns if c != "NACCID"})


def followup_events(index: pd.DataFrame, uds: pd.DataFrame, min_years: float = 0.25) -> pd.DataFrame:
    """Per participant: what happened at visits after the index visit."""
    m = index[["NACCID", "i_VISITDATE", "i_NACCUDSD", "i_CDRSUM", "i_CDRGLOB"]].merge(
        uds[["NACCID", "VISITDATE", "NACCUDSD", "CDRSUM", "CDRGLOB"]], on="NACCID")
    m["years"] = (m["VISITDATE"] - m["i_VISITDATE"]).dt.days / 365.25
    m = m[m["years"] > min_years]
    m["d_cdrsb"] = m["CDRSUM"] - m["i_CDRSUM"]
    g = m.groupby("NACCID")
    return pd.DataFrame({
        "i_dx": g["i_NACCUDSD"].first(),
        "followup_years": g["years"].max(),
        "worst_dx": g["NACCUDSD"].max(),
        "max_cdrsb_increase": g["d_cdrsb"].max(),
    }).reset_index()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# One fixed colour order, used the same way in every figure.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
PALETTE = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
DX_COLORS = {1: BLUE, 2: YELLOW, 3: ORANGE, 4: RED}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def set_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "figure.dpi": 110, "savefig.dpi": 130, "font.size": 10,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e", "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "legend.frameon": False, "lines.linewidth": 2,
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
    })


def savefig(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 1) if d else float("nan")
