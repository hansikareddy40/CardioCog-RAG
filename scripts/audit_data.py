"""Audit the raw NACC data under data/raw and write an aggregate-only report.

Outputs
-------
reports/data_audit_summary.md      aggregate summary, safe to share
reports/data_audit_aggregate.json  the same numbers, machine readable
data/interim/pet_scan_inventory.csv  one row per PET scan folder (participant
                                     level, stays under the gitignored data/)

Nothing participant-level is ever written to reports/: the script refuses to
save a report that contains anything shaped like a NACCID.

Usage
-----
    python scripts/audit_data.py
    python scripts/audit_data.py --skip-pet        # tables only
    python scripts/audit_data.py --data-dir D:/somewhere/raw
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

# Small columns kept in memory for id / date / category summaries.
ID_COL = "NACCID"
DATE_COL_RE = re.compile(r"(DATE|DT\d?)$")
CATEGORY_COLS = [
    "TRACER", "RADIOTRACER", "PROJECT", "SCAN_PROJECT", "VISIT", "QCSTATUS",
    "QC_IMAGE", "PASSFAIL", "AMYLOIDSTATUS", "AMYLOID_STATUS", "DYNAMIC",
    "PACKET", "NACCUDSD",
]
EXTRA_KEY_COLS = ["NACCADC", "LONIUID", "NACCVNUM"]
# NACC "not available / not applicable" codes, counted separately from blanks.
NACC_MISSING_CODES = ["-4", "-4.4", "-4.0"]

PET_DIR_RE = re.compile(r"^SCAN_(NACC\d+)_I(\d+)_(.*)$")
NACCID_RE = re.compile(r"NACC\d{4,}")

TRACER_PATTERNS = [
    # (regex on lowercased text, tracer name, class)
    (r"fdg|fluorodeoxyglucose", "FDG", "fdg"),
    (r"pib|pittsburgh", "PiB", "amyloid"),
    (r"florbetapir|av-?45|amyvid", "Florbetapir", "amyloid"),
    (r"florbetaben|neuraceq|fbb", "Florbetaben", "amyloid"),
    (r"nav-?4694|flutafuranol", "NAV4694", "amyloid"),
    (r"flutemetamol|vizamyl", "Flutemetamol", "amyloid"),
    (r"mk-?6240", "MK-6240", "tau"),
    (r"flortaucipir|av-?1451|t807|tauvid", "Flortaucipir", "tau"),
    (r"pi-?2620", "PI-2620", "tau"),
    (r"gtp-?1", "GTP1", "tau"),
]


def log(msg: str) -> None:
    print(f"[audit] {msg}", file=sys.stderr, flush=True)


def pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def month(ts) -> str | None:
    """Dates are reported at month resolution only."""
    return None if pd.isna(ts) else pd.Timestamp(ts).strftime("%Y-%m")


def describe(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return {}
    return {
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "mean": round(float(a.mean()), 2),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
    }


def counts(series: pd.Series) -> dict:
    vc = series.fillna("(blank)").astype(str).value_counts()
    return {str(k): int(v) for k, v in vc.items()}


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------

def audit_table(path: Path, chunksize: int) -> tuple[dict, pd.DataFrame]:
    """One streaming pass over a CSV. Returns (aggregates, key-column frame)."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    key_cols = [
        c for c in header
        if c == ID_COL or c in CATEGORY_COLS or c in EXTRA_KEY_COLS
        or DATE_COL_RE.search(c)
    ]
    n_rows = 0
    blank = pd.Series(0, index=header, dtype="int64")
    coded = pd.Series(0, index=header, dtype="int64")
    keys = []
    reader = pd.read_csv(
        path, dtype=str, chunksize=chunksize, keep_default_na=False,
        na_values=[""], encoding="utf-8", encoding_errors="replace",
    )
    for chunk in reader:
        n_rows += len(chunk)
        blank += chunk.isna().sum()
        coded += chunk.isin(NACC_MISSING_CODES).sum()
        keys.append(chunk[key_cols].copy())
        if n_rows % (chunksize * 5) == 0:
            log(f"  {path.name}: {n_rows:,} rows")
    keys = pd.concat(keys, ignore_index=True) if keys else pd.DataFrame(columns=key_cols)

    blank_pct = (blank / max(n_rows, 1) * 100).round(2)
    any_missing_pct = ((blank + coded) / max(n_rows, 1) * 100).round(2)
    out = {
        "file": path.name,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "rows": n_rows,
        "columns": len(header),
        "has_naccid": ID_COL in header,
        "missingness": {
            "note": "blank = empty cell; coded = NACC -4 / -4.4 (not available)",
            "overall_blank_pct": pct(int(blank.sum()), n_rows * len(header)),
            "overall_blank_or_coded_pct": pct(int((blank + coded).sum()), n_rows * len(header)),
            "columns_fully_missing": int((any_missing_pct >= 100).sum()),
            "columns_over_90pct_missing": int((any_missing_pct > 90).sum()),
            "columns_over_50pct_missing": int((any_missing_pct > 50).sum()),
            "columns_complete": int((any_missing_pct == 0).sum()),
            "median_column_missing_pct": float(any_missing_pct.median()),
            "per_column": {
                c: {"blank_pct": float(blank_pct[c]), "blank_or_coded_pct": float(any_missing_pct[c])}
                for c in header
            },
        },
    }

    if ID_COL in keys:
        per_id = keys[ID_COL].value_counts()
        out["unique_naccids"] = int(per_id.size)
        out["rows_per_participant"] = describe(per_id.values)
        buckets = pd.cut(
            per_id, [0, 1, 2, 3, 4, 5, 10, np.inf],
            labels=["1", "2", "3", "4", "5", "6-10", "11+"],
        ).value_counts().sort_index()
        out["rows_per_participant_histogram"] = {str(k): int(v) for k, v in buckets.items()}
    if "NACCADC" in keys:
        out["unique_adcs"] = int(keys["NACCADC"].nunique())

    out["date_ranges"] = {}
    for c in [c for c in keys.columns if DATE_COL_RE.search(c)]:
        d = pd.to_datetime(keys[c], errors="coerce")
        if d.notna().any():
            out["date_ranges"][c] = {
                "min": month(d.min()), "max": month(d.max()),
                "parsed_pct": pct(int(d.notna().sum()), n_rows),
            }
    out["categories"] = {
        c: counts(keys[c]) for c in CATEGORY_COLS
        if c in keys and keys[c].nunique(dropna=False) <= 40
    }
    return out, keys


# --------------------------------------------------------------------------
# PET images
# --------------------------------------------------------------------------

def classify_tracer(*texts: str) -> tuple[str, str]:
    for text in texts:
        t = (text or "").lower()
        for pattern, name, klass in TRACER_PATTERNS:
            if re.search(pattern, t):
                return name, klass
    return "unknown", "unknown"


def read_header(path: str) -> dict:
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)

    def get(name, default=None):
        v = getattr(ds, name, default)
        return default if v in (None, "") else v

    radio = ""
    seq = get("RadiopharmaceuticalInformationSequence")
    if seq:
        radio = str(getattr(seq[0], "Radiopharmaceutical", "") or "")
    spacing = get("PixelSpacing") or [np.nan, np.nan]
    return {
        "dicom_modality": str(get("Modality", "")),
        "manufacturer": str(get("Manufacturer", "")).strip(),
        "scanner_model": str(get("ManufacturerModelName", "")).strip(),
        "series_description": str(get("SeriesDescription", "")),
        "rows": int(get("Rows", 0)),
        "cols": int(get("Columns", 0)),
        "n_slices": int(get("NumberOfSlices", 0)),
        "n_frames": int(get("NumberOfTimeSlices", 1) or 1),
        "pixel_spacing_x": float(spacing[0]),
        "pixel_spacing_y": float(spacing[1]),
        "slice_thickness": float(get("SliceThickness", np.nan)),
        "units": str(get("Units", "")),
        "series_type": "/".join(map(str, get("SeriesType", []) or [])),
        "reconstruction": str(get("ReconstructionMethod", "")),
        "attenuation_corrected": "ATTN" in list(get("CorrectedImage", []) or []),
        "bits_allocated": int(get("BitsAllocated", 0)),
        "radiopharmaceutical": radio,
        "patient_id": str(get("PatientID", "")),
        "patient_name": str(get("PatientName", "")),
        "has_birth_date": bool(get("PatientBirthDate")),
    }


def inventory_pet(pet_root: Path) -> pd.DataFrame:
    """One row per scan folder; reads a single DICOM header per folder."""
    rows = []
    for mod_dir in sorted(p for p in pet_root.iterdir() if p.is_dir()):
        entries = list(os.scandir(mod_dir))
        zips = {e.name[:-4]: e.stat().st_size for e in entries if e.name.lower().endswith(".zip")}
        dirs = [e for e in entries if e.is_dir()]
        log(f"PET {mod_dir.name}: {len(dirs)} scan folders, {len(zips)} zips")
        for i, d in enumerate(sorted(dirs, key=lambda e: e.name), 1):
            m = PET_DIR_RE.match(d.name)
            n_files, n_bytes, first, exts = 0, 0, None, Counter()
            for root, _, files in os.walk(d.path):
                for f in files:
                    n_files += 1
                    exts[os.path.splitext(f)[1].lower() or "(none)"] += 1
                    fp = os.path.join(root, f)
                    n_bytes += os.path.getsize(fp)
                    if first is None and f.lower().endswith((".dcm", ".ima")):
                        first = fp
            row = {
                "folder_modality": mod_dir.name,
                "naccid": m.group(1) if m else "",
                "image_id": m.group(2) if m else "",
                "folder_description": m.group(3) if m else "",
                "name_parsed": bool(m),
                "n_files": n_files,
                "size_mb": round(n_bytes / 1e6, 1),
                "file_format": "DICOM" if exts.get(".dcm") else (max(exts, key=exts.get) if exts else "empty"),
                "has_zip_twin": d.name in zips,
                "zip_size_mb": round(zips.get(d.name, 0) / 1e6, 1),
                "header_ok": False,
            }
            if first:
                try:
                    row.update(read_header(first))
                    row["header_ok"] = True
                except Exception as exc:  # keep going, count failures
                    row["header_error"] = type(exc).__name__
            rows.append(row)
            if i % 50 == 0:
                log(f"  {mod_dir.name}: {i}/{len(dirs)}")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    tr = df.apply(
        lambda r: classify_tracer(r.get("radiopharmaceutical", ""), r.get("series_description", ""),
                                  r["folder_description"]),
        axis=1, result_type="expand",
    )
    df["tracer"], df["tracer_class"] = tr[0], tr[1]
    df["expected_files"] = df["n_slices"] * df["n_frames"]
    df["file_count_matches"] = df["expected_files"].eq(df["n_files"]) & df["expected_files"].gt(0)
    df["matrix"] = df.apply(lambda r: f"{r['rows']:.0f}x{r['cols']:.0f}x{r['n_slices']:.0f}"
                            if r["header_ok"] else "unreadable", axis=1)
    df["voxel_mm"] = df.apply(
        lambda r: f"{r['pixel_spacing_x']:.2f}x{r['pixel_spacing_y']:.2f}x{r['slice_thickness']:.2f}"
        if r["header_ok"] else "unreadable", axis=1)
    return df


def summarise_pet(df: pd.DataFrame) -> dict:
    out = {"scan_folders": int(len(df)), "by_folder": {}}
    folder_class = {"amyloid": "amyloid", "tau": "tau", "fdg": "fdg"}
    for mod, g in df.groupby("folder_modality"):
        ok = g[g["header_ok"]]
        expected = folder_class.get(mod.lower())
        out["by_folder"][mod] = {
            "scan_folders": int(len(g)),
            "unique_participants": int(g["naccid"].nunique()),
            "participants_with_multiple_scans": int((g["naccid"].value_counts() > 1).sum()),
            "folder_names_parsed": int(g["name_parsed"].sum()),
            "headers_readable": int(len(ok)),
            "file_format": counts(g["file_format"]),
            "dicom_modality": counts(ok["dicom_modality"]),
            "files_total": int(g["n_files"].sum()),
            "size_gb_extracted": round(float(g["size_mb"].sum()) / 1e3, 1),
            "size_gb_zips": round(float(g["zip_size_mb"].sum()) / 1e3, 1),
            "folders_with_zip_twin": int(g["has_zip_twin"].sum()),
            "tracer": counts(ok["tracer"]),
            "tracer_class": counts(ok["tracer_class"]),
            "tracer_class_agrees_with_folder": int((ok["tracer_class"] == expected).sum()),
            "matrix_rows_x_cols_x_slices": counts(ok["matrix"]),
            "voxel_spacing_mm": counts(ok["voxel_mm"]),
            "in_plane_spacing_mm": describe(ok["pixel_spacing_x"].dropna()),
            "slice_thickness_mm": describe(ok["slice_thickness"].dropna()),
            "frames_per_scan": counts(ok["n_frames"]),
            "files_per_scan": describe(g["n_files"]),
            "file_count_equals_slices_x_frames": int(ok["file_count_matches"].sum()),
            "series_type": counts(ok["series_type"]),
            "units": counts(ok["units"]),
            "attenuation_corrected": int(ok["attenuation_corrected"].sum()),
            "manufacturer": counts(ok["manufacturer"]),
            "scanner_model": counts(ok["scanner_model"]),
            "deidentification": {
                "patient_id_equals_folder_naccid": int((ok["patient_id"] == ok["naccid"]).sum()),
                "patient_name_deidentified": int(ok["patient_name"].str.upper().str.contains("DE-IDENT").sum()),
                "headers_with_birth_date": int(ok["has_birth_date"].sum()),
            },
        }
    return out


# --------------------------------------------------------------------------
# Linkage between images and tables
# --------------------------------------------------------------------------

def norm_uid(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"^I", "", regex=True)


def linkage(pet: pd.DataFrame, keys: dict[str, pd.DataFrame], uds_name: str | None) -> dict:
    out: dict = {}
    sets = {m.lower(): set(g["naccid"]) - {""} for m, g in pet.groupby("folder_modality")}
    any_pet = set().union(*sets.values()) if sets else set()
    amy, tau, fdg = sets.get("amyloid", set()), sets.get("tau", set()), sets.get("fdg", set())
    out["participants_with_images"] = {
        "any_pet": len(any_pet), "amyloid": len(amy), "tau": len(tau), "fdg": len(fdg),
        "amyloid_and_tau": len(amy & tau), "amyloid_only": len(amy - tau - fdg),
        "tau_only": len(tau - amy - fdg), "all_three": len(amy & tau & fdg),
    }

    out["participants_with_images_found_in_table"] = {}
    for name, k in keys.items():
        if ID_COL not in k:
            continue
        ids = set(k[ID_COL].dropna())
        out["participants_with_images_found_in_table"][name] = {
            "any_pet": len(any_pet & ids), "amyloid": len(amy & ids), "tau": len(tau & ids),
            "fdg": len(fdg & ids), "amyloid_and_tau": len(amy & tau & ids),
        }

    # Image-level link: folder image id -> PET QC table (carries scan date).
    qc_name = next((n for n in keys if "petqc" in n), None)
    scans = pet[["folder_modality", "naccid", "image_id", "tracer", "tracer_class"]].copy()
    if qc_name:
        qc = keys[qc_name].copy()
        qc["image_id"] = norm_uid(qc["LONIUID"])
        qc = qc.drop_duplicates("image_id")
        scans = scans.merge(
            qc[["image_id", "SCANDATE", "RADIOTRACER", "PASSFAIL", "PROJECT"]],
            on="image_id", how="left")
        hit = scans["SCANDATE"].notna()
        out["images_linked_to_petqc"] = {
            "table": qc_name, "linked": int(hit.sum()), "of": int(len(scans)),
            "by_folder": {m: int(g["SCANDATE"].notna().sum()) for m, g in scans.groupby("folder_modality")},
            "qc_passfail_of_local_images": counts(scans.loc[hit, "PASSFAIL"]),
            "project_of_local_images": counts(scans.loc[hit, "PROJECT"]),
            "scan_date_range": {
                "min": month(pd.to_datetime(scans["SCANDATE"], errors="coerce").min()),
                "max": month(pd.to_datetime(scans["SCANDATE"], errors="coerce").max()),
            },
            # Empirical decoding of the numeric tracer code from DICOM headers.
            "radiotracer_code_vs_dicom_tracer": {
                str(code): counts(g["tracer"]) for code, g in scans[hit].groupby("RADIOTRACER")
            },
        }

        # Image -> quantification (SUVR) rows, joined on participant + scan date.
        out["images_with_quantification_row"] = {}
        for name, k in keys.items():
            if name == qc_name or not {"SCANDATE", ID_COL} <= set(k.columns) or "TRACER" not in k:
                continue
            pairs = set(zip(k[ID_COL], k["SCANDATE"]))
            flag = [(a, b) in pairs for a, b in zip(scans["naccid"], scans["SCANDATE"])]
            out["images_with_quantification_row"][name] = {
                m: int(np.sum(np.asarray(flag)[(scans["folder_modality"] == m).values]))
                for m in sorted(scans["folder_modality"].unique())
            }

        # Image -> nearest clinical visit.
        if uds_name and "VISITDATE" in keys[uds_name]:
            uds = keys[uds_name][[ID_COL, "VISITDATE"]].copy()
            uds["VISITDATE"] = pd.to_datetime(uds["VISITDATE"], errors="coerce")
            uds = uds.dropna()
            s = scans[hit].copy()
            s["scan_dt"] = pd.to_datetime(s["SCANDATE"], errors="coerce")
            j = s.merge(uds, left_on="naccid", right_on=ID_COL, how="inner")
            j["gap"] = (j["VISITDATE"] - j["scan_dt"]).dt.days.abs()
            gap = j.groupby("image_id")["gap"].min()
            s = s.merge(gap.rename("nearest_visit_gap_days"), on="image_id", how="left")
            scans = scans.merge(s[["image_id", "nearest_visit_gap_days"]], on="image_id", how="left")
            g = s["nearest_visit_gap_days"].dropna()
            out["nearest_clinical_visit_to_scan"] = {
                "table": uds_name,
                "images_with_any_visit": int(g.size),
                "abs_gap_days": describe(g),
                "within_90_days": int((g <= 90).sum()),
                "within_180_days": int((g <= 180).sum()),
                "within_365_days": int((g <= 365).sum()),
                "participants_with_visit_within_365_days": {
                    m: int(x.loc[x["nearest_visit_gap_days"] <= 365, "naccid"].nunique())
                    for m, x in s.groupby("folder_modality")
                },
                "participants_amyloid_and_tau_each_within_365_days": len(
                    set(s.loc[(s["folder_modality"].str.lower() == "amyloid")
                              & (s["nearest_visit_gap_days"] <= 365), "naccid"])
                    & set(s.loc[(s["folder_modality"].str.lower() == "tau")
                                & (s["nearest_visit_gap_days"] <= 365), "naccid"])),
            }
            vis = keys[uds_name][ID_COL].value_counts()
            out["clinical_visits_among_pet_participants"] = describe(
                vis.reindex(sorted(any_pet & set(vis.index))).values)
    out["_scans"] = scans  # stripped before saving; goes to data/interim only
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def fmt_counts(d: dict, top: int = 8) -> str:
    items = sorted(d.items(), key=lambda kv: -kv[1])
    s = ", ".join(f"{k} ({v})" for k, v in items[:top])
    return s + (f", +{len(items) - top} more" if len(items) > top else "")


def build_markdown(agg: dict) -> str:
    L = ["# NACC data audit (aggregate only)", "",
         f"Generated {agg['generated']} by `scripts/audit_data.py`. "
         "Counts, ranges and percentages only; no participant-level values. "
         "Dates are shown to the month.", ""]

    L += ["## Tables", ""]
    rows = []
    for t in agg["tables"]:
        rp = t.get("rows_per_participant", {})
        dates = "; ".join(f"{c} {v['min']} to {v['max']}" for c, v in list(t["date_ranges"].items())[:3]) or "n/a"
        rows.append([
            f"`{t['file']}`", f"{t['rows']:,}", f"{t['columns']:,}", f"{t.get('unique_naccids', 0):,}",
            f"{rp.get('median', 0):g} / {rp.get('mean', 0):g} / {rp.get('max', 0):g}",
            dates, f"{t['missingness']['overall_blank_or_coded_pct']}%",
        ])
    L += [md_table(["File", "Rows", "Columns", "Unique NACCIDs", "Rows per participant (median / mean / max)",
                    "Date range", "Cells missing"], rows), "",
          "Cells missing = blank cells plus NACC not-available codes (-4, -4.4).", ""]

    L += ["### Missingness by column", ""]
    rows = []
    for t in agg["tables"]:
        m = t["missingness"]
        rows.append([f"`{t['file']}`", f"{m['overall_blank_pct']}%", m["columns_complete"],
                     m["columns_over_50pct_missing"], m["columns_over_90pct_missing"],
                     m["columns_fully_missing"], f"{m['median_column_missing_pct']}%"])
    L += [md_table(["File", "Blank cells", "Complete columns", "Columns >50% missing",
                    "Columns >90% missing", "Columns 100% missing", "Median column missing"], rows), "",
          "Per-column figures are in `data_audit_aggregate.json`.", ""]

    L += ["### Rows per participant and key categories", ""]
    for t in agg["tables"]:
        L.append(f"**`{t['file']}`**")
        if "rows_per_participant_histogram" in t:
            L.append("- Rows per participant: " + ", ".join(
                f"{k}: {v:,}" for k, v in t["rows_per_participant_histogram"].items()))
        if "unique_adcs" in t:
            L.append(f"- Centers (ADCs): {t['unique_adcs']}")
        for c, d in t["categories"].items():
            L.append(f"- {c}: {fmt_counts(d, 12)}")
        L.append("")

    pet = agg.get("pet")
    if pet:
        L += ["## PET images", "", f"{pet['scan_folders']} scan folders in total.", ""]
        rows = []
        for m, p in pet["by_folder"].items():
            rows.append([m, p["scan_folders"], p["unique_participants"], fmt_counts(p["file_format"]),
                         f"{p['files_total']:,}", p["size_gb_extracted"], p["size_gb_zips"],
                         f"{p['tracer_class_agrees_with_folder']}/{p['headers_readable']}"])
        L += [md_table(["Folder", "Scans", "Participants", "Format", "Files", "Extracted GB", "Zip GB",
                        "Header tracer class agrees with folder"], rows), ""]
        for m, p in pet["by_folder"].items():
            sp, st = p["in_plane_spacing_mm"], p["slice_thickness_mm"]
            L += [f"### {m}", "",
                  f"- Tracers: {fmt_counts(p['tracer'])}",
                  f"- Matrix (rows x cols x slices): {fmt_counts(p['matrix_rows_x_cols_x_slices'])}",
                  f"- Voxel spacing mm (x by y by slice): {fmt_counts(p['voxel_spacing_mm'], 6)}",
                  f"- In-plane spacing: {sp.get('min', 0):.2f} to {sp.get('max', 0):.2f} mm "
                  f"(median {sp.get('median', 0):.2f}); slice thickness: {st.get('min', 0):.2f} to "
                  f"{st.get('max', 0):.2f} mm (median {st.get('median', 0):.2f})",
                  f"- Frames per scan: {fmt_counts(p['frames_per_scan'])}",
                  f"- Files per scan: median {p['files_per_scan'].get('median', 0):g}, range "
                  f"{p['files_per_scan'].get('min', 0):g} to {p['files_per_scan'].get('max', 0):g}; "
                  f"file count equals slices x frames in {p['file_count_equals_slices_x_frames']}/{p['headers_readable']}",
                  f"- Series type: {fmt_counts(p['series_type'])}; units: {fmt_counts(p['units'])}; "
                  f"attenuation corrected: {p['attenuation_corrected']}/{p['headers_readable']}",
                  f"- Scanners: {fmt_counts(p['manufacturer'])}; models: {fmt_counts(p['scanner_model'], 6)}",
                  f"- Participants with more than one scan: {p['participants_with_multiple_scans']}",
                  f"- De-identification: PatientName de-identified in "
                  f"{p['deidentification']['patient_name_deidentified']}/{p['headers_readable']}, "
                  f"birth date present in {p['deidentification']['headers_with_birth_date']}", ""]

    link = agg.get("linkage")
    if link:
        pw = link["participants_with_images"]
        L += ["## PET images matched to tables", "",
              f"- Participants with any PET image: **{pw['any_pet']}** "
              f"(amyloid {pw['amyloid']}, tau {pw['tau']}, FDG {pw['fdg']})",
              f"- Amyloid and tau both: **{pw['amyloid_and_tau']}**; all three: {pw['all_three']}; "
              f"amyloid only: {pw['amyloid_only']}; tau only: {pw['tau_only']}", ""]
        rows = [[f"`{n}`", v["any_pet"], v["amyloid"], v["tau"], v["fdg"], v["amyloid_and_tau"]]
                for n, v in link["participants_with_images_found_in_table"].items()]
        L += ["Participants with a local image who also appear in each table:", "",
              md_table(["Table", "Any PET", "Amyloid", "Tau", "FDG", "Amyloid and tau"], rows), ""]
        if "images_linked_to_petqc" in link:
            q = link["images_linked_to_petqc"]
            L += [f"- Image IDs found in `{q['table']}`: {q['linked']}/{q['of']}",
                  f"- QC result of local images: {fmt_counts(q['qc_passfail_of_local_images'])}",
                  f"- Project of local images: {fmt_counts(q['project_of_local_images'])}",
                  f"- Scan dates of local images: {q['scan_date_range']['min']} to {q['scan_date_range']['max']}",
                  "- Tracer code decoded from DICOM headers: " + "; ".join(
                      f"{code} = {fmt_counts(d, 3)}" for code, d in
                      sorted(q["radiotracer_code_vs_dicom_tracer"].items(), key=lambda kv: kv[0].zfill(3))), ""]
        if link.get("images_with_quantification_row"):
            mods = sorted(next(iter(link["images_with_quantification_row"].values())).keys())
            rows = [[f"`{n}`"] + [v[m] for m in mods] for n, v in link["images_with_quantification_row"].items()]
            L += ["Local images that have a SUVR row for the same participant and scan date:", "",
                  md_table(["Table"] + mods, rows), ""]
        if "nearest_clinical_visit_to_scan" in link:
            n = link["nearest_clinical_visit_to_scan"]
            g = n["abs_gap_days"]
            L += [f"Nearest clinical visit in `{n['table']}` to each scan:", "",
                  f"- Images with any visit for that participant: {n['images_with_any_visit']}",
                  f"- Gap in days: median {g.get('median', 0):g}, p75 {g.get('p75', 0):g}, max {g.get('max', 0):g}",
                  f"- Within 90 / 180 / 365 days: {n['within_90_days']} / {n['within_180_days']} / {n['within_365_days']}",
                  "- Participants with a visit within 365 days of the scan: " + ", ".join(
                      f"{m} {v}" for m, v in n["participants_with_visit_within_365_days"].items()),
                  f"- Participants with amyloid and tau scans each within 365 days of a visit: "
                  f"**{n['participants_amyloid_and_tau_each_within_365_days']}**", ""]
        if "clinical_visits_among_pet_participants" in link:
            c = link["clinical_visits_among_pet_participants"]
            L += [f"- Clinical visits per PET participant: median {c.get('median', 0):g}, "
                  f"range {c.get('min', 0):g} to {c.get('max', 0):g}", ""]
    return "\n".join(L) + "\n"


def assert_aggregate_only(text: str, what: str) -> None:
    if NACCID_RE.search(text):
        raise SystemExit(f"REFUSING to write {what}: it contains something shaped like a NACCID.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--interim-dir", type=Path, default=REPO_ROOT / "data" / "interim")
    ap.add_argument("--chunksize", type=int, default=5000)
    ap.add_argument("--skip-pet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    agg: dict = {"generated": time.strftime("%Y-%m-%d"), "tables": []}
    keys: dict[str, pd.DataFrame] = {}
    for path in sorted(args.data_dir.rglob("*.csv")):
        log(f"table {path.name} ({path.stat().st_size / 1e6:,.0f} MB)")
        summary, k = audit_table(path, args.chunksize)
        agg["tables"].append(summary)
        keys[path.name] = k

    # The clinical (UDS) table is the one with visit dates.
    uds_name = next((n for n, k in keys.items() if "VISITDATE" in k.columns), None)

    pet_root = args.data_dir / "pet"
    if not args.skip_pet and pet_root.is_dir():
        pet = inventory_pet(pet_root)
        agg["pet"] = summarise_pet(pet)
        link = linkage(pet, keys, uds_name)
        scans = link.pop("_scans")
        agg["linkage"] = link
        args.interim_dir.mkdir(parents=True, exist_ok=True)
        inv = pet.drop(columns=["patient_id", "patient_name"], errors="ignore").merge(
            scans.drop(columns=["folder_modality", "naccid", "tracer", "tracer_class"]),
            on="image_id", how="left")
        inv.to_csv(args.interim_dir / "pet_scan_inventory.csv", index=False)
        log(f"wrote participant-level inventory to {args.interim_dir / 'pet_scan_inventory.csv'} (gitignored)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    js = json.dumps(agg, indent=2, default=str)
    md = build_markdown(agg)
    assert_aggregate_only(js, "data_audit_aggregate.json")
    assert_aggregate_only(md, "data_audit_summary.md")
    (args.out_dir / "data_audit_aggregate.json").write_text(js, encoding="utf-8")
    (args.out_dir / "data_audit_summary.md").write_text(md, encoding="utf-8")
    log(f"done in {time.time() - t0:,.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
