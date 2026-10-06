"""Phase 5b: amyloid SUVR from our own processed images, checked against NACC's values.

For each registered amyloid image:
    SUVR = mean uptake in the cortical target region
           / mean uptake in the whole cerebellum (reference region)

The two regions are the Centiloid project's standard masks (Klunk et al. 2015),
which are defined in MNI space. Our registered images are in a study template
space with MNI orientation and position; the masks are applied directly. A
visual overlay on the averaged image showed they sit on the cortex and the
cerebellum, and the numeric agreement below is the real test.

Why this matters: NACC's PET core computed SUVR for the same scans with a
different pipeline (MRI-based, native space). If our PET-only pipeline is
sound, the two sets of numbers should agree closely. That is independent
evidence that conversion and registration worked.

Output  data/processed/pet_image_suvr.csv         one row per scan (not committed)
        reports/phase5_pet_image_results.json     agreement statistics (aggregate)

Usage:  python scripts/pet_suvr.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

import nacc_utils as nu

MNI = nu.REPO / "data" / "processed" / "pet_mni"
VOI = nu.REPO / "data" / "external" / "centiloid" / "Centiloid_Std_VOI" / "nifti" / "2mm"
OUT = nu.REPO / "data" / "processed" / "pet_image_suvr.csv"


def load_masks(like: sitk.Image) -> dict:
    """Resample the standard masks onto the grid of our images."""
    masks = {}
    for name, file in [("cortex", "voi_ctx_2mm.nii"), ("cerebellum", "voi_WhlCbl_2mm.nii")]:
        m = sitk.Resample(sitk.ReadImage(str(VOI / file), sitk.sitkFloat32), like, sitk.Transform(), sitk.sitkNearestNeighbor, 0.0)
        masks[name] = sitk.GetArrayFromImage(m) > 0
    return masks


def main():
    reg = pd.read_csv(MNI / "registration_log.csv").query("ok and `pass` == 2 and modality == 'Amyloid'")
    inv = nu.load_pet_images()[["image_id", "NACCID", "SCANDATE", "tracer", "manufacturer"]]
    masks, rows = None, []
    for r in reg.itertuples():
        img = sitk.ReadImage(str(MNI / "Amyloid" / f"{r.image_id}.nii.gz"), sitk.sitkFloat32)
        if masks is None:
            masks = load_masks(img)
        a = sitk.GetArrayFromImage(img)
        ref = a[masks["cerebellum"]].mean()
        rows.append({"image_id": r.image_id, "image_suvr": a[masks["cortex"]].mean() / ref if ref > 0 else np.nan,
                     "cerebellum_coverage": float((a[masks["cerebellum"]] > 0).mean()), "scale_det": r.scale_det})
    df = pd.DataFrame(rows).merge(inv, on="image_id")
    raw = pd.read_csv(nu.SCAN_DIR / "investigator_scan_clariti_amyloidpetgaain_nacc74.csv", parse_dates=["SCANDATE"])
    df = df.merge(raw[["NACCID", "SCANDATE", "GAAINSUMMARYSUVR", "CENTILOIDS", "AMYLOIDSTATUS"]], on=["NACCID", "SCANDATE"], how="left")
    df.to_csv(OUT, index=False)

    def agreement(d: pd.DataFrame) -> dict:
        d = d.dropna(subset=["image_suvr", "GAAINSUMMARYSUVR"])
        if len(d) < 10:
            return {"n": int(len(d))}
        diff = d.image_suvr - d.GAAINSUMMARYSUVR
        slope, intercept = np.polyfit(d.GAAINSUMMARYSUVR, d.image_suvr, 1)
        out = {"n": int(len(d)), "pearson_r": float(pearsonr(d.image_suvr, d.GAAINSUMMARYSUVR)[0]),
               "spearman_r": float(spearmanr(d.image_suvr, d.GAAINSUMMARYSUVR)[0]), "slope": float(slope), "intercept": float(intercept),
               "mean_difference": float(diff.mean()), "limits_of_agreement": [float(diff.mean() - 1.96 * diff.std()), float(diff.mean() + 1.96 * diff.std())]}
        s = d.dropna(subset=["AMYLOIDSTATUS"])
        if s.AMYLOIDSTATUS.nunique() == 2:
            out["auroc_for_amyloid_status"] = float(roc_auc_score(s.AMYLOIDSTATUS, s.image_suvr))
            out["n_status"], out["n_positive"] = int(len(s)), int(s.AMYLOIDSTATUS.sum())
        return out

    ok = df[(df.cerebellum_coverage > 0.95)]
    res = {"scans_processed": int(len(df)), "matched_to_nacc_suvr": int(df.GAAINSUMMARYSUVR.notna().sum()),
           "excluded_cerebellum_outside_image": int(len(df) - len(ok)),
           "all": agreement(ok), "by_tracer": {t: agreement(g) for t, g in ok.groupby("tracer")},
           "by_scanner_maker": {str(m).title(): agreement(g) for m, g in ok.groupby("manufacturer")},
           "image_suvr_summary": ok.image_suvr.describe().round(3).to_dict(), "nacc_suvr_summary": ok.GAAINSUMMARYSUVR.describe().round(3).to_dict()}
    # binned scatter for the report figure (aggregate: means within bins of the NACC value)
    b = ok.dropna(subset=["image_suvr", "GAAINSUMMARYSUVR"])
    q = pd.qcut(b.GAAINSUMMARYSUVR, 12, duplicates="drop")
    g = b.groupby(q, observed=True).agg(n=("image_suvr", "size"), nacc=("GAAINSUMMARYSUVR", "mean"), image=("image_suvr", "mean"), image_sd=("image_suvr", "std"))
    res["binned"] = g.round(4).to_dict("list")
    text = json.dumps(res, indent=1)
    nu.assert_no_ids(text, "PET image results")
    (nu.REPORTS / "phase5_pet_image_results.json").write_text(text)
    print(json.dumps({k: v for k, v in res.items() if k != "binned"}, indent=1))


if __name__ == "__main__":
    main()
