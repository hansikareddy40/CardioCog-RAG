"""Phase 5a, step 1: raw dynamic PET DICOM -> one static 3D image per scan.

Each scan on disk is a "dynamic" series: the same brain volume repeated over
several time frames. To get one image we
  1. read every slice and convert stored values to activity (Bq/ml),
  2. group slices into frames and sort each frame from bottom to top,
  3. keep the frames inside the tracer's standard measurement window
     (only relevant for full dynamic scans that start at injection),
  4. average the kept frames, weighting each by its duration.

The raw DICOM files are only read. Output goes to data/processed/pet_static
(gitignored): one .nii.gz per scan plus conversion_log.csv.

Usage:  python scripts/pet_convert.py [--limit N] [--workers 4]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[1]
PET_ROOT = REPO / "data" / "raw" / "pet"
OUT = REPO / "data" / "processed" / "pet_static"
INVENTORY = REPO / "data" / "interim" / "pet_scan_inventory.csv"

# Standard static windows, minutes after injection (start, end). Frames whose
# midpoint falls inside the window are averaged; if fewer than two do, all
# frames are used (short series were acquired at the intended time anyway).
WINDOWS = {"PiB": (50, 70), "Florbetapir": (50, 70), "Florbetaben": (90, 110), "NAV4694": (50, 70),
           "Flortaucipir": (80, 100), "MK-6240": (90, 110), "FDG": (30, 60)}
FULL_DYNAMIC_MIN_FRAMES = 9
WINDOW_TOLERANCE = 5          # minutes either side of the window


def seconds(t) -> float:
    """DICOM time string HHMMSS.ffffff -> seconds since midnight."""
    t = str(t)
    return int(t[0:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:] or 0)


def convert(job: dict) -> dict:
    log = {"image_id": job["image_id"], "modality": job["modality"], "ok": False}
    try:
        folder = PET_ROOT / job["modality"] / job["folder"]
        slices = []
        for f in sorted(folder.rglob("*.dcm")):
            ds = pydicom.dcmread(f)
            px = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            slices.append((float(ds.FrameReferenceTime), float(ds.ActualFrameDuration) / 1000.0, seconds(ds.AcquisitionTime),
                           np.array(ds.ImagePositionPatient, float), px, ds))
        ds0 = slices[0][5]
        normal = np.cross(ds0.ImageOrientationPatient[:3], ds0.ImageOrientationPatient[3:])
        frames: dict = {}
        for ref, dur, acq, pos, px, _ in slices:
            frames.setdefault(round(ref), []).append((float(pos @ normal), dur, acq, pos, px))
        keys = sorted(frames)
        n_slices = {len(frames[k]) for k in keys}
        if len(n_slices) != 1:
            raise ValueError(f"frames have different slice counts: {sorted(n_slices)}")

        # minutes after injection of each frame's midpoint
        inj = ds0.RadiopharmaceuticalInformationSequence[0].get("RadiopharmaceuticalStartTime")
        mids = []
        for k in keys:
            dur, acq = frames[k][0][1], min(s[2] for s in frames[k])
            start = (acq - seconds(inj)) % 86400 if inj else np.nan
            mids.append((start + dur / 2) / 60.0)
        use = list(range(len(keys)))
        lo, hi = WINDOWS.get(job["tracer"], (None, None))
        inside = [i for i, m in enumerate(mids) if lo is not None and lo - WINDOW_TOLERANCE <= m <= hi + WINDOW_TOLERANCE]
        if len(inside) >= 2:
            use = inside
        elif len(keys) >= FULL_DYNAMIC_MIN_FRAMES:
            # full dynamic scan with unusable timing: take the last 20 minutes
            use = [i for i, m in enumerate(mids) if not m < max(mids) - 20] or use
        vol, wsum = None, 0.0
        for i in use:
            fr = sorted(frames[keys[i]], key=lambda s: s[0])
            stack = np.stack([s[4] for s in fr])               # (slices, rows, cols)
            w = fr[0][1]
            vol = stack * w if vol is None else vol + stack * w
            wsum += w
        vol /= wsum

        fr = sorted(frames[keys[use[0]]], key=lambda s: s[0])
        spacing_z = float(np.median(np.diff([s[0] for s in fr])))
        img = sitk.GetImageFromArray(vol)
        img.SetSpacing((float(ds0.PixelSpacing[1]), float(ds0.PixelSpacing[0]), spacing_z))
        img.SetOrigin(tuple(fr[0][3]))
        row, col = np.array(ds0.ImageOrientationPatient[:3], float), np.array(ds0.ImageOrientationPatient[3:], float)
        img.SetDirection(tuple(np.column_stack([row, col, normal]).ravel()))
        out = OUT / job["modality"] / f"{job['image_id']}.nii.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(img, str(out))
        log.update(ok=True, n_frames=len(keys), frames_used=len(use), minutes_from=round(min(mids[i] for i in use), 1),
                   minutes_to=round(max(mids[i] for i in use), 1), seconds_averaged=wsum, shape=str(vol.shape),
                   spacing=str(tuple(round(s, 2) for s in img.GetSpacing())), mean_activity=float(vol.mean()))
    except Exception as err:                                    # keep going; failures are listed in the log
        log["error"] = f"{type(err).__name__}: {err}"[:200]
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    inv = pd.read_csv(INVENTORY)
    path = OUT / "conversion_log.csv"
    done = set(pd.read_csv(path).query("ok").image_id) if path.exists() else set()   # resume after an interruption
    jobs = [{"image_id": r.image_id, "modality": r.folder_modality, "tracer": r.tracer,
             "folder": next(d for d in os.listdir(PET_ROOT / r.folder_modality)
                            if f"_I{r.image_id}_" in d and not d.endswith(".zip"))}
            for r in inv.itertuples() if r.image_id not in done]
    jobs = jobs[: args.limit] if args.limit else jobs
    print(f"[convert] {len(jobs)} scans to convert", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logs = []

    def save():
        new = pd.DataFrame(logs)
        if path.exists():
            new = pd.concat([pd.read_csv(path), new]).drop_duplicates("image_id", keep="last")
        new.to_csv(path, index=False)
        return new

    with ProcessPoolExecutor(args.workers) as pool:
        for i, log in enumerate(pool.map(convert, jobs), 1):
            logs.append(log)
            if i % 25 == 0 or i == len(jobs):
                save(); logs.clear()
                print(f"[convert] {i}/{len(jobs)} done", flush=True)
    new = pd.read_csv(path)
    print(new.groupby("modality").ok.agg(["size", "sum"]).to_string())
    if "error" in new:
        print(new.error.dropna().str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
