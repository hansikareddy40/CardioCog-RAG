"""Phase 5a, step 2: bring every static PET image into one standard space.

Scans come from about 20 scanner models with different image sizes, voxel
sizes and head positions. A model (or a region-of-interest measurement) needs
the same voxel to mean the same place in every brain. Registration finds, for
each scan, the rotation, shift, scaling and shear (an affine transform, 12
numbers) that best lines it up with a reference image in MNI space.

Two passes, as is usual for PET without an MRI:
  pass 1  align each scan to the MNI152 T1 template using rotation and shift
          only. PET and T1 look different, so similarity is measured with
          mutual information.
  pass 2  average the pass-1 results per tracer class to get a PET-shaped
          template, then align each scan to that with a full affine.
          Like-with-like is more robust.

The pass-2 space is "study template space": MNI orientation and position, with
brain size set by the average of these participants. An automatic affine fit
of the averaged PET image to the skull-stripped T1 template was tried and
rejected because it over-stretched the image.

Output  data/processed/pet_mni/<modality>/<image_id>.nii.gz   2 mm MNI grid
        data/processed/pet_mni/registration_log.csv
Nothing is written under data/raw.

Usage:  python scripts/pet_register.py [--pass 1|2] [--limit N] [--workers 4]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "data" / "processed" / "pet_static"
MNI = REPO / "data" / "processed" / "pet_mni"
TEMPLATE = MNI / "templates"


def mni_reference() -> tuple[Path, Path]:
    """Write nilearn's bundled MNI152 template and brain mask (2 mm) to disk once."""
    t1, mask = TEMPLATE / "mni152_t1_2mm.nii.gz", TEMPLATE / "mni152_mask_2mm.nii.gz"
    if not t1.exists():
        from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
        TEMPLATE.mkdir(parents=True, exist_ok=True)
        load_mni152_template(resolution=2).to_filename(t1)
        load_mni152_brain_mask(resolution=2).to_filename(mask)
    return t1, mask


def register(job: dict) -> dict:
    log = {"image_id": job["image_id"], "modality": job["modality"], "pass": job["pass"], "ok": False}
    try:
        fixed = sitk.ReadImage(job["fixed"], sitk.sitkFloat32)
        moving = sitk.ReadImage(str(STATIC / job["modality"] / f"{job['image_id']}.nii.gz"), sitk.sitkFloat32)
        moving = sitk.Clamp(moving, lowerBound=0)
        smooth = sitk.SmoothingRecursiveGaussian(moving, 3.0)          # registration only; output uses the unsmoothed image

        # Pass 1 compares PET (which shows scalp and skull) with a skull-stripped T1
        # template. Letting it stretch the image gave unstable results, so it is
        # limited to rotation and shift (6 numbers). Size and shape differences
        # are handled in pass 2, PET against PET, with the full affine.
        kind = sitk.Euler3DTransform() if job["pass"] == 1 else sitk.AffineTransform(3)
        init = sitk.CenteredTransformInitializer(fixed, smooth, kind, sitk.CenteredTransformInitializerFilter.MOMENTS)
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(32)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.25, seed=1)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=200,
                                                     gradientMagnitudeTolerance=1e-6)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([4, 2, 0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        reg.SetInitialTransform(init, inPlace=False)
        tx = reg.Execute(fixed, smooth)

        out = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        path = MNI / job["modality"] / f"{job['image_id']}.nii.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(out, str(path))
        m = np.array(sitk.CompositeTransform(tx).GetNthTransform(0).Downcast().GetMatrix()).reshape(3, 3)
        log.update(ok=True, metric=float(reg.GetMetricValue()), iterations=int(reg.GetOptimizerIteration()),
                   scale_det=float(np.linalg.det(m)))                  # volume scaling; far from 1 suggests a bad fit
    except Exception as err:
        log["error"] = f"{type(err).__name__}: {err}"[:200]
    return log


def build_pet_templates() -> dict:
    """Mean of pass-1 images per modality, each scaled to its own in-brain mean."""
    _, mask_path = mni_reference()
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))) > 0
    log = pd.read_csv(MNI / "registration_log.csv")
    good = log[(log["pass"] == 1) & log.ok]
    out = {}
    for modality, g in good.groupby("modality"):
        total, ref = None, None
        for image_id in g.image_id:
            img = sitk.ReadImage(str(MNI / modality / f"{image_id}.nii.gz"))
            a = sitk.GetArrayFromImage(img)
            a = a / max(a[mask].mean(), 1e-6)
            total, ref = (a if total is None else total + a), img
        mean = sitk.GetImageFromArray((total / len(g)).astype(np.float32))
        mean.CopyInformation(ref)
        out[modality] = str(TEMPLATE / f"pet_template_{modality.lower()}.nii.gz")
        sitk.WriteImage(mean, out[modality])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="which", type=int, default=1, choices=[1, 2])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t1, _ = mni_reference()
    scans = pd.read_csv(STATIC / "conversion_log.csv").query("ok")
    fixed = {m: str(t1) for m in scans.modality.unique()} if args.which == 1 else build_pet_templates()
    jobs = [{"image_id": r.image_id, "modality": r.modality, "fixed": fixed[r.modality], "pass": args.which} for r in scans.itertuples() if r.modality in fixed]
    jobs = jobs[: args.limit] if args.limit else jobs
    print(f"[register] pass {args.which}: {len(jobs)} scans", flush=True)
    logs = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, log in enumerate(pool.map(register, jobs), 1):
            logs.append(log)
            if i % 25 == 0 or i == len(jobs):
                print(f"[register] {i}/{len(jobs)}", flush=True)
    new = pd.DataFrame(logs)
    path = MNI / "registration_log.csv"
    if path.exists():
        new = pd.concat([pd.read_csv(path), new]).drop_duplicates(["image_id", "pass"], keep="last")
    new.to_csv(path, index=False)
    cur = new[new["pass"] == args.which]
    print(cur.groupby("modality").agg(n=("ok", "size"), ok=("ok", "sum"), metric=("metric", "median"),
                                      scale_low=("scale_det", "min"), scale_high=("scale_det", "max")).round(3).to_string())


if __name__ == "__main__":
    main()
