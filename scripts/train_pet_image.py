"""Phase 5c: an image model with a target the data can support.

Task     amyloid positive vs negative, from the registered amyloid PET image.
Labels   the PET core's amyloid status (about 290 labelled scans).
Why not "predict decline from the image"? Only 5 people with an image went on
to dementia. No image model can learn from 5 examples.

Three models, each evaluated by 5-fold cross-validation (every scan is tested
once by a model that never saw it):

  1. cortical SUVR threshold   one number per scan from pet_suvr.py. The
                               transparent baseline, and how PET is read in practice.
  2. regional logistic model   mean uptake in a coarse grid of blocks.
  3. 3D CNN                    a small 3D convolutional network on the whole volume.

Grad-CAM then shows which parts of the image drive the CNN, averaged over
amyloid-positive scans, and how much of that falls inside the standard
cortical region.

Output  reports/phase5_pet_cnn_results.json, reports/figures/11_gradcam_mean.png
Usage:  python scripts/train_pet_image.py
"""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import modeling as M
import nacc_utils as nu
from pet_suvr import MNI, OUT as SUVR_CSV, load_masks

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SHAPE = (80, 96, 80)            # z, y, x after cropping to the head and resizing (about 2.4 mm voxels)
EPOCHS, BATCH, SEED = 40, 8, 0


def load_volumes(df: pd.DataFrame):
    """Each image divided by its cerebellar mean (so voxels are SUVR), cropped and resized."""
    vols, masks, like = [], None, None
    for image_id in df.image_id:
        img = sitk.ReadImage(str(MNI / "Amyloid" / f"{image_id}.nii.gz"), sitk.sitkFloat32)
        if masks is None:
            masks, like = load_masks(img), img
        a = sitk.GetArrayFromImage(img)
        a = np.clip(a / a[masks["cerebellum"]].mean(), 0, 4)
        t = torch.from_numpy(a[4:-4, 6:-6, 8:-8])[None, None]
        vols.append(F.interpolate(t, size=SHAPE, mode="trilinear", align_corners=False)[0].numpy())
    ctx = torch.from_numpy(masks["cortex"][4:-4, 6:-6, 8:-8].astype(np.float32))[None, None]
    ctx = F.interpolate(ctx, size=SHAPE, mode="trilinear")[0, 0].numpy() > 0.5
    return np.stack(vols).astype(np.float32), ctx


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.ReLU(inplace=True),
                                 nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class Small3DCNN(nn.Module):
    """Four convolution stages, each halving the resolution, then global average pooling.
    About 0.3 million weights: deliberately small for a few hundred scans."""

    def __init__(self, width=12):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.stages = nn.ModuleList([Block(1, w[0]), Block(w[0], w[1]), Block(w[1], w[2]), Block(w[2], w[3])])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(w[3], 1)

    def features(self, x):
        for i, stage in enumerate(self.stages):
            x = stage(x if i == 0 else F.max_pool3d(x, 2))
        return x                                    # (batch, channels, 10, 12, 10)

    def forward(self, x):
        return self.fc(self.drop(self.features(x).mean(dim=(2, 3, 4)))).squeeze(-1)


def augment(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Left-right flip, shift by a few voxels, small intensity change."""
    if torch.rand(1, generator=g).item() < 0.5:
        x = x.flip(-1)
    s = torch.randint(-3, 4, (3,), generator=g).tolist()
    x = torch.roll(x, shifts=s, dims=(-3, -2, -1))
    return x * (1 + 0.1 * (torch.rand(1, generator=g).item() - 0.5))


def train_cnn(Xtr, ytr, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Small3DCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EPOCHS * int(np.ceil(len(ytr) / BATCH)))
    pos_weight = torch.tensor((1 - ytr.mean()) / ytr.mean(), device=DEVICE)
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).float()
    for _ in range(EPOCHS):
        model.train()
        for idx in torch.randperm(len(y), generator=g).split(BATCH):
            xb = torch.stack([augment(X[i], g) for i in idx]).to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(xb), y[idx].to(DEVICE), pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def predict_cnn(model, X):
    return torch.cat([torch.sigmoid(model(torch.from_numpy(X[i:i + 16]).to(DEVICE))) for i in range(0, len(X), 16)]).cpu().numpy()


def grad_cam(model, x: np.ndarray) -> np.ndarray:
    """Grad-CAM: weight each feature map of the last stage by how much the
    output depends on it, sum, keep the positive part, and upsample."""
    t = torch.from_numpy(x[None]).to(DEVICE)
    feats = model.features(t)
    feats.retain_grad()
    model.fc(feats.mean(dim=(2, 3, 4))).sum().backward()
    cam = F.relu((feats.grad.mean(dim=(2, 3, 4), keepdim=True) * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=SHAPE, mode="trilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def block_features(X: np.ndarray) -> np.ndarray:
    """Mean uptake in an 8 x 8 x 8-voxel grid of blocks."""
    t = torch.from_numpy(X)
    return F.avg_pool3d(t, 8).flatten(1).numpy()


def main():
    t0 = time.time()
    df = pd.read_csv(SUVR_CSV)
    df = df[(df.cerebellum_coverage > 0.95) & df.AMYLOIDSTATUS.isin([0, 1])].reset_index(drop=True)
    y = df.AMYLOIDSTATUS.to_numpy(int)
    X, ctx = load_volumes(df)
    print(f"[image] {len(df)} labelled scans, {y.sum()} positive, volume {X.shape[1:]}, device {DEVICE}", flush=True)

    strata = df.AMYLOIDSTATUS.astype(int).astype(str) + "_" + df.tracer.astype(str)
    strata = strata.where(strata.map(strata.value_counts()) >= 5, df.AMYLOIDSTATUS.astype(int).astype(str))
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strata))
    oof = {k: np.zeros(len(y)) for k in ["Cortical SUVR (one number)", "Regional logistic model", "3D CNN"]}
    cam_sum, cam_n, B = np.zeros(SHAPE), 0, block_features(X)
    for f, (tr, te) in enumerate(folds):
        oof["Cortical SUVR (one number)"][te] = df.image_suvr.to_numpy()[te]
        mu, sd = B[tr].mean(0), B[tr].std(0) + 1e-6
        lr = LogisticRegression(C=0.01, max_iter=3000).fit((B[tr] - mu) / sd, y[tr])
        oof["Regional logistic model"][te] = lr.predict_proba((B[te] - mu) / sd)[:, 1]
        model = train_cnn(X[tr], y[tr], seed=f)
        oof["3D CNN"][te] = predict_cnn(model, X[te])
        for i in te[(y[te] == 1) & (oof["3D CNN"][te] > 0.5)]:      # correctly detected positives
            cam_sum += grad_cam(model, X[i]); cam_n += 1
        print(f"[image] fold {f + 1}: CNN AUROC {roc_auc_score(y[te], oof['3D CNN'][te]):.3f} ({time.time() - t0:.0f}s)", flush=True)

    res = {"n": int(len(y)), "positive": int(y.sum()), "tracers": df.tracer.value_counts().to_dict(), "volume_shape": list(SHAPE),
           "cnn_parameters": int(sum(p.numel() for p in Small3DCNN().parameters())), "epochs": EPOCHS, "models": {}}
    for name, p in oof.items():
        b = M.bootstrap(y, p, n_boot=1000, p_ref=None if name.startswith("Cortical") else oof["Cortical SUVR (one number)"])
        res["models"][name] = {k: b[k] for k in b if k in ("n", "events", "auroc", "auroc_ci", "auprc", "delta_auroc", "delta_auroc_ci")}
    res["cnn_auroc_by_tracer"] = {t: float(roc_auc_score(y[m], oof["3D CNN"][m])) for t in df.tracer.unique()
                                  if (m := (df.tracer == t).to_numpy()).sum() >= 20 and 0 < y[m].sum() < m.sum()}
    # Grad-CAM: where does the network look, on average, for true positives?
    cam = cam_sum / max(cam_n, 1)
    top = cam >= np.quantile(cam, 0.95)
    res["grad_cam"] = {"scans_averaged": int(cam_n), "share_of_top5pct_inside_cortical_region": float((top & ctx).sum() / top.sum()),
                       "share_expected_by_chance": float(ctx.mean())}
    mean_pos, mean_neg = X[y == 1].mean(0)[0], X[y == 0].mean(0)[0]
    nu.set_style()
    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    z, yy, x = [s // 2 for s in SHAPE]
    slices = [np.s_[z + 10], np.s_[z], np.s_[:, yy], np.s_[:, :, x + 4]]
    for j, sl in enumerate(slices):
        axes[0, j].imshow(mean_neg[sl], cmap="gray", origin="lower", vmin=0, vmax=2.2)
        axes[1, j].imshow(mean_pos[sl], cmap="gray", origin="lower", vmin=0, vmax=2.2)
        axes[2, j].imshow(mean_pos[sl], cmap="gray", origin="lower", vmin=0, vmax=2.2)
        axes[2, j].imshow(np.ma.masked_less(cam[sl], 0.25 * cam.max()), cmap="autumn", origin="lower", alpha=0.6, vmin=0, vmax=cam.max())
        axes[2, j].contour(ctx[sl], colors="#2a78d6", linewidths=0.6)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0, 0].set_ylabel("Mean of amyloid-negative scans"); axes[1, 0].set_ylabel("Mean of amyloid-positive scans")
    axes[2, 0].set_ylabel("Mean Grad-CAM (blue = standard cortical region)")
    fig.suptitle("Group averages: SUVR images and where the 3D CNN looks", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(); nu.savefig(fig, "11_gradcam_mean")
    text = json.dumps(res, indent=1)
    nu.assert_no_ids(text, "PET CNN results")
    (nu.REPORTS / "phase5_pet_cnn_results.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
