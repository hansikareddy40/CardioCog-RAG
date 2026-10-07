"""Look at one reconstructed 3D PET image on your own screen.

Shows the three standard views (from above, from the front, from the side) of
a registered scan, with a slider-free keyboard control:
    up / down arrows   move through the slices
    q                  close

The image is participant data: it is shown on screen only and nothing is saved.

Usage:
    python scripts/view_pet.py                 # first amyloid scan
    python scripts/view_pet.py Tau 5           # 6th tau scan (counting from 0)
    python scripts/view_pet.py Amyloid 0 raw   # before registration (scanner space)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    modality = sys.argv[1] if len(sys.argv) > 1 else "Amyloid"
    number = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    folder = PROCESSED / ("pet_static" if "raw" in sys.argv[3:] else "pet_mni") / modality
    files = sorted(folder.glob("*.nii.gz"))
    if not files:
        sys.exit(f"No images in {folder}. Run scripts/pet_convert.py and scripts/pet_register.py first.")
    vol = sitk.GetArrayFromImage(sitk.ReadImage(str(files[number % len(files)])))      # (z, y, x)
    vmax = np.percentile(vol, 99.7)
    pos = [s // 2 for s in vol.shape]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.suptitle(f"{modality} scan {number % len(files) + 1} of {len(files)}   |   volume {vol.shape[2]} x {vol.shape[1]} x {vol.shape[0]} voxels"
                 "   |   up/down arrows move through slices")
    titles = ["From above (axial)", "From the front (coronal)", "From the side (sagittal)"]

    def draw():
        views = [vol[pos[0]], vol[:, pos[1]], vol[:, :, pos[2]]]
        for ax, view, title, p, n in zip(axes, views, titles, pos, vol.shape):
            ax.clear()
            ax.imshow(view, cmap="inferno", origin="lower", vmin=0, vmax=vmax, aspect="equal")
            ax.set_title(f"{title}: slice {p + 1} of {n}")
            ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        step = {"up": 1, "down": -1}.get(event.key)
        if event.key == "q":
            plt.close(fig)
        elif step:
            for i in range(3):
                pos[i] = int(np.clip(pos[i] + step * max(1, vol.shape[i] // 40), 0, vol.shape[i] - 1))
            draw()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw()
    plt.show()


if __name__ == "__main__":
    main()
