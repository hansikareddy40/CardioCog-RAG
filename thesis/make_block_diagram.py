"""Block diagram of the whole CardioCog project (plain black and white).

Run:  python thesis/make_block_diagram.py
      -> thesis/figures/project_block_diagram.png and .pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures" / "project_block_diagram"
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

fig = plt.figure(figsize=(15, 6.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 150)
ax.set_ylim(10, 72)
ax.axis("off")


def box(x, y, w, h, text, size=12):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, linespacing=1.4)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color="black", lw=1.3, shrinkA=0, shrinkB=0))


MY, MH = 44, 12          # main row
mid = MY + MH / 2

box(2, MY, 18, MH, "NACC\nDataset")
box(26, MY, 18, MH, "Data\nPreprocessing")
arrow(20, mid, 26, mid)

groups = ["Demographics", "Cardiovascular", "Cognitive Tests", "Clinical Stage", "Genetics"]
gh, step = 5, 6.5
gys = [mid + 2 * step - gh / 2 - i * step for i in range(5)]
for gy, g in zip(gys, groups):
    box(52, gy, 20, gh, g, size=10.5)
    arrow(44, mid, 52, gy + gh / 2)
    arrow(72, gy + gh / 2, 80, mid + (gy + gh / 2 - mid) * 0.32)

box(80, MY, 16, MH, "Encoders")
box(102, MY, 18, MH, "Attention\nFusion")
box(126, MY, 20, MH, "Risk\nPrediction")
arrow(96, mid, 102, mid)
arrow(120, mid, 126, mid)

# bottom row
BY, BH = 16, 10
bmid = BY + BH / 2
box(2, BY, 18, BH, "PET Scans")
box(26, BY, 18, BH, "3D CNN")
box(50, BY, 18, BH, "Amyloid\nStatus")
arrow(11, MY, 11, BY + BH)
arrow(20, bmid, 26, bmid)
arrow(44, bmid, 50, bmid)

box(78, BY, 18, BH, "RAG Evidence\nAssistant", size=11)
box(102, BY, 18, BH, "Web\nApplication")
box(126, BY, 20, BH, "SHAP\nExplanation")
arrow(136, MY, 136, BY + BH)
arrow(126, bmid, 120, bmid)
arrow(96, bmid, 102, bmid)

fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
print("wrote", OUT.with_suffix(".png"))
