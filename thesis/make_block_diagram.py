"""Block diagram of the whole CardioCog project (plain black and white).

Run:  python thesis/make_block_diagram.py
      -> thesis/figures/project_block_diagram.png and .pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures" / "project_block_diagram"
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

fig = plt.figure(figsize=(15, 7.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 150)
ax.set_ylim(0, 72)
ax.axis("off")


def box(x, y, w, h, text, size=12, bold=False):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 color="black", lw=1.3, shrinkA=0, shrinkB=0))


# inputs
ax.add_patch(Rectangle((3, 30), 24, 39, fc="white", ec="black", lw=1.3))
ax.text(15, 66, "NACC Dataset", ha="center", va="center", fontsize=12.5, fontweight="bold")
inputs = ["Demographics", "Cardiovascular", "Cognitive Tests", "Clinical Stage", "Genetics (APOE)", "PET"]
for i, name in enumerate(inputs):
    box(5.5, 60 - 3.8 - i * 4.8, 19, 3.8, name, size=10.5)

# row 1
RY, RH, BW = 44, 11, 21
xs = [36, 64, 92, 120]
row1 = ["Data\nPreprocessing", "Feature\nExtraction", "Train / Validation /\nTest Split"]
for x, t in zip(xs, row1):
    box(x, RY, BW, RH, t)
arrow(27, RY + RH / 2, xs[0], RY + RH / 2)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + BW, RY + RH / 2, b, RY + RH / 2)

# model training, with the models inside
ax.add_patch(Rectangle((120, 24), 27, RY + RH - 24, fc="white", ec="black", lw=1.3))
ax.text(133.5, RY + RH - 3.2, "Model Training", ha="center", va="center", fontsize=12)
ax.text(133.5, 36.5, "Logistic Regression\nRandom Forest\nXGBoost\nMLP\nFT-Transformer\nFusion Network",
        ha="center", va="center", fontsize=10, linespacing=1.4)

# row 2 (right to left)
R2Y = 5
row2 = [(120, 27, "Evaluation"), (92, BW, "SHAP\nExplanation"), (64, BW, "Risk\nPrediction"), (36, BW, "Web\nApplication")]
for x, w, t in row2:
    box(x, R2Y, w, RH, t)
arrow(133.5, 24, 133.5, R2Y + RH)
for (xa, wa, _), (xb, wb, _) in zip(row2[:-1], row2[1:]):
    arrow(xa, R2Y + RH / 2, xb + wb, R2Y + RH / 2)

box(3, R2Y, 24, RH, "RAG Evidence\nAssistant")
arrow(27, R2Y + RH / 2, 36, R2Y + RH / 2)

fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
print("wrote", OUT.with_suffix(".png"))
