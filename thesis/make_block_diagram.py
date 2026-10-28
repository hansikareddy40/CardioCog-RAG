"""Block diagram of the whole CardioCog project (plain black and white).

The Model Training block shows the baseline models and the modality-encoder
fusion network from the review deck (fusion_architecture.png).

Run:  python thesis/make_block_diagram.py
      -> thesis/figures/project_block_diagram.png and .pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures" / "project_block_diagram"
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

fig = plt.figure(figsize=(18, 10.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 180)
ax.set_ylim(0, 104)
ax.axis("off")


def rect(x, y, w, h, lw=1.3):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec="black", lw=lw))


def box(x, y, w, h, text, size=12, bold=False, lw=1.3):
    rect(x, y, w, h, lw)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal", linespacing=1.4)


def arrow(x1, y1, x2, y2, lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color="black", lw=lw, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ dataset
rect(3, 54, 24, 46)
ax.text(15, 96.5, "NACC Dataset", ha="center", va="center", fontsize=12.5, fontweight="bold")
inputs = ["Demographics", "Cardiovascular", "Cognitive Tests", "Clinical Stage", "Genetics (APOE)", "PET"]
for i, name in enumerate(inputs):
    box(5.5, 92 - 4.6 - i * 6.2, 19, 4.6, name, size=10.5)

# ------------------------------------------------------------------ main flow
RY, RH, BW = 72, 10, 18
xs = [34, 58, 82]
for x, t in zip(xs, ["Data\nPreprocessing", "Feature\nExtraction", "Train / Validation /\nTest Split"]):
    box(x, RY, BW, RH, t)
arrow(27, RY + RH / 2, xs[0], RY + RH / 2)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(a + BW, RY + RH / 2, b, RY + RH / 2)

# ------------------------------------------------------------------ model training
PX, PY, PW, PH = 107, 26, 70, 74
rect(PX, PY, PW, PH, lw=1.6)
ax.text(PX + PW / 2, 96.5, "Model Training", ha="center", va="center", fontsize=13, fontweight="bold")
arrow(xs[-1] + BW, RY + RH / 2, PX, RY + RH / 2)

# baseline models
rect(110, 84, 64, 8.5)
ax.text(142, 90.2, "Baseline Models", ha="center", va="center", fontsize=11)
ax.text(142, 86.4, "Logistic Regression  |  Random Forest  |  XGBoost  |  MLP  |  FT-Transformer",
        ha="center", va="center", fontsize=9.5)

# fusion network
ax.text(142, 79.6, "Modality-Encoder Fusion Network", ha="center", va="center", fontsize=11)
groups = ["Demographics", "Genetics", "Cardiovascular", "Cognitive\nTests", "Clinical\nStage"]
gw, gg = 11.6, 1.5
gx = [110 + i * (gw + gg) for i in range(5)]
for x, g in zip(gx, groups):
    box(x, 69.5, gw, 6, g, size=9)
    box(x, 58.5, gw, 6, "Encoder", size=9.5)
    arrow(x + gw / 2, 69.5, x + gw / 2, 64.5)

AX0, AW = 120, 44
box(AX0, 47.5, AW, 6, "Multi-Head Attention (Fusion)", size=10.5)
for x in gx:
    arrow(x + gw / 2, 58.5, AX0 + AW / 2 + (x + gw / 2 - 142) * 0.55, 53.5)

box(112, 37.5, 26, 6, "Risk Head", size=10.5)
box(146, 37.5, 26, 6, "Progression Head", size=10.5)
arrow(136, 47.5, 125, 43.5)
arrow(148, 47.5, 159, 43.5)

box(112, 28.5, 26, 5.5, "PET Late Fusion", size=10.5)
arrow(125, 37.5, 125, 34)

# ------------------------------------------------------------------ PET image branch
PRY, PRH = 36, 10
box(3, PRY, 24, PRH, "PET Image\nPreprocessing")
arrow(15, 54, 15, PRY + PRH)
box(34, PRY, BW, PRH, "3D CNN")
box(58, PRY, BW, PRH, "Amyloid Status\nClassification", size=11.5)
arrow(27, PRY + PRH / 2, 34, PRY + PRH / 2)
arrow(34 + BW, PRY + PRH / 2, 58, PRY + PRH / 2)

# ------------------------------------------------------------------ output row (right to left)
OY, OH = 8, 10
box(128, OY, 28, OH, "Evaluation")
arrow(142, PY, 142, OY + OH)
row = [(82, "SHAP\nExplanation"), (58, "Risk\nPrediction"), (34, "Web\nApplication")]
for x, t in row:
    box(x, OY, BW, OH, t)
arrow(128, OY + OH / 2, 82 + BW, OY + OH / 2)
for (xa, _), (xb, _) in zip(row[:-1], row[1:]):
    arrow(xa, OY + OH / 2, xb + BW, OY + OH / 2)
box(3, OY, 24, OH, "RAG Evidence\nAssistant")
arrow(27, OY + OH / 2, 34, OY + OH / 2)

fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
print("wrote", OUT.with_suffix(".png"))
