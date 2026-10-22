"""Draw the explanatory diagrams used in the thesis (no participant data).

Usage:  python thesis/make_figures.py      (writes PNG files to thesis/figures/)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
INK, MUTED, BLUE, ORANGE, GREEN, GREY, LIGHT = "#1b2430", "#5b6675", "#2a78d6", "#eb6834", "#1baf7a", "#c9d0da", "#f3f5f8"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def box(ax, x, y, w, h, title, sub="", fc=LIGHT, ec=GREY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center", fontsize=9.4, fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.6, color=MUTED)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.2))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. Overall pipeline
fig, ax = plt.subplots(figsize=(10, 4.7)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.7); ax.axis("off")
top = [("NACC Freeze 74", "10 tables + 599\nPET acquisitions"), ("Data audit", "files, codes,\nform versions"),
       ("Cleaning and\nfeatures", "z-scores, history,\nPET timing"), ("Cohort and split", "19,470 participants;\nperson and centre split"),
       ("Models", "five classifiers;\nXGBoost main")]
for i, (t, s) in enumerate(top):
    box(ax, 0.1 + i * 2.0, 3.25, 1.75, 1.2, t, s, fc="#eaf2fd" if i else "#fff5d6", ec=BLUE if i else "#d9b23a")
    if i:
        arrow(ax, 0.1 + i * 2.0 - 0.25, 3.85, 0.1 + i * 2.0, 3.85)
mid = [("Ablation", "category\ncontributions"), ("Longitudinal", "trajectory features,\nGRU"), ("Explanation", "TreeSHAP"),
       ("PET fusion", "late fusion with\nlogistic regression"), ("Modality dropout", "robust to missing\ncategories")]
for i, (t, s) in enumerate(mid):
    box(ax, 0.1 + i * 2.0, 1.7, 1.75, 1.05, t, s)
    arrow(ax, 8.98, 3.25, 0.98 + i * 2.0, 2.75)
bottom = [("Validation", "subgroups, calibration,\nsurvival, sensitivity", "#e6f3ea", GREEN),
          ("PET image pipeline", "registration, SUVR,\n3D CNN", "#e6f3ea", GREEN),
          ("Verification", "17 independent\nchecks", "#fdecea", ORANGE),
          ("Prototype", "clinician interface,\nevidence assistant", "#fdecea", ORANGE)]
for i, (t, s, fc, ec) in enumerate(bottom):
    box(ax, 0.55 + i * 2.35, 0.15, 2.0, 1.05, t, s, fc=fc, ec=ec)
save(fig, "pipeline.png")

# 2. Partitions
fig, ax = plt.subplots(figsize=(10, 2.2)); ax.axis("off"); ax.set_xlim(0, 19470); ax.set_ylim(0, 1)
x = 0
for name, n, ev, col in [("Training", 10879, 1599, BLUE), ("Validation", 2241, 342, "#7fb0ea"),
                         ("Test", 2286, 332, ORANGE), ("Held-out centres (9)", 4064, 545, GREEN)]:
    ax.add_patch(plt.Rectangle((x, 0.42), n - 60, 0.3, color=col))
    ax.text(x + n / 2, 0.57, f"{n:,}", ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)
    ax.text(x + n / 2, 0.86, name, ha="center", va="center", fontweight="bold", fontsize=9.5, color=INK)
    ax.text(x + n / 2, 0.24, f"{ev:,} converters", ha="center", va="center", fontsize=8.4, color=MUTED)
    x += n
save(fig, "split.png")

# 3. Outcome definition
fig, ax = plt.subplots(figsize=(10, 2.9)); ax.axis("off"); ax.set_xlim(-1.6, 6.2); ax.set_ylim(0, 3.3)
ax.axvspan(0, 3, ymin=0.04, ymax=0.93, color=LIGHT)
ax.text(1.5, 3.17, "three-year window after the index visit", ha="center", fontsize=8.5, color=MUTED)
rows = [("Example A", [(0, "normal"), (1, "normal"), (2.1, "MCI"), (2.8, "dementia")], "label 1 (converter)", ORANGE),
        ("Example B", [(0, "MCI"), (1.1, "MCI"), (2.0, "MCI"), (3.2, "MCI")], "label 0 (followed beyond 3 years)", BLUE),
        ("Example C", [(0, "normal"), (1.0, "normal")], "excluded (outcome unknown)", MUTED)]
for i, (name, visits, verdict, col) in enumerate(rows):
    y = 2.55 - i * 0.95
    ax.plot([0, max(v for v, _ in visits)], [y, y], color=GREY, lw=2, zorder=1)
    for v, dx in visits:
        dem = dx == "dementia"
        ax.scatter([v], [y], s=90, color=ORANGE if dem else "white", edgecolor=ORANGE if dem else INK, lw=1.4, zorder=2)
        ax.text(v, y - 0.3, dx, ha="center", fontsize=8, color=ORANGE if dem else MUTED)
    ax.text(-1.55, y, name, va="center", fontweight="bold", fontsize=9.5, color=INK)
    ax.text(3.45, y, verdict, va="center", fontsize=9, color=col, fontweight="bold")
for yr in range(4):
    ax.text(yr, 0.0, "index visit" if yr == 0 else f"year {yr}", ha="center", fontsize=8, color=MUTED)
save(fig, "outcome.png")

# 4. Demographically adjusted z-score (MoCA example, coefficients from the fitted normative model)
fig, ax = plt.subplots(figsize=(10, 2.5))
xs = np.linspace(17, 32, 300); mu, sd = 25.75, 2.556
dens = np.exp(-0.5 * ((xs - mu) / sd) ** 2)
ax.fill_between(xs, dens, color="#eaf2fd"); ax.plot(xs, dens, color=BLUE, lw=1.5)
ax.axvline(mu, color=BLUE, lw=1.1, ls="--")
ax.text(mu + 0.15, 1.03, "expected score, age 72, female,\n16 years of education: 25.8", fontsize=8.4, color=BLUE, va="bottom")
ax.axvline(21, color=ORANGE, lw=1.8); ax.text(20.85, 0.62, "observed\nscore 21", fontsize=8.8, color=ORANGE, ha="right", fontweight="bold")
ax.annotate("", xy=(21, 0.32), xytext=(mu, 0.32), arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.1))
ax.text((21 + mu) / 2, 0.37, r"$z = (21 - 25.8)\,/\,2.56 = -1.9$", ha="center", fontsize=9.5, color=INK)
ax.set_ylim(0, 1.45); ax.set_yticks([]); ax.set_xlabel("MoCA total score")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
save(fig, "zscore.png")
print("figures written to", OUT)
