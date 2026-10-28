"""Block diagram of the whole CardioCog project.

Run:  python thesis/make_block_diagram.py
      -> thesis/figures/project_block_diagram.png and .pdf
All numbers are aggregates taken from the thesis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "project_block_diagram"

INK = "#1c2430"
MUTED = "#4f5b6b"
ARROW = "#5a6472"
THEMES = {
    "data": ("#d4a72c", "#fff5d9"),
    "prep": ("#2a78d4", "#eaf2fd"),
    "model": ("#7b5cd6", "#f1edfd"),
    "eval": ("#22a06b", "#e6f4ec"),
    "out": ("#e5673a", "#fdecea"),
    "grey": ("#8a94a3", "#f3f5f8"),
}
GROUPS = [  # feature-group chips: name, count, colour
    ("Demographics", 8, "#5b8def"),
    ("Cardiovascular & medical", 25, "#e0564f"),
    ("Cognitive tests", 9, "#8e6bd8"),
    ("Clinical stage", 3, "#e0953a"),
    ("Genetics (APOE)", 3, "#2fa37a"),
]

fig = plt.figure(figsize=(22, 13.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 200)
ax.set_ylim(0, 120)
ax.axis("off")


def box(x, y, w, h, theme, lw=1.6, dashed=False, radius=1.0):
    edge, fill = THEMES[theme]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fill, ec=edge, lw=lw, ls="--" if dashed else "-"))


def item(x, y, w, h, title, body, theme, dashed=False):
    box(x, y, w, h, theme, lw=1.2, dashed=dashed, radius=0.7)
    cy = y + h / 2
    if body:
        ax.text(x + w / 2, cy + h * 0.17, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, cy - h * 0.17, body, ha="center", va="center", fontsize=8.9,
                color=MUTED, linespacing=1.25)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=INK)


def panel(x, y, w, h, number, title, theme):
    box(x, y, w, h, theme, lw=2.0)
    edge, _ = THEMES[theme]
    ax.add_patch(FancyBboxPatch((x, y + h - 5.2), w, 5.2, boxstyle="round,pad=0,rounding_size=1.0",
                                fc=edge, ec=edge, lw=2.0))
    ax.add_patch(plt.Rectangle((x, y + h - 5.2), w, 2.0, fc=edge, ec=edge, lw=0))
    ax.text(x + w / 2, y + h - 2.6, f"{number}   {title}", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="white")


def stack(x, y, w, h, entries, theme, gap=1.2, pad=1.4, top=5.2):
    """Place entries (title, body[, dashed]) evenly inside a panel."""
    inner_top = y + h - top - pad
    n = len(entries)
    eh = (inner_top - (y + pad) - gap * (n - 1)) / n
    for i, e in enumerate(entries):
        ey = inner_top - (i + 1) * eh - i * gap
        item(x + pad, ey, w - 2 * pad, eh, e[0], e[1], theme, dashed=len(e) > 2 and e[2])


def arrow(x1, y1, x2, y2, rad=0.0, color=ARROW, lw=2.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ title
ax.text(100, 116.6, "CardioCog: block diagram of the whole project", ha="center", va="center",
        fontsize=21, fontweight="bold", color=INK)
ax.text(100, 112.6, "3-year prediction of dementia and cognitive decline from demographic, cardiovascular, "
        "cognitive, clinical, genetic and PET information (NACC)", ha="center", va="center",
        fontsize=12, color=MUTED)

# ------------------------------------------------------------------ row A: main flow
W, GAP, X0 = 29.9, 3.5, 2.0
XS = [X0 + i * (W + GAP) for i in range(6)]
AY, AH = 45.0, 63.5

panel(XS[0], AY, W, AH, "1", "Data sources", "data")
stack(XS[0], AY, W, AH, [
    ("NACC clinical table", "217,598 visits · 57,038 people\n46 centres · 2005–2026"),
    ("PET summary tables", "amyloid Centiloids (~5,100 people)\ntau SUVR (2,724 people)"),
    ("Raw PET images", "599 DICOM scans · 446 people\n~100 GB (amyloid, tau, FDG)"),
], "data")

panel(XS[1], AY, W, AH, "2", "Cleaning & harmonisation", "prep")
stack(XS[1], AY, W, AH, [
    ("Missing-value codes", "cleaned column by column\n114 variables mapped by hand"),
    ("Visit filter", "in-person visits\nUDS form versions 1–3"),
    ("Cognitive z-scores", "norms from healthy training people\n(age, sex, education) → 6 domains"),
    ("No future information", "fixed traits carried forward only\nPET used only if ≤ 365 days before"),
], "prep")

# panel 3: feature chips
panel(XS[2], AY, W, AH, "3", "48 features in 5 groups", "prep")
pad, gap = 1.4, 1.0
top = AY + AH - 5.2 - pad
ch = 7.2
for i, (name, n, col) in enumerate(GROUPS):
    cy = top - (i + 1) * ch - i * gap
    ax.add_patch(FancyBboxPatch((XS[2] + pad, cy), W - 2 * pad, ch, boxstyle="round,pad=0,rounding_size=0.7",
                                fc="white", ec=col, lw=1.6))
    ax.add_patch(FancyBboxPatch((XS[2] + pad, cy), 6.0, ch, boxstyle="round,pad=0,rounding_size=0.7",
                                fc=col, ec=col, lw=1.6))
    ax.text(XS[2] + pad + 3.0, cy + ch / 2, str(n), ha="center", va="center", fontsize=13,
            fontweight="bold", color="white")
    ax.text(XS[2] + pad + 7.6, cy + ch / 2, name, ha="left", va="center", fontsize=10.5,
            fontweight="bold", color=INK)
cy = top - 6 * ch - 5 * gap
ax.add_patch(FancyBboxPatch((XS[2] + pad, cy), W - 2 * pad, ch, boxstyle="round,pad=0,rounding_size=0.7",
                            fc="white", ec="#8a94a3", lw=1.4, ls="--"))
ax.text(XS[2] + pad + 3.0, cy + ch / 2, "+4", ha="center", va="center", fontsize=12,
        fontweight="bold", color=MUTED)
ax.text(XS[2] + pad + 7.6, cy + ch / 2 + 1.3, "PET (Tier 2 only)", ha="left", va="center",
        fontsize=10.5, fontweight="bold", color=INK)
ax.text(XS[2] + pad + 7.6, cy + ch / 2 - 1.6, "Centiloids · amyloid status · 2 tau", ha="left",
        va="center", fontsize=8.6, color=MUTED)
ax.text(XS[2] + W / 2, AY + 3.0, "8 + 25 + 9 + 3 + 3 = 48 model inputs", ha="center", va="center",
        fontsize=9.5, color=MUTED, style="italic")

panel(XS[3], AY, W, AH, "4", "Outcomes, cohort & split", "prep")
stack(XS[3], AY, W, AH, [
    ("Dementia within 3 years", "main outcome · 19,470 people\n2,818 events (14.5%)"),
    ("CDR-SB rise ≥ 1 point", "decline outcome · 20,355 people\n5,035 events"),
    ("3-year change in CDR-SB", "regression target\n(visit within ±6 months of 3 years)"),
    ("Participant-level split", "train 70 · validation 15 · test 15\n+ 9 of 46 centres held out (seed 2026)"),
], "prep")

panel(XS[4], AY, W, AH, "5", "Models", "model")
stack(XS[4], AY, W, AH, [
    ("Tier 1 classifiers (48 features)", "Logistic regression · Random forest\nXGBoost ★ main · MLP · FT-Transformer"),
    ("Modality-encoder fusion network", "5 encoders → 32-d tokens → attention\n→ risk head + CDR-SB change head"),
    ("Modality dropout", "each group hidden with p = 0.35\n→ robust when data are missing"),
    ("Longitudinal models", "landmark visit + trajectory features\nGRU over up to 6 visits"),
    ("Tier 2: PET late fusion", "regularised logistic regression on\nclinical risk + PET measures"),
], "model")

panel(XS[5], AY, W, AH, "6", "Explain & evaluate", "eval")
stack(XS[5], AY, W, AH, [
    ("TreeSHAP explanations", "contribution of each feature\nand each group to every estimate"),
    ("Ablation", "each group alone, added in order,\nand removed from the full model"),
    ("Metrics", "AUROC · AUPRC · Brier · calibration\nnet benefit · 1,000-bootstrap 95% CI"),
    ("Robustness", "subgroups · Cox survival model\nnew seeds · sensitivity analyses"),
], "eval")

ymid = AY + AH / 2
for i in range(5):
    arrow(XS[i] + W + 0.2, ymid, XS[i + 1] - 0.2, ymid, lw=2.6)

# ------------------------------------------------------------------ row B
BY, BH = 21.0, 19.5

# PET image pipeline (spans columns 1-3)
px, pw = XS[0], XS[2] + W - XS[0]
panel(px, BY, pw, BH, "7", "PET image pipeline (separate experiment)", "eval")
steps = [
    ("DICOM → static", "frames in tracer\nwindow averaged"),
    ("Registration", "rigid + affine to MNI152\nmutual information"),
    ("Amyloid SUVR", "cortex ÷ cerebellum\nr = 0.98 vs PET core"),
    ("Amyloid status", "SUVR 0.982 · CNN 0.960\n(292 scans, 5-fold CV)"),
    ("Grad-CAM", "maps diffuse →\nnot used as evidence"),
]
sw, sg = (pw - 2 * 1.4 - 4 * 2.6) / 5, 2.6
for i, (t, b) in enumerate(steps):
    sx = px + 1.4 + i * (sw + sg)
    item(sx, BY + 1.4, sw, BH - 5.2 - 2.8, t, b, "eval")
    if i < 4:
        arrow(sx + sw + 0.15, BY + 1.4 + (BH - 8.0) / 2, sx + sw + sg - 0.15, BY + 1.4 + (BH - 8.0) / 2, lw=1.8)

# key results (column 4)
panel(XS[3], BY, W, BH, "8", "Key results", "grey")
res = [
    ("Dementia AUROC 0.94", "0.95 at held-out centres · 0.84 within MCI"),
    ("Decline AUROC 0.86", "0.89 at held-out centres · all 5 models alike"),
    ("Cardio adds ≈ 0.001 AUROC", "earlier visits ≤ 0.005 · PET no clear gain"),
]
for i, (a, b) in enumerate(res):
    yy = BY + BH - 8.0 - i * 4.6
    ax.text(XS[3] + W / 2, yy + 0.95, a, ha="center", va="center", fontsize=10.2, fontweight="bold", color=INK)
    ax.text(XS[3] + W / 2, yy - 1.15, b, ha="center", va="center", fontsize=8.6, color=MUTED)

# prototype and evidence assistant (columns 5-6)
panel(XS[4], BY, W, BH, "9", "Clinician prototype", "out")
item(XS[4] + 1.4, BY + 1.4, W - 2.8, BH - 8.0,
     "React + FastAPI  (Streamlit fallback)",
     "risk % vs similar people · SHAP reasons\nplausibility checks · plain-language report", "out")
panel(XS[5], BY, W, BH, "10", "Evidence assistant (RAG)", "out")
item(XS[5] + 1.4, BY + 1.4, W - 2.8, BH - 8.0,
     "11 papers → 447 passages",
     "MiniLM embeddings + BM25 → rank fusion\n→ Qwen2.5-1.5B with quoted citations", "out")

# connecting arrows into row B
arrow(XS[0] + W / 2, AY - 0.2, XS[0] + W / 2, BY + BH + 0.2, lw=2.2)                        # raw PET images
arrow(XS[5] + W / 2, AY - 0.2, XS[4] + W / 2, BY + BH + 0.2, rad=0.0, lw=2.2)                 # trained model → app
arrow(XS[4] + W / 2 + 4, AY - 0.2, XS[3] + W / 2, BY + BH + 0.2, lw=1.8, ls="--")             # models → results
arrow(XS[5] - 0.2, BY + BH / 2 - 2.2, XS[4] + W + 0.2, BY + BH / 2 - 2.2, style="<|-|>", lw=2.0)
ax.text(XS[5] - GAP / 2, BY + BH / 2 + 0.6, "evidence\nquotes", ha="center", va="bottom", fontsize=7.6,
        color=MUTED)

# ------------------------------------------------------------------ row C: verification
CY, CH = 2.5, 14.0
panel(X0, CY, XS[5] + W - X0, CH, "11", "Independent verification (scripts/verify_results.py): 17 of 17 checks pass", "grey")
checks = [
    "raw files unchanged",
    "no person or centre\nin two partitions",
    "features unchanged when\nlater visits are deleted",
    "labels recomputed:\n0 disagreements",
    "shuffled labels →\nAUROC 0.508",
    "reported numbers =\nsaved predictions",
    "2 leaks found & fixed\n(results moved in 3rd decimal)",
]
cw = (XS[5] + W - X0 - 2.8 - 6 * 1.6) / 7
for i, c in enumerate(checks):
    cx = X0 + 1.4 + i * (cw + 1.6)
    item(cx, CY + 1.3, cw, CH - 5.2 - 2.4, c, "", "grey")
for t in ax.texts[-7:]:
    t.set_fontsize(9.2)
    t.set_fontweight("normal")
    t.set_color(INK)

ax.text(199, 0.8, "Research prototype · not for clinical use · all numbers are aggregates", ha="right",
        va="bottom", fontsize=8.5, color=MUTED, style="italic")

fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
print("wrote", OUT.with_suffix(".png"))
