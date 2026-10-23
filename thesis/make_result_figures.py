"""Draw the thesis figures for the decline outcomes and the fusion network.

Every value is read from the aggregate result files in reports/, so the figures
cannot drift from the numbers in the text. Output: thesis/figures/*.png
Usage:  python thesis/make_result_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
BLUE, ORANGE, GREY = "#2b78d6", "#e8693b", "#9a9a9a"
GROUPS = ["All non-demented", "Cognitively normal", "MCI"]
LABELS = ["All not demented", "Cognitively normal", "MCI"]
SHORT = ["All", "Normal", "MCI"]
SETS = [("test", "Test set"), ("external", "Held-out centres")]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load(name):
    return json.loads((REPO / "reports" / f"{name}.json").read_text())


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def save_kept(fig, name):
    """Save without tight_layout, which would undo the space kept for a figure legend."""
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def outcomes_by_group():
    """AUROC with 95% intervals: dementia (XGBoost) against CDR-SB decline (XGBoost)."""
    dem = load("phase3_by_baseline_group")["trained_on_all"]["Full model"]
    dec = load("phase3b_decline_results")["classification"]["xgboost"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True, sharey=True)
    for ax, (s, title) in zip(axes, SETS):
        y = np.arange(len(GROUPS))
        for off, src, col, lab in [(-0.15, dem, BLUE, "Dementia within 3 years"), (0.15, dec, ORANGE, "CDR-SB rise of 1 or more")]:
            a = np.array([src[s][g]["auroc"] for g in GROUPS])
            lo = np.array([src[s][g]["auroc_ci"][0] for g in GROUPS])
            hi = np.array([src[s][g]["auroc_ci"][1] for g in GROUPS])
            ax.errorbar(a, y + off, xerr=[a - lo, hi - a], fmt="o", color=col, capsize=3, label=lab)
            for v, yy in zip(a, y + off):
                ax.text(v, yy - 0.09, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=col)
        ax.set_yticks(y, LABELS)
        ax.set_title(title)
        ax.set_xlabel("AUROC (95% interval)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.28)
    save_kept(fig, "decline_vs_dementia")


def progression_error():
    """Mean absolute error of the predicted 3-year change in CDR-SB against predicting no change."""
    r = load("phase3b_decline_results")["regression"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for ax, (s, title) in zip(axes, SETS):
        x = np.arange(len(GROUPS))
        base = [r[s][g]["mae_predict_no_change"] for g in GROUPS]
        model = [r[s][g]["mae_model"] for g in GROUPS]
        ax.bar(x - 0.2, base, 0.4, color=GREY, label="Predict no change")
        ax.bar(x + 0.2, model, 0.4, color=BLUE, label="XGBoost regression")
        for xi, b, m in zip(x, base, model):
            ax.text(xi - 0.2, b + 0.03, f"{b:.3f}", ha="center", fontsize=8)
            ax.text(xi + 0.2, m + 0.03, f"{m:.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, SHORT)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean absolute error (CDR-SB points)")
    axes[0].legend(fontsize=8, frameon=False)
    save(fig, "progression_error")


def attention_shares():
    """Mean attention share of each modality in the fusion network."""
    a = load("phase6b_fusion_network_results")["attention_share"]
    mods = list(a["test"]["All non-demented"])
    short = {"Genetics / APOE": "Genetics", "Cardiovascular / medical": "Cardiovascular",
             "Clinical stage (CDR-SB, diagnosis)": "Clinical stage"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for ax, (s, title) in zip(axes, SETS):
        x = np.arange(len(mods))
        for k, (g, lab) in enumerate(zip(GROUPS, LABELS)):
            vals = [100 * a[s][g][m] for m in mods]
            ax.bar(x + (k - 1) * 0.27, vals, 0.27, label=lab, color=[BLUE, ORANGE, "#3aa37a"][k])
        ax.axhline(20, color=GREY, ls="--", lw=1)
        ax.set_xticks(x, [short.get(m, m) for m in mods], rotation=15)
        ax.set_ylim(0, 25)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Attention share (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.3)
    save_kept(fig, "fusion_attention")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    outcomes_by_group()
    progression_error()
    attention_shares()
