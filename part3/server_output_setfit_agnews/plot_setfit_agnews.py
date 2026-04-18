"""
SetFit AG-News Few-Shot Visualization
=======================================
Reads setfit_agnews.json (1%, 5%, 10% only)
Outputs 4 PNG figures to the specified output directory.

Usage:
    python plot_setfit_agnews.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─── PATHS ────────────────────────────────────────────────────────────────────

BASE = r"C:\Users\yunman\Desktop\SEM2\DSADL\text_models\part3"
EFF_JSON = os.path.join(BASE, r"server_output_setfit_agnews\results\setfit_agnews.json")
OUT_DIR  = os.path.join(BASE, r"output_setfit_agnews")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

with open(EFF_JSON, "r") as f:
    data = json.load(f)

# ─── STYLE ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#181b24",
    "axes.edgecolor":   "#363c4e",
    "axes.labelcolor":  "#c8cdd8",
    "text.color":       "#e0e4ef",
    "xtick.color":      "#8890a4",
    "ytick.color":      "#8890a4",
    "grid.color":       "#262a36",
    "grid.alpha":       0.6,
    "legend.facecolor": "#1e2230",
    "legend.edgecolor": "#363c4e",
    "legend.fontsize":  9,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.grid":        True,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"#0f1117",
})

C = {
    "blue":   "#6c9bff",
    "rose":   "#ff6c8a",
    "teal":   "#5eead4",
    "amber":  "#fbbf24",
    "violet": "#a78bfa",
    "muted":  "#8890a4",
}

SEED_COLORS = [C["blue"], C["rose"], C["teal"]]

MODEL_LABEL = "SetFit (MiniLM-L6)"
DATASET     = "AG News"
SEEDS       = [123, 456, 789]

# Sort fractions
frac_keys_sorted = sorted(data.keys(), key=lambda k: data[k]["fraction"])
fracs      = [data[k]["fraction"] for k in frac_keys_sorted]
frac_labels = [f"{f*100:.0f}%" for f in fracs]

# Detect class names
class_names = list(data[frac_keys_sorted[0]]["avg_per_class"].keys())
_palette = [C["rose"], C["teal"], C["amber"], C["violet"], C["blue"]]
CLASS_COLORS = {cn: _palette[i % len(_palette)] for i, cn in enumerate(class_names)}

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved {path}")

# ─── FIGURE 1: Data Efficiency — Accuracy ± Std + Per-Seed ───────────────────

def plot_data_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    means = [data[k]["acc_mean"] for k in frac_keys_sorted]
    stds  = [data[k]["acc_std"] for k in frac_keys_sorted]

    # Left: Mean ± Std
    ax = axes[0]
    x = np.arange(len(fracs))
    ax.fill_between(x, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=C["blue"], alpha=0.15, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="-o", color=C["blue"],
                lw=2.5, ms=7, capsize=5, capthick=1.5, zorder=3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.003, f"{m:.4f}", ha="center", fontsize=9,
                color=C["blue"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(frac_labels)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean Accuracy ± Std (3 seeds)", fontsize=12, fontweight="bold")

    # Right: Per-seed lines
    ax = axes[1]
    for si, seed_val in enumerate(SEEDS):
        seed_accs = []
        for k in frac_keys_sorted:
            for ps in data[k]["per_seed"]:
                if ps["seed"] == seed_val:
                    seed_accs.append(ps["accuracy"])
        ax.plot(range(len(seed_accs)), seed_accs, "-o", lw=2, ms=6,
                color=SEED_COLORS[si], label=f"Seed {seed_val}")

    ax.set_xticks(range(len(frac_labels)))
    ax.set_xticklabels(frac_labels)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Seed Accuracy", fontsize=12, fontweight="bold")
    ax.legend()

    fig.suptitle(f"Data Efficiency — {MODEL_LABEL} on {DATASET}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "1_data_efficiency.png")

# ─── FIGURE 2: Seed Stability — grouped bars at each fraction ────────────────

def plot_seed_stability():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    n_fracs = len(frac_keys_sorted)
    n_seeds = len(SEEDS)
    x = np.arange(n_fracs)
    width = 0.22

    for si, seed_val in enumerate(SEEDS):
        accs = []
        for k in frac_keys_sorted:
            for ps in data[k]["per_seed"]:
                if ps["seed"] == seed_val:
                    accs.append(ps["accuracy"])
        offset = (si - (n_seeds - 1) / 2) * width
        bars = ax.bar(x + offset, accs, width * 0.9, label=f"Seed {seed_val}",
                      color=SEED_COLORS[si], edgecolor="none", zorder=3)
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, a + 0.001,
                    f"{a:.4f}", ha="center", fontsize=8,
                    color=SEED_COLORS[si], fontweight="bold", rotation=0)

    # Mean reference lines
    for i, k in enumerate(frac_keys_sorted):
        m = data[k]["acc_mean"]
        ax.hlines(m, i - 0.35, i + 0.35, colors=C["amber"],
                  linestyles="--", lw=1.2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(frac_labels)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Seed Stability — {MODEL_LABEL} on {DATASET}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right")

    all_accs = [ps["accuracy"] for k in frac_keys_sorted for ps in data[k]["per_seed"]]
    ax.set_ylim(min(all_accs) - 0.008, max(all_accs) + 0.008)

    save(fig, "2_seed_stability.png")

# ─── FIGURE 3: Per-Class Metrics — F1, Precision, Recall ─────────────────────

def plot_per_class():
    metrics = ["f1", "precision", "recall"]
    titles  = ["F1 Score", "Precision", "Recall"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(frac_labels))

    for ax, metric, title in zip(axes, metrics, titles):
        for cn in class_names:
            vals = [data[k]["avg_per_class"][cn][metric] for k in frac_keys_sorted]
            ax.plot(x, vals, "-o", lw=2, ms=5, color=CLASS_COLORS[cn], label=cn)

        ax.set_xticks(x)
        ax.set_xticklabels(frac_labels, fontsize=9)
        ax.set_xlabel("Data Fraction")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Per-Class Metrics — {MODEL_LABEL} on {DATASET}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "3_per_class_metrics.png")

# ─── FIGURE 4: Speed — Training & Inference ─────────────────────────────────

def plot_speed():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    train_times = [data[k]["train_time_mean"] for k in frac_keys_sorted]
    infer_times = [data[k]["infer_time_mean"] for k in frac_keys_sorted]

    # Left: Training time
    ax = axes[0]
    bars = ax.bar(frac_labels, train_times, color=C["violet"], width=0.5,
                  edgecolor="none", zorder=3)
    for bar, t in zip(bars, train_times):
        ax.text(bar.get_x() + bar.get_width() / 2, t + max(train_times) * 0.02,
                f"{t:.0f}s", ha="center", fontsize=10, color=C["violet"], fontweight="bold")
    ax.set_xlabel("Data Fraction")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Training Time", fontsize=12, fontweight="bold")

    # Right: Inference time
    ax = axes[1]
    x = np.arange(len(frac_labels))
    ax.plot(x, infer_times, "-D", color=C["amber"], lw=2.5, ms=7, zorder=3)
    for i, t in enumerate(infer_times):
        ax.text(i, t + 0.008, f"{t:.3f}s", ha="center", fontsize=10,
                color=C["amber"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(frac_labels)
    ax.set_xlabel("Data Fraction")
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("Inference Time (Test Set)", fontsize=12, fontweight="bold")
    y_range = max(infer_times) - min(infer_times)
    ax.set_ylim(min(infer_times) - max(y_range * 0.5, 0.05),
                max(infer_times) + max(y_range * 0.5, 0.05))

    fig.suptitle(f"Speed — {MODEL_LABEL} on {DATASET}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "4_speed.png")

# ─── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  SetFit AG-News Visualization")
    print(f"  Model  : {MODEL_LABEL}")
    print(f"  Dataset: {DATASET}")
    print(f"  Fracs  : {frac_labels}")
    print(f"  Classes: {class_names}")
    print(f"{'='*50}\n")

    plot_data_efficiency()
    plot_seed_stability()
    plot_per_class()
    plot_speed()

    print(f"\n  All figures saved to:\n  {OUT_DIR}\n")
