"""
BERT AG-News Experiment Visualization
======================================
Reads bert_agnews.json + efficiency_detail_bert.json
Outputs 6 PNG figures to the specified output directory.

Usage:
    python plot_bert_agnews.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ─── PATHS ────────────────────────────────────────────────────────────────────

BASE = r"C:\Users\yunman\Desktop\SEM2\DSADL\text_models\part3"
MAIN_JSON = os.path.join(BASE, r"server_output_bert_agnews\results\bert_agnews.json")
EFF_JSON  = os.path.join(BASE, r"server_output_bert_agnews\results\efficiency_detail_bert\efficiency_detail_bert.json")
OUT_DIR   = os.path.join(BASE, r"output_bert_agnews")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

with open(MAIN_JSON, "r") as f:
    main = json.load(f)
with open(EFF_JSON, "r") as f:
    eff_detail = json.load(f)

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
CLASS_COLORS_MAP = {}  # will be populated dynamically

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saved {path}")

# ─── HELPERS ──────────────────────────────────────────────────────────────────

model_name = main.get("model", "bert-base-uncased")
dataset    = main.get("dataset", "ag_news")
seeds      = main.get("seeds", [])
hp_search  = main.get("hp_search", {})
best_lr    = main.get("best_lr", None)
full_data  = main.get("full_data", {})
per_seed   = main.get("full_data_per_seed", [])
efficiency = main.get("efficiency", {})

# detect class names from first seed
class_names = list(per_seed[0]["per_class"].keys()) if per_seed else []
# assign colors to classes
_palette = [C["rose"], C["teal"], C["amber"], C["violet"], C["blue"]]
for i, cn in enumerate(class_names):
    CLASS_COLORS_MAP[cn] = _palette[i % len(_palette)]

# efficiency fractions sorted
eff_fracs_sorted = sorted(efficiency.keys(), key=lambda x: float(x))

# ─── FIGURE 1: Learning Rate Search ──────────────────────────────────────────

def plot_lr_search():
    lrs = sorted(hp_search.keys(), key=lambda x: float(x))
    accs = [hp_search[lr] for lr in lrs]
    labels = [f"{float(lr):.0e}" for lr in lrs]
    best_idx = accs.index(max(accs))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [C["teal"] if i == best_idx else C["blue"] for i in range(len(lrs))]
    alphas = [1.0 if i == best_idx else 0.55 for i in range(len(lrs))]

    bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="none", zorder=3)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)

    for i, (lbl, acc) in enumerate(zip(labels, accs)):
        ax.text(i, acc + 0.001, f"{acc:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=C["teal"] if i == best_idx else C["muted"])

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Hyperparameter Search — {model_name}",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(min(accs) - 0.008, max(accs) + 0.006)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    fig.text(0.99, 0.01, f"Best LR: {best_lr}", ha="right", fontsize=9,
             color=C["teal"], fontstyle="italic")
    save(fig, "1_lr_search.png")

# ─── FIGURE 2: Seed Stability ────────────────────────────────────────────────

def plot_seed_stability():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Accuracy per seed ---
    ax = axes[0]
    seed_labels = [f"Seed {s['seed']}" for s in per_seed]
    accs = [s["accuracy"] for s in per_seed]
    mean_acc = full_data["acc_mean"]

    ax.bar(seed_labels, accs, color=SEED_COLORS[:len(per_seed)],
           width=0.45, edgecolor="none", zorder=3)
    ax.axhline(mean_acc, color=C["amber"], ls="--", lw=1.2, zorder=2,
               label=f"Mean = {mean_acc:.4f}")
    for i, v in enumerate(accs):
        ax.text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=9,
                color=SEED_COLORS[i % len(SEED_COLORS)], fontweight="bold")

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy per Seed (Full Data)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    y_min = min(accs) - 0.003
    y_max = max(accs) + 0.003
    ax.set_ylim(y_min, y_max)

    # --- Right: Per-class F1 grouped ---
    ax = axes[1]
    n_seeds = len(per_seed)
    n_classes = len(class_names)
    x = np.arange(n_seeds)
    width = 0.8 / n_classes

    for j, cn in enumerate(class_names):
        f1s = [s["per_class"][cn]["f1"] for s in per_seed]
        offset = (j - (n_classes - 1) / 2) * width
        ax.bar(x + offset, f1s, width * 0.9, label=cn,
               color=CLASS_COLORS_MAP[cn], edgecolor="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s['seed']}" for s in per_seed])
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 by Seed", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Seed Stability — {model_name} on {dataset}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "2_seed_stability.png")

# ─── FIGURE 3: Training Curves ───────────────────────────────────────────────

def plot_training_curves():
    n = len(per_seed)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, s in enumerate(per_seed):
        logs = s.get("epoch_logs", [])
        epochs = [l["epoch"] for l in logs]
        t_loss = [l["train_loss"] for l in logs]
        v_loss = [l["val_loss"] for l in logs]
        v_acc  = [l["val_acc"] for l in logs]

        # Top row: loss
        ax = axes[0, i]
        ax.plot(epochs, t_loss, "-o", color=C["blue"], lw=2, ms=5, label="Train Loss")
        ax.plot(epochs, v_loss, "-s", color=C["rose"], lw=2, ms=5, label="Val Loss")
        ax.set_title(f"Seed {s['seed']}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Loss" if i == 0 else "")
        ax.legend(fontsize=8)

        # Bottom row: val accuracy
        ax = axes[1, i]
        ax.plot(epochs, v_acc, "-D", color=C["teal"], lw=2, ms=5, label="Val Acc")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Accuracy" if i == 0 else "")
        ax.legend(fontsize=8)

    fig.suptitle(f"Training Curves — {model_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "3_training_curves.png")

# ─── FIGURE 4: Data Efficiency ───────────────────────────────────────────────

def plot_data_efficiency():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    fracs = eff_fracs_sorted
    frac_labels = [f"{float(f)*100:.0f}%" for f in fracs]
    means = [efficiency[f]["acc_mean"] for f in fracs]
    stds  = [efficiency[f]["acc_std"] for f in fracs]

    # --- Left: Mean ± Std ---
    ax = axes[0]
    x = np.arange(len(fracs))
    ax.fill_between(x, np.array(means) - np.array(stds),
                    np.array(means) + np.array(stds),
                    color=C["blue"], alpha=0.15, zorder=2)
    ax.errorbar(x, means, yerr=stds, fmt="-o", color=C["blue"],
                lw=2.5, ms=7, capsize=5, capthick=1.5, zorder=3)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.002, f"{m:.4f}", ha="center", fontsize=8,
                color=C["blue"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(frac_labels)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Mean Accuracy ± Std", fontsize=12, fontweight="bold")

    # --- Right: Per-seed lines ---
    ax = axes[1]
    # build per-seed data from eff_detail + full_data_per_seed for 100%
    detail_keys_sorted = sorted(eff_detail.keys(),
                                key=lambda k: eff_detail[k]["fraction"])

    for si, seed_val in enumerate(seeds):
        seed_accs = []
        seed_fracs = []
        for dk in detail_keys_sorted:
            block = eff_detail[dk]
            for ps in block["per_seed"]:
                if ps["seed"] == seed_val:
                    seed_fracs.append(f"{block['fraction']*100:.0f}%")
                    seed_accs.append(ps["accuracy"])
        # add 100% from full_data_per_seed
        for ps in per_seed:
            if ps["seed"] == seed_val:
                seed_fracs.append("100%")
                seed_accs.append(ps["accuracy"])

        ax.plot(range(len(seed_fracs)), seed_accs, "-o", lw=2, ms=6,
                color=SEED_COLORS[si % len(SEED_COLORS)],
                label=f"Seed {seed_val}")

    ax.set_xticks(range(len(seed_fracs)))
    ax.set_xticklabels(seed_fracs)
    ax.set_xlabel("Training Data Fraction")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Seed Accuracy", fontsize=12, fontweight="bold")
    ax.legend()

    fig.suptitle(f"Data Efficiency — {model_name} on {dataset}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "4_data_efficiency.png")

# ─── FIGURE 5: Per-Class Metrics ─────────────────────────────────────────────

def plot_per_class():
    metrics = ["f1", "precision", "recall"]
    titles  = ["F1 Score", "Precision", "Recall"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # collect per-class data across fractions
    detail_keys_sorted = sorted(eff_detail.keys(),
                                key=lambda k: eff_detail[k]["fraction"])
    frac_labels = [f"{eff_detail[k]['fraction']*100:.0f}%" for k in detail_keys_sorted]
    frac_labels.append("100%")

    # build avg_per_class for each fraction
    all_class_data = {}  # {class_name: {metric: [values]}}
    for cn in class_names:
        all_class_data[cn] = {m: [] for m in metrics}

    for dk in detail_keys_sorted:
        avg_pc = eff_detail[dk].get("avg_per_class", {})
        for cn in class_names:
            for m in metrics:
                val = avg_pc.get(cn, {}).get(m, None)
                if val is None and m == "f1":
                    val = avg_pc.get(cn, {}).get("f1", None)
                all_class_data[cn][m].append(val)

    # add 100% from full_data_per_seed average
    for cn in class_names:
        for m in metrics:
            vals = [s["per_class"][cn][m] for s in per_seed if cn in s["per_class"]]
            all_class_data[cn][m].append(np.mean(vals) if vals else None)

    x = np.arange(len(frac_labels))
    for ax, metric, title in zip(axes, metrics, titles):
        for cn in class_names:
            vals = all_class_data[cn][metric]
            ax.plot(x, vals, "-o", lw=2, ms=5, color=CLASS_COLORS_MAP[cn], label=cn)
        ax.set_xticks(x)
        ax.set_xticklabels(frac_labels, fontsize=9)
        ax.set_xlabel("Data Fraction")
        ax.set_ylabel(title)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Per-Class Metrics — {model_name} on {dataset}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "5_per_class_metrics.png")

# ─── FIGURE 6: Speed ─────────────────────────────────────────────────────────

def plot_speed():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    fracs = eff_fracs_sorted
    frac_labels = [f"{float(f)*100:.0f}%" for f in fracs]
    train_times = [efficiency[f]["train_time_mean"] for f in fracs]
    infer_times = [efficiency[f]["infer_time_mean"] for f in fracs]

    # --- Left: Training time bar ---
    ax = axes[0]
    bars = ax.bar(frac_labels, train_times, color=C["violet"], width=0.55,
                  edgecolor="none", zorder=3)
    for bar, t in zip(bars, train_times):
        ax.text(bar.get_x() + bar.get_width() / 2, t + max(train_times) * 0.02,
                f"{t:.0f}s", ha="center", fontsize=9, color=C["violet"], fontweight="bold")
    ax.set_xlabel("Data Fraction")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Training Time", fontsize=12, fontweight="bold")

    # --- Right: Inference time line ---
    ax = axes[1]
    x = np.arange(len(fracs))
    ax.plot(x, infer_times, "-D", color=C["amber"], lw=2.5, ms=7, zorder=3)
    for i, t in enumerate(infer_times):
        ax.text(i, t + 0.003, f"{t:.3f}s", ha="center", fontsize=9,
                color=C["amber"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(frac_labels)
    ax.set_xlabel("Data Fraction")
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("Inference Time (Test Set)", fontsize=12, fontweight="bold")
    y_range = max(infer_times) - min(infer_times)
    ax.set_ylim(min(infer_times) - y_range * 0.5 - 0.01,
                max(infer_times) + y_range * 0.5 + 0.01)

    fig.suptitle(f"Speed — {model_name} on {dataset}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "6_speed.png")

# ─── RUN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  BERT AG-News Visualization")
    print(f"  Model : {model_name}")
    print(f"  Dataset: {dataset}")
    print(f"  Seeds : {seeds}")
    print(f"  Classes: {class_names}")
    print(f"{'='*50}\n")

    plot_lr_search()
    plot_seed_stability()
    plot_training_curves()
    plot_data_efficiency()
    plot_per_class()
    plot_speed()

    print(f"\n  All figures saved to:\n  {OUT_DIR}\n")
