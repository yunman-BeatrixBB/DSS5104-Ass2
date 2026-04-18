"""
setfit_imdb.py — SetFit few-shot on IMDB (1%, 5%, 10% only)
DSS5104 Text Classification Assignment (Tier 3 supplement)

SetFit is designed for low-data regimes. Per assignment instructions,
evaluate only at 1%, 5%, and at most 10% of training data.

Matches the same data splits, seeds, metrics, and output format as
bert_imdb_v3_a100.py for direct comparison.

Usage:
  pip install setfit
  python setfit_imdb.py

Optional:
  python setfit_imdb.py --fractions 0.01 0.05 0.10
  python setfit_imdb.py --batch_size 16
"""

import os, sys, json, time, random, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════

# SetFit backbone — sentence-transformers model
ST_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
MODEL_LABEL   = "SetFit (MiniLM-L6)"
SHORT_NAME    = "setfit"

DATASET_NAME  = "imdb"
NUM_CLASSES   = 2
LABEL_NAMES   = ["Negative", "Positive"]

# SetFit training hyperparameters
NUM_ITERATIONS = 20       # number of text pairs per class to generate
NUM_EPOCHS_BODY = 1       # epochs for contrastive body fine-tuning
NUM_EPOCHS_HEAD = 16      # epochs for classification head
BATCH_SIZE     = 16       # contrastive pair batch size

SEEDS          = [123, 456, 789]
DATA_FRACTIONS = [0.01, 0.05, 0.10]   # SetFit: small data only
VAL_RATIO      = 0.1

# Paths
if os.path.exists("/content/drive"):
    BASE_DIR = "/content/drive/MyDrive/part3"
else:
    BASE_DIR = "."

DATA_DIR    = os.path.join(BASE_DIR, "data/clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# ═══════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name} ({mem:.1f} GB)", flush=True)
    else:
        device = "cpu"
    print(f"[Device] {device}", flush=True)
    return device


# ═══════════════════════════════════════════════
# DATA LOADING — same splits as BERT scripts
# ═══════════════════════════════════════════════

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "imdb_train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "imdb_test.csv"))

    train_df["text"] = train_df["review"].fillna("")
    test_df["text"]  = test_df["review"].fillna("")

    # Same split as BERT: random_state=42, test_size=0.1
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(), train_df["label"].tolist(),
        test_size=VAL_RATIO, stratify=train_df["label"].tolist(), random_state=42,
    )

    test_texts  = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"[Data] train={len(train_texts):,}  val={len(val_texts):,}  "
          f"test={len(test_texts):,}", flush=True)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def stratified_subsample(texts, labels, fraction, seed=42):
    if fraction >= 1.0:
        return texts, labels
    n = max(int(len(texts) * fraction), len(set(labels)))
    idx, _ = train_test_split(
        range(len(texts)), train_size=n, stratify=labels, random_state=seed,
    )
    return [texts[i] for i in idx], [labels[i] for i in idx]


# ═══════════════════════════════════════════════
# SETFIT TRAIN & EVALUATE
# ═══════════════════════════════════════════════

def train_and_evaluate(
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, seed, device, batch_size,
):
    """Train SetFit and evaluate on test set. Returns dict with metrics."""
    set_seed(seed)

    # Prepare HuggingFace datasets
    train_ds = HFDataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds   = HFDataset.from_dict({"text": val_texts,   "label": val_labels})
    test_ds  = HFDataset.from_dict({"text": test_texts,  "label": test_labels})

    # Initialize model
    model = SetFitModel.from_pretrained(
        ST_MODEL_NAME,
        labels=LABEL_NAMES,
    )

    # Use the SetFit Trainer with TrainingArguments
    args = TrainingArguments(
        batch_size=batch_size,
        num_iterations=NUM_ITERATIONS,
        num_epochs=NUM_EPOCHS_BODY,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    # Train (contrastive + head)
    t_start = time.time()
    trainer.train()
    train_time = time.time() - t_start

    # Inference on test set
    t_infer_start = time.time()
    y_pred = model.predict(test_texts)
    infer_time = time.time() - t_infer_start

    # If predictions are tensors, convert to list
    if hasattr(y_pred, "tolist"):
        y_pred = y_pred.tolist()

    # Fix: convert string labels to int if needed
    if y_pred and isinstance(y_pred[0], str):
        y_pred = [LABEL_NAMES.index(p) for p in y_pred]

    y_true = test_labels

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)

    prec, rec, f1_per, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
    )
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[name] = {
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]), 4),
            "f1":        round(float(f1_per[i]), 4),
            "support":   int(sup[i]),
        }

    return {
        "seed":       seed,
        "accuracy":   round(acc, 4),
        "macro_f1":   round(f1, 4),
        "per_class":  per_class,
        "train_time": round(train_time, 1),
        "infer_time": round(infer_time, 3),
        "y_true":     y_true,
        "y_pred":     y_pred,
    }


# ═══════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════

def plot_confusion(y_true, y_pred, frac, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"SetFit IMDB — {frac*100:.0f}% Data — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_{frac*100:.0f}pct.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [Plot] {path}", flush=True)


def plot_classf1(per_class, frac, save_dir):
    names = list(per_class.keys())
    f1s   = [per_class[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(names, f1s, color=colors, edgecolor="white")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{f1:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"SetFit IMDB — {frac*100:.0f}% Data — Per-Class F1")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(save_dir, f"classf1_{frac*100:.0f}pct.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [Plot] {path}", flush=True)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", type=float, nargs="+", default=None,
                        help="Data fractions to evaluate (default: 0.01, 0.05, 0.10)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Contrastive pair batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    fractions = args.fractions if args.fractions else DATA_FRACTIONS
    batch_size = args.batch_size

    save_dir = os.path.join(RESULTS_DIR, f"{SHORT_NAME}_{DATASET_NAME}_detail")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"  SetFit Few-Shot — IMDB", flush=True)
    print(f"  Backbone:   {ST_MODEL_NAME}", flush=True)
    print(f"  Fractions:  {fractions}", flush=True)
    print(f"  Seeds:      {SEEDS}", flush=True)
    print(f"  Iterations: {NUM_ITERATIONS}", flush=True)
    print(f"  Batch size: {batch_size}", flush=True)
    print(f"  Output:     {save_dir}", flush=True)
    print(f"{'='*60}", flush=True)

    device = get_device()
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data()

    # Load existing progress (resume support)
    detail_json_path = os.path.join(save_dir, f"{SHORT_NAME}_{DATASET_NAME}.json")
    if os.path.exists(detail_json_path):
        with open(detail_json_path, "r") as f:
            all_detail = json.load(f)
        print(f"[Resume] Found {len(all_detail)} completed fractions", flush=True)
    else:
        all_detail = {}

    for frac in fractions:
        frac_key = f"{frac*100:.0f}pct"

        if frac_key in all_detail:
            s = all_detail[frac_key]
            print(f"\n  [SKIP] {frac*100:.0f}% → f1={s['f1_mean']:.4f}±{s['f1_std']:.4f} "
                  f"(cached)", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  Fraction: {frac*100:.0f}%", flush=True)
        print(f"{'='*60}", flush=True)

        frac_seed_results = []
        first_y_true, first_y_pred = None, None

        for seed in SEEDS:
            sub_texts, sub_labels = stratified_subsample(
                train_texts, train_labels, fraction=frac, seed=seed,
            )
            print(f"\n  Seed={seed}  n_train={len(sub_texts):,}", flush=True)

            res = train_and_evaluate(
                sub_texts, sub_labels, val_texts, val_labels,
                test_texts, test_labels, seed, device, batch_size,
            )

            frac_seed_results.append({
                k: v for k, v in res.items() if k not in ("y_true", "y_pred")
            })
            print(f"    acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}  "
                  f"train={res['train_time']:.1f}s  infer={res['infer_time']:.3f}s", flush=True)

            if first_y_true is None:
                first_y_true = res["y_true"]
                first_y_pred = res["y_pred"]

        # Summarize across seeds
        accs = [r["accuracy"] for r in frac_seed_results]
        f1s  = [r["macro_f1"] for r in frac_seed_results]

        avg_per_class = {}
        for name in LABEL_NAMES:
            avg_per_class[name] = {
                "precision": round(np.mean([r["per_class"][name]["precision"]
                                            for r in frac_seed_results]), 4),
                "recall":    round(np.mean([r["per_class"][name]["recall"]
                                            for r in frac_seed_results]), 4),
                "f1":        round(np.mean([r["per_class"][name]["f1"]
                                            for r in frac_seed_results]), 4),
            }

        n_train = len(stratified_subsample(train_texts, train_labels, frac, seed=42)[0])

        all_detail[frac_key] = {
            "fraction":      frac,
            "n_train":       n_train,
            "acc_mean":      round(float(np.mean(accs)), 4),
            "acc_std":       round(float(np.std(accs)), 4),
            "f1_mean":       round(float(np.mean(f1s)), 4),
            "f1_std":        round(float(np.std(f1s)), 4),
            "train_time_mean": round(float(np.mean([r["train_time"] for r in frac_seed_results])), 1),
            "infer_time_mean": round(float(np.mean([r["infer_time"] for r in frac_seed_results])), 3),
            "avg_per_class": avg_per_class,
            "per_seed":      frac_seed_results,
        }

        # Plots (first seed)
        plot_confusion(first_y_true, first_y_pred, frac, save_dir)
        plot_classf1(avg_per_class, frac, save_dir)

        # Save after each fraction (resume support)
        with open(detail_json_path, "w") as f:
            json.dump(all_detail, f, indent=2)
        print(f"  [Saved] {detail_json_path}", flush=True)

    # ── Summary ──
    print(f"\n{'='*60}", flush=True)
    print(f"  SetFit IMDB — COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    for frac_key in sorted(all_detail.keys(),
                            key=lambda k: all_detail[k]["fraction"]):
        s = all_detail[frac_key]
        print(f"  {s['fraction']*100:5.1f}%  n={s['n_train']:>6,}  "
              f"acc={s['acc_mean']:.4f}±{s['acc_std']:.4f}  "
              f"f1={s['f1_mean']:.4f}±{s['f1_std']:.4f}  "
              f"train={s['train_time_mean']:.1f}s", flush=True)
    print(f"\n  Results → {detail_json_path}", flush=True)
    print(f"  Plots  → {save_dir}/", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
