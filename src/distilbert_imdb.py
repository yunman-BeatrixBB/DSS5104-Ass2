"""
distilbert_imdb_v3_merged.py — DistilBERT on IMDB (with checkpoint/resume)
DSS5104 Text Classification Assignment (Tier 3)

Merged from:
  - distilbert_imdb_v3_opt.py  (main pipeline)
  - efficiency_distibert_imdb.py (detailed efficiency outputs)

Changes:
  - Fixed: total_mem → total_memory (attribute name)
  - Data efficiency experiment skips frac=1.0 (already covered by full_data_experiment)
  - Efficiency experiment now outputs per-class metrics, confusion matrices,
    per-class F1 bar charts per fraction, and a summary comparison plot

Usage:
  python distilbert_imdb_v3_merged.py
"""

import os, sys, json, time, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════

MODEL_NAME    = "distilbert-base-uncased"
DATASET_NAME  = "imdb"
NUM_CLASSES   = 2
LABEL_NAMES   = ["Negative", "Positive"]
MAX_LEN       = 512
BATCH_SIZE    = 48          # optimized for 50-series GPU
NUM_EPOCHS    = 5
PATIENCE      = 2
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
SEEDS         = [123, 456, 789]
LR_CANDIDATES = [1e-5, 2e-5, 5e-5]
# 1.0 is excluded — full data is already covered by full_data_experiment
DATA_FRACTIONS = [0.5, 0.25, 0.10, 0.05, 0.01]
VAL_RATIO     = 0.1

# Paths — auto-detect Colab vs local
if os.path.exists("/content/drive"):
    BASE_DIR = "/content/drive/MyDrive/part3"
else:
    BASE_DIR = "."

DATA_DIR    = os.path.join(BASE_DIR, "data/clean")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
EFFICIENCY_DETAIL_DIR = os.path.join(RESULTS_DIR, "efficiency_detail_distilbert")
SHORT_NAME  = "distilbert"

# Checkpoint file — saves progress so we can resume
CHECKPOINT_PATH = os.path.join(RESULTS_DIR, f"{SHORT_NAME}_{DATASET_NAME}_checkpoint.json")


# ═══════════════════════════════════════════════
# CHECKPOINT HELPERS
# ═══════════════════════════════════════════════

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            ckpt = json.load(f)
        print(f"[Resume] Loaded checkpoint: {list(ckpt.keys())}", flush=True)
        return ckpt
    return {}

def save_checkpoint(ckpt):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"[Checkpoint] Saved → {CHECKPOINT_PATH}", flush=True)


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
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # FIXED: was total_mem
        print(f"  GPU: {name} ({mem:.1f} GB)", flush=True)
        print(f"  TF32 + cudnn.benchmark enabled", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}", flush=True)
    return device


# ═══════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════

def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "imdb_train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "imdb_test.csv"))

    train_df["text"] = train_df["review"].fillna("")
    test_df["text"]  = test_df["review"].fillna("")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(), train_df["label"].tolist(),
        test_size=VAL_RATIO, stratify=train_df["label"].tolist(), random_state=42,
    )

    test_texts  = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    print(f"[Data] train={len(train_texts):,}  val={len(val_texts):,}  test={len(test_texts):,}", flush=True)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


def make_loader(texts, labels, tokenizer, max_len, batch_size, shuffle=False):
    ds = TextDataset(texts, labels, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=4, persistent_workers=True)


def stratified_subsample(texts, labels, fraction, seed=42):
    if fraction >= 1.0:
        return texts, labels
    n = max(int(len(texts) * fraction), len(set(labels)))
    idx, _ = train_test_split(
        range(len(texts)), train_size=n, stratify=labels, random_state=seed,
    )
    return [texts[i] for i in idx], [labels[i] for i in idx]


# ═══════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    acc = accuracy_score(targets, preds)
    f1  = f1_score(targets, preds, average="macro", zero_division=0)

    prec, rec, f1_per, sup = precision_recall_fscore_support(
        targets, preds, average=None, zero_division=0,
    )
    per_class = {}
    for i, name in enumerate(LABEL_NAMES):
        per_class[name] = {
            "precision": round(prec[i], 4),
            "recall":    round(rec[i], 4),
            "f1":        round(f1_per[i], 4),
            "support":   int(sup[i]),
            "accuracy":  round(
                sum(1 for t, p in zip(targets, preds) if t == i and p == i)
                / max(sum(1 for t in targets if t == i), 1), 4
            ),
        }

    avg_loss = total_loss / len(loader)
    return acc, f1, per_class, targets, preds, avg_loss


def train_and_evaluate(
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, tokenizer, device,
    lr, seed, verbose=True,
):
    set_seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES,
    ).to(device)

    train_loader = make_loader(train_texts, train_labels, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(val_texts, val_labels, tokenizer, MAX_LEN, BATCH_SIZE)
    test_loader  = make_loader(test_texts, test_labels, tokenizer, MAX_LEN, BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_state  = None
    patience_cnt = 0
    epoch_logs = []
    scaler = torch.amp.GradScaler("cuda")

    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler)
        val_acc, val_f1, _, _, _, val_loss = evaluate(model, val_loader, device)

        log = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
            "val_f1":     round(val_f1, 4),
        }
        epoch_logs.append(log)

        if verbose:
            print(f"  Epoch {epoch}/{NUM_EPOCHS} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.4f} | val_f1={val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}", flush=True)
                break

    train_time = time.time() - t_start

    model.load_state_dict(best_state)
    model.to(device)

    t_infer = time.time()
    test_acc, test_f1, per_class, y_true, y_pred, _ = evaluate(model, test_loader, device)
    infer_time = time.time() - t_infer

    return {
        "seed":        seed,
        "best_lr":     lr,
        "accuracy":    round(test_acc, 4),
        "macro_f1":    round(test_f1, 4),
        "per_class":   per_class,
        "train_time":  round(train_time, 1),
        "infer_time":  round(infer_time, 3),
        "epoch_logs":  epoch_logs,
        "best_epoch":  len(epoch_logs) - patience_cnt if patience_cnt > 0 else len(epoch_logs),
        "y_true":      y_true,
        "y_pred":      y_pred,
    }


# ═══════════════════════════════════════════════
# HP SEARCH (with resume)
# ═══════════════════════════════════════════════

def hp_search(train_texts, train_labels, val_texts, val_labels,
              test_texts, test_labels, tokenizer, device, ckpt):
    print("\n" + "=" * 60, flush=True)
    print(f"[HP Search] {MODEL_NAME} on {DATASET_NAME}", flush=True)
    print("=" * 60, flush=True)

    hp_results = ckpt.get("hp_results", {})
    best_lr = None
    best_f1 = -1.0

    for lr in LR_CANDIDATES:
        lr_key = str(lr)
        if lr_key in hp_results:
            val_f1 = hp_results[lr_key]
            print(f"\n  [SKIP] lr={lr} → best_val_f1={val_f1:.4f} (cached)", flush=True)
        else:
            print(f"\n  Testing lr={lr}", flush=True)
            res = train_and_evaluate(
                train_texts, train_labels, val_texts, val_labels,
                test_texts, test_labels, tokenizer, device,
                lr=lr, seed=SEEDS[0], verbose=True,
            )
            val_f1 = max(log["val_f1"] for log in res["epoch_logs"])
            hp_results[lr_key] = round(val_f1, 4)
            print(f"  lr={lr} → best_val_f1={val_f1:.4f}", flush=True)

            # Save after each LR tested
            ckpt["hp_results"] = hp_results
            save_checkpoint(ckpt)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_lr = lr

    ckpt["best_lr"] = best_lr
    save_checkpoint(ckpt)
    print(f"\n  ★ Best lr={best_lr} (val_f1={best_f1:.4f})", flush=True)
    return best_lr, hp_results


# ═══════════════════════════════════════════════
# FULL DATA EXPERIMENT (with resume)
# ═══════════════════════════════════════════════

def full_data_experiment(train_texts, train_labels, val_texts, val_labels,
                         test_texts, test_labels, tokenizer, device, best_lr, ckpt):
    print("\n" + "=" * 60, flush=True)
    print(f"[Full Data] {MODEL_NAME} on {DATASET_NAME}", flush=True)
    print("=" * 60, flush=True)

    cached_results = ckpt.get("full_data_per_seed", [])
    done_seeds = {r["seed"] for r in cached_results}
    all_results = list(cached_results)

    for seed in SEEDS:
        if seed in done_seeds:
            r = next(r for r in cached_results if r["seed"] == seed)
            print(f"\n  [SKIP] Seed {seed} → acc={r['accuracy']:.4f} f1={r['macro_f1']:.4f} (cached)", flush=True)
            continue

        print(f"\n  --- Seed {seed} ---", flush=True)
        res = train_and_evaluate(
            train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels, tokenizer, device,
            lr=best_lr, seed=seed, verbose=True,
        )
        # Strip large arrays for checkpoint (keep y_true/y_pred only for first seed)
        res_save = {k: v for k, v in res.items() if k not in ("y_true", "y_pred")}
        if seed == SEEDS[0]:
            res_save["y_true"] = res["y_true"]
            res_save["y_pred"] = res["y_pred"]

        all_results.append(res_save)
        print(f"  acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}  "
              f"train={res['train_time']:.1f}s  infer={res['infer_time']:.3f}s", flush=True)

        ckpt["full_data_per_seed"] = all_results
        save_checkpoint(ckpt)

    summary = summarize_results(all_results)
    ckpt["full_data"] = summary
    save_checkpoint(ckpt)

    print(f"\n  [MEAN] acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}  "
          f"f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}", flush=True)

    return all_results, summary


def summarize_results(results_list):
    accs = [r["accuracy"]  for r in results_list]
    f1s  = [r["macro_f1"]  for r in results_list]
    tts  = [r["train_time"] for r in results_list]
    its  = [r["infer_time"] for r in results_list]
    return {
        "acc_mean":        round(np.mean(accs), 4),
        "acc_std":         round(np.std(accs), 4),
        "f1_mean":         round(np.mean(f1s), 4),
        "f1_std":          round(np.std(f1s), 4),
        "train_time_mean": round(np.mean(tts), 1),
        "infer_time_mean": round(np.mean(its), 3),
    }


# ═══════════════════════════════════════════════
# DATA EFFICIENCY EXPERIMENT (with resume + detailed outputs)
# Merged from efficiency_distibert_imdb.py
# NOTE: frac=1.0 is excluded — already covered by full_data_experiment
# ═══════════════════════════════════════════════

def data_efficiency_experiment(train_texts, train_labels, val_texts, val_labels,
                               test_texts, test_labels, tokenizer, device, best_lr, ckpt):
    print("\n" + "=" * 60, flush=True)
    print(f"[Data Efficiency] {MODEL_NAME} on {DATASET_NAME}", flush=True)
    print(f"  Fractions (excluding 1.0): {DATA_FRACTIONS}", flush=True)
    print("=" * 60, flush=True)

    os.makedirs(EFFICIENCY_DETAIL_DIR, exist_ok=True)

    efficiency_results = ckpt.get("efficiency", {})

    # Load detailed results (for per-class metrics & plots)
    detail_json_path = os.path.join(EFFICIENCY_DETAIL_DIR, "efficiency_detail_distilbert.json")
    if os.path.exists(detail_json_path):
        with open(detail_json_path, "r") as f:
            all_detail = json.load(f)
        print(f"[Resume] Loaded {len(all_detail)} fraction detail results", flush=True)
    else:
        all_detail = {}

    for frac in DATA_FRACTIONS:
        frac_key = str(frac)
        detail_key = f"{frac*100:.0f}pct"

        if frac_key in efficiency_results and detail_key in all_detail:
            s = efficiency_results[frac_key]
            print(f"\n  [SKIP] frac={frac*100:.0f}% → f1={s['f1_mean']:.4f}±{s['f1_std']:.4f} (cached)", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"  Fraction: {frac*100:.0f}%", flush=True)
        print(f"{'='*60}", flush=True)

        frac_results = []
        frac_seed_detail = []
        first_y_true, first_y_pred = None, None

        for seed in SEEDS:
            set_seed(seed)
            sub_texts, sub_labels = stratified_subsample(
                train_texts, train_labels, fraction=frac, seed=seed,
            )
            print(f"\n  Seed={seed}  n_train={len(sub_texts):,}", flush=True)

            res = train_and_evaluate(
                sub_texts, sub_labels, val_texts, val_labels,
                test_texts, test_labels, tokenizer, device,
                lr=best_lr, seed=seed, verbose=False,
            )

            frac_results.append({k: v for k, v in res.items() if k not in ("y_true", "y_pred")})
            frac_seed_detail.append({
                "seed": seed,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "per_class": res["per_class"],
                "train_time": res["train_time"],
            })

            print(f"    acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}  "
                  f"train={res['train_time']:.1f}s", flush=True)

            if first_y_true is None:
                first_y_true = res["y_true"]
                first_y_pred = res["y_pred"]

        # Summary for checkpoint
        summary = summarize_results(frac_results)
        efficiency_results[frac_key] = summary
        print(f"  [{frac*100:.0f}%] f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}", flush=True)

        # Detailed per-class averages
        avg_per_class = {}
        for name in LABEL_NAMES:
            avg_per_class[name] = {
                "precision": round(np.mean([r["per_class"][name]["precision"] for r in frac_seed_detail]), 4),
                "recall":    round(np.mean([r["per_class"][name]["recall"]    for r in frac_seed_detail]), 4),
                "f1":        round(np.mean([r["per_class"][name]["f1"]        for r in frac_seed_detail]), 4),
            }

        all_detail[detail_key] = {
            "fraction": frac,
            "n_train": len(stratified_subsample(train_texts, train_labels, frac, seed=42)[0]),
            "acc_mean": summary["acc_mean"],
            "acc_std":  summary["acc_std"],
            "f1_mean":  summary["f1_mean"],
            "f1_std":   summary["f1_std"],
            "avg_per_class": avg_per_class,
            "per_seed": frac_seed_detail,
        }

        # Per-fraction plots (confusion matrix + per-class F1)
        if first_y_true is not None:
            plot_efficiency_confusion(first_y_true, first_y_pred, frac, EFFICIENCY_DETAIL_DIR)
            plot_efficiency_classf1(avg_per_class, frac, EFFICIENCY_DETAIL_DIR)

        # Save after each fraction (checkpoint + detail)
        ckpt["efficiency"] = efficiency_results
        save_checkpoint(ckpt)
        with open(detail_json_path, "w") as f:
            json.dump(all_detail, f, indent=2)
        print(f"  [Saved] {detail_json_path}", flush=True)

    # Summary comparison plot across all fractions
    if len(all_detail) > 1:
        plot_efficiency_summary(all_detail, EFFICIENCY_DETAIL_DIR)

    return efficiency_results


# ═══════════════════════════════════════════════
# EFFICIENCY PLOTS (from efficiency_distibert_imdb.py)
# ═══════════════════════════════════════════════

def plot_efficiency_confusion(y_true, y_pred, frac, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"DistilBERT IMDB — {frac*100:.0f}% Data — Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_{frac*100:.0f}pct.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [Plot] {path}", flush=True)


def plot_efficiency_classf1(per_class, frac, save_dir):
    names = list(per_class.keys())
    f1s   = [per_class[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(names, f1s, color=colors, edgecolor="white")
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{f1:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"DistilBERT IMDB — {frac*100:.0f}% Data — Per-Class F1")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(save_dir, f"classf1_{frac*100:.0f}pct.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"    [Plot] {path}", flush=True)


def plot_efficiency_summary(all_detail, save_dir):
    """Summary bar chart: per-class F1 across all data fractions."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fracs_sorted = sorted(all_detail.keys(), key=lambda k: all_detail[k]["fraction"])
    x = np.arange(len(fracs_sorted))
    width = 0.35
    neg_f1s = [all_detail[k]["avg_per_class"]["Negative"]["f1"] for k in fracs_sorted]
    pos_f1s = [all_detail[k]["avg_per_class"]["Positive"]["f1"] for k in fracs_sorted]
    ax.bar(x - width/2, neg_f1s, width, label="Negative", color="#e74c3c")
    ax.bar(x + width/2, pos_f1s, width, label="Positive", color="#2ecc71")
    for i in range(len(x)):
        ax.text(x[i] - width/2, neg_f1s[i] + 0.005, f"{neg_f1s[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + width/2, pos_f1s[i] + 0.005, f"{pos_f1s[i]:.3f}", ha="center", fontsize=8)
    pct_labels = [f"{all_detail[k]['fraction']*100:.0f}%" for k in fracs_sorted]
    ax.set_xticks(x); ax.set_xticklabels(pct_labels)
    ax.set_xlabel("Training Data Fraction"); ax.set_ylabel("F1 Score")
    ax.set_title("DistilBERT IMDB — Per-Class F1 Across Data Fractions")
    ax.set_ylim(0.5, 1.0); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "classf1_all_fractions.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"\n  [Plot] {path}", flush=True)


# ═══════════════════════════════════════════════
# ERROR ANALYSIS
# ═══════════════════════════════════════════════

def error_analysis(test_texts, y_true, y_pred, save_prefix):
    print("\n" + "=" * 60, flush=True)
    print("[Error Analysis]", flush=True)
    print("=" * 60, flush=True)

    print("\n--- Classification Report ---", flush=True)
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0), flush=True)

    print("--- Per-Class Accuracy ---", flush=True)
    for i, name in enumerate(LABEL_NAMES):
        cls_total = sum(1 for t in y_true if t == i)
        cls_correct = sum(1 for t, p in zip(y_true, y_pred) if t == i and p == i)
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        print(f"  {name:<12s}: {cls_acc:.4f}  ({cls_correct}/{cls_total})", flush=True)

    errors = [(test_texts[i], y_true[i], y_pred[i])
              for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    print(f"\nTotal errors: {len(errors)} / {len(y_true)} "
          f"({len(errors)/len(y_true)*100:.1f}%)", flush=True)
    print(f"\n--- 25 Misclassified Examples ---", flush=True)
    for i, (text, true, pred) in enumerate(errors[:25]):
        snippet = text[:150] + ("..." if len(text) > 120 else "")
        print(f"  [{i+1:2d}] TRUE={LABEL_NAMES[true]:<12s} PRED={LABEL_NAMES[pred]:<12s} {snippet}", flush=True)

    pred_df = pd.DataFrame({
        "text":       test_texts,
        "true_label": [LABEL_NAMES[t] for t in y_true],
        "pred_label": [LABEL_NAMES[p] for p in y_pred],
        "correct":    [t == p for t, p in zip(y_true, y_pred)],
    })
    csv_path = f"{save_prefix}_errors.csv"
    pred_df.to_csv(csv_path, index=False)
    print(f"\n  Predictions saved → {csv_path}", flush=True)


# ═══════════════════════════════════════════════
# PLOTTING (full data)
# ═══════════════════════════════════════════════

def plot_training_curves(all_results, save_prefix):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 4), squeeze=False)
    for idx, res in enumerate(all_results):
        ax = axes[0][idx]
        logs = res["epoch_logs"]
        epochs = [l["epoch"] for l in logs]
        train_losses = [l["train_loss"] for l in logs]
        val_losses   = [l["val_loss"] for l in logs]
        ax.plot(epochs, train_losses, "o-", label="Train Loss", color="#e74c3c")
        ax.plot(epochs, val_losses,   "s-", label="Val Loss",   color="#3498db")
        best_ep = res["best_epoch"]
        ax.axvline(x=best_ep, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch={best_ep}")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"Seed {res['seed']}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"{MODEL_NAME} on {DATASET_NAME.upper()} — Training Curves", fontsize=13)
    plt.tight_layout()
    path = f"{save_prefix}_train_curve.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [Plot saved] {path}", flush=True)


def plot_confusion_matrix(y_true, y_pred, save_prefix):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{MODEL_NAME} on {DATASET_NAME.upper()} — Confusion Matrix")
    plt.tight_layout()
    path = f"{save_prefix}_confusion.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [Plot saved] {path}", flush=True)


def plot_per_class_f1(per_class, save_prefix):
    names = list(per_class.keys())
    f1s   = [per_class[n]["f1"] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#e74c3c" if f == min(f1s) else "#2ecc71" if f == max(f1s) else "#3498db" for f in f1s]
    bars = ax.bar(names, f1s, color=colors, edgecolor="white", linewidth=0.5)
    for bar, f1 in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"{MODEL_NAME} on {DATASET_NAME.upper()} — Per-Class F1")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{save_prefix}_classf1.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  [Plot saved] {path}", flush=True)


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def run_distilbert_imdb():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_prefix = os.path.join(RESULTS_DIR, f"{SHORT_NAME}_{DATASET_NAME}")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data()

    # Load checkpoint
    ckpt = load_checkpoint()

    # 1. HP Search
    best_lr, hp_results = hp_search(
        train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels, tokenizer, device, ckpt,
    )

    # 2. Full Data Experiment (100% data, multi-seed)
    all_results, full_summary = full_data_experiment(
        train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels, tokenizer, device, best_lr, ckpt,
    )

    # 3. Data Efficiency (sub-fractions only, no 1.0 — already done above)
    efficiency_results = data_efficiency_experiment(
        train_texts, train_labels, val_texts, val_labels,
        test_texts, test_labels, tokenizer, device, best_lr, ckpt,
    )

    # 4. Error Analysis (use first seed's y_true/y_pred from checkpoint)
    first_res = all_results[0]
    if "y_true" in first_res and "y_pred" in first_res:
        error_analysis(test_texts, first_res["y_true"], first_res["y_pred"], save_prefix)
    else:
        print("[WARN] No y_true/y_pred in checkpoint, skipping error analysis", flush=True)

    # 5. Plots
    print("\n[Generating plots...]", flush=True)
    plot_training_curves(all_results, save_prefix)
    if "y_true" in first_res and "y_pred" in first_res:
        plot_confusion_matrix(first_res["y_true"], first_res["y_pred"], save_prefix)
    if "per_class" in first_res:
        plot_per_class_f1(first_res["per_class"], save_prefix)

    # 6. Save final results
    save_obj = {
        "model":       MODEL_NAME,
        "dataset":     DATASET_NAME,
        "best_lr":     best_lr,
        "hp_search":   hp_results,
        "seeds":       SEEDS,
        "full_data":   full_summary,
        "full_data_per_seed": [
            {k: v for k, v in r.items() if k not in ("y_true", "y_pred")}
            for r in all_results
        ],
        "efficiency":  efficiency_results,
    }
    json_path = f"{save_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"\n  Results saved → {json_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"  {MODEL_NAME} on {DATASET_NAME.upper()} — COMPLETE", flush=True)
    print(f"  Best LR: {best_lr}", flush=True)
    print(f"  Accuracy: {full_summary['acc_mean']:.4f} ± {full_summary['acc_std']:.4f}", flush=True)
    print(f"  Macro-F1: {full_summary['f1_mean']:.4f} ± {full_summary['f1_std']:.4f}", flush=True)
    print(f"  Train time: {full_summary['train_time_mean']:.1f}s", flush=True)
    print(f"  Infer time: {full_summary['infer_time_mean']:.3f}s", flush=True)
    print(f"  Efficiency detail → {EFFICIENCY_DETAIL_DIR}/", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    run_distilbert_imdb()
