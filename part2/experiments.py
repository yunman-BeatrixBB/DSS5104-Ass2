"""
experiments.py — Member B: Main Experiment Script
DSS5104 Text Classification Assignment

Usage:
  python experiments.py --dataset agnews
  python experiments.py --dataset imdb

Results saved to results/member_b_{dataset}.json
"""

import os
import json
import argparse
import random
import numpy as np
import torch

from data_utils import (
    Vocabulary, load_agnews, load_imdb,
    stratified_subsample, text_length_stats
)
from models import build_model
from trainer import run_single_seed, error_analysis


# ─────────────────────────────────────────────
# Global Configuration
# ─────────────────────────────────────────────

SEEDS = [42, 123, 456]

# Data fractions for efficiency experiment
DATA_FRACTIONS = [1.0, 0.5, 0.25, 0.10, 0.05, 0.01]

# Best hyperparameters per model (selected via validation set)
BEST_HPS = {
    "fasttext": {"lr": 1e-3, "num_epochs": 10, "batch_size": 128, "embed_dim": 100},
    "textcnn":  {"lr": 1e-3, "num_epochs": 10, "batch_size": 64,  "embed_dim": 100,
                 "filter_sizes": [2, 3, 4], "num_filters": 128, "dropout": 0.5},
    "bilstm":   {"lr": 1e-3, "num_epochs": 10, "batch_size": 64,  "embed_dim": 100,
                 "hidden_dim": 128, "num_layers": 2, "dropout": 0.3},
}

MAX_LEN = {
    "agnews": 128,   # AG News texts are relatively short
    "imdb":   256,   # IMDB reviews are longer
}

LABEL_NAMES = {
    "agnews": ["World", "Sports", "Business", "Sci/Tech"],
    "imdb":   ["Negative", "Positive"],
}


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")
    return device


def summarize_seeds(results_list):
    """Aggregate multi-seed results into mean ± std"""
    accs = [r["accuracy"]   for r in results_list]
    f1s  = [r["macro_f1"]   for r in results_list]
    tts  = [r["train_time"] for r in results_list]
    its  = [r["infer_time"] for r in results_list]
    return {
        "acc_mean":  np.mean(accs),
        "acc_std":   np.std(accs),
        "f1_mean":   np.mean(f1s),
        "f1_std":    np.std(f1s),
        "train_time_mean": np.mean(tts),
        "infer_time_mean": np.mean(its),
    }


# ─────────────────────────────────────────────
# Hyperparameter Search (on validation set)
# ─────────────────────────────────────────────

def hyperparameter_search(
    model_name, vocab, num_classes, device,
    train_texts, train_labels, val_texts, val_labels,
    max_len, seed=42
):
    """
    Simple grid search over lr x num_epochs (select by val macro-F1).
    """
    print(f"\n{'='*50}")
    print(f"[HP Search] {model_name.upper()}")

    lr_candidates     = [1e-3, 5e-4]
    epoch_candidates  = [5, 10]

    best_f1 = -1
    best_hp = {}

    set_seed(seed)
    for lr in lr_candidates:
        for epochs in epoch_candidates:
            hp = {**BEST_HPS[model_name], "lr": lr, "num_epochs": epochs}
            model = build_model(model_name, len(vocab), num_classes, **hp).to(device)

            from trainer import (make_loader, train_model, evaluate)
            import torch.optim as optim
            import torch.nn as nn

            optimizer    = optim.Adam(model.parameters(), lr=lr)
            criterion    = nn.CrossEntropyLoss()
            train_loader = make_loader(train_texts, train_labels, vocab,
                                       hp["batch_size"], shuffle=True, max_len=max_len)
            val_loader   = make_loader(val_texts, val_labels, vocab,
                                       hp["batch_size"], shuffle=False, max_len=max_len)

            model, _ = train_model(model, train_loader, val_loader,
                                   optimizer, criterion, device,
                                   num_epochs=epochs, verbose=False)
            val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
            print(f"  lr={lr}  epochs={epochs}  =>  val_f1={val_f1:.4f}")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_hp = {"lr": lr, "num_epochs": epochs}

    print(f"  Best HP: {best_hp}  (val_f1={best_f1:.4f})")
    return best_hp


# ─────────────────────────────────────────────
# Full Data Experiment: 3 seeds
# ─────────────────────────────────────────────

def full_data_experiment(
    model_name, hp, vocab, num_classes, device,
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, max_len, label_names
):
    print(f"\n{'='*50}")
    print(f"[Full Data] {model_name.upper()}")

    seed_results = []
    for seed in SEEDS:
        print(f"\n  --- Seed {seed} ---")
        set_seed(seed)
        model = build_model(model_name, len(vocab), num_classes, **{**BEST_HPS[model_name], **hp}).to(device)
        res = run_single_seed(
            model,
            train_texts, train_labels,
            val_texts,   val_labels,
            test_texts,  test_labels,
            vocab, device,
            lr=hp["lr"],
            num_epochs=hp["num_epochs"],
            batch_size=BEST_HPS[model_name]["batch_size"],
            max_len=max_len,
        )
        seed_results.append(res)

    summary = summarize_seeds(seed_results)
    print(f"\n  [Result] acc={summary['acc_mean']:.4f}±{summary['acc_std']:.4f}  "
          f"f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}  "
          f"train={summary['train_time_mean']:.1f}s  "
          f"infer={summary['infer_time_mean']:.3f}s")

    # Error analysis using first seed results
    best_res = seed_results[0]
    error_analysis(test_texts, best_res["y_true"], best_res["y_pred"], label_names)

    return summary, seed_results


# ─────────────────────────────────────────────
# Data Efficiency Experiment
# ─────────────────────────────────────────────

def data_efficiency_experiment(
    model_name, hp, vocab, num_classes, device,
    train_texts, train_labels, val_texts, val_labels,
    test_texts, test_labels, max_len
):
    print(f"\n{'='*50}")
    print(f"[Data Efficiency] {model_name.upper()}")

    efficiency_results = {}

    for frac in DATA_FRACTIONS:
        frac_results = []
        for seed in SEEDS:
            set_seed(seed)

            # Stratified subsample by fraction
            sub_texts, sub_labels = stratified_subsample(
                train_texts, train_labels, fraction=frac, seed=seed
            )
            print(f"\n  fraction={frac*100:.0f}%  n_train={len(sub_texts)}  seed={seed}")

            model = build_model(model_name, len(vocab), num_classes,
                                **{**BEST_HPS[model_name], **hp}).to(device)
            res = run_single_seed(
                model,
                sub_texts, sub_labels,
                val_texts, val_labels,
                test_texts, test_labels,
                vocab, device,
                lr=hp["lr"],
                num_epochs=hp["num_epochs"],
                batch_size=BEST_HPS[model_name]["batch_size"],
                max_len=max_len,
                verbose=False,
            )
            frac_results.append(res)
            print(f"    acc={res['accuracy']:.4f}  f1={res['macro_f1']:.4f}  "
                  f"train={res['train_time']:.1f}s")

        summary = summarize_seeds(frac_results)
        efficiency_results[frac] = summary
        print(f"  [frac={frac*100:.0f}%] f1={summary['f1_mean']:.4f}±{summary['f1_std']:.4f}")

    return efficiency_results


# ─────────────────────────────────────────────
# Print Data Efficiency Summary Table
# ─────────────────────────────────────────────

def print_efficiency_table(all_model_results: dict):
    """
    all_model_results: {model_name: {fraction: summary_dict}}
    """
    print("\n\n" + "="*70)
    print("DATA EFFICIENCY TABLE  (macro-F1, mean ± std)")
    print("="*70)

    models = list(all_model_results.keys())
    fracs  = DATA_FRACTIONS

    # header
    header = f"{'Fraction':>10}" + "".join(f"  {m.upper():>16}" for m in models)
    print(header)
    print("-" * len(header))

    for frac in fracs:
        row = f"{frac*100:>9.0f}%"
        for m in models:
            s = all_model_results[m].get(frac, {})
            if s:
                row += f"  {s['f1_mean']:.4f}±{s['f1_std']:.4f}"
            else:
                row += f"  {'N/A':>16}"
        print(row)
    print("="*70)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["agnews", "imdb"], default="agnews")
    parser.add_argument("--train_path", default="data/ag_news/train.csv")
    parser.add_argument("--test_path",  default="data/ag_news/test.csv")
    parser.add_argument("--imdb_path",  default="data/imdb/IMDB Dataset.csv")
    parser.add_argument("--skip_hp_search", action="store_true",
                        help="Skip hyperparameter search and use defaults from BEST_HPS")
    parser.add_argument("--models", nargs="+",
                        default=["textcnn", "bilstm"],
                        help="List of models to run")
    args = parser.parse_args()

    device = get_device()
    os.makedirs("results", exist_ok=True)

    # ── Load data ──
    if args.dataset == "agnews":
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            load_agnews(args.train_path, args.test_path)
        num_classes = 4
    else:
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = \
            load_imdb(args.imdb_path)
        num_classes = 2

    max_len     = MAX_LEN[args.dataset]
    label_names = LABEL_NAMES[args.dataset]

    # Text length statistics (for report)
    text_length_stats(train_texts + val_texts + test_texts, name=args.dataset)

    # ── Build vocabulary (training set only) ──
    vocab = Vocabulary(max_size=30_000, min_freq=2)
    vocab.build(train_texts)

    # ── Run experiments for each model ──
    all_efficiency = {}
    all_full       = {}

    for model_name in args.models:
        # Hyperparameter search
        if not args.skip_hp_search:
            best_hp = hyperparameter_search(
                model_name, vocab, num_classes, device,
                train_texts, train_labels, val_texts, val_labels,
                max_len
            )
        else:
            best_hp = {
                "lr": BEST_HPS[model_name]["lr"],
                "num_epochs": BEST_HPS[model_name]["num_epochs"],
            }

        # Full data experiment
        summary, seed_results = full_data_experiment(
            model_name, best_hp, vocab, num_classes, device,
            train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels, max_len, label_names
        )
        all_full[model_name] = summary

        # Data efficiency experiment
        eff_results = data_efficiency_experiment(
            model_name, best_hp, vocab, num_classes, device,
            train_texts, train_labels, val_texts, val_labels,
            test_texts, test_labels, max_len
        )
        all_efficiency[model_name] = eff_results

    # ── Print summary ──
    print_efficiency_table(all_efficiency)

    print("\n\nFULL DATA SUMMARY")
    print("="*60)
    for m, s in all_full.items():
        print(f"  {m.upper():<12}  "
              f"acc={s['acc_mean']:.4f}±{s['acc_std']:.4f}  "
              f"f1={s['f1_mean']:.4f}±{s['f1_std']:.4f}  "
              f"train={s['train_time_mean']:.1f}s")

    # Save results (strip non-serializable fields)
    save_obj = {
        "dataset":    args.dataset,
        "full_data":  all_full,
        "efficiency": {
            m: {str(f): v for f, v in eff.items()}
            for m, eff in all_efficiency.items()
        },
    }
    out_path = f"results/member_b_{args.dataset}.json"
    with open(out_path, "w") as f:
        json.dump(save_obj, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
