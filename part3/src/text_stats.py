"""
text_stats.py — Text Length & Class Distribution Statistics
DSS5104 Text Classification Assignment

Reports for each dataset:
  - Sample counts per split
  - Class distribution (count + percentage)
  - Text length statistics (word-level): mean, median, std, min, 95th pct, max
  - Number and percentage of samples exceeding 128 / 256 / 512 tokens

Outputs:
  - Console print (same as before)
  - data/clean/text_stats.csv  (all statistics in one table)

Usage:
  python text_stats.py
"""

import os
import numpy as np
import pandas as pd
from collections import Counter


# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════

AGNEWS_TRAIN = "data/clean/ag_news_train.csv"
AGNEWS_TEST  = "data/clean/ag_news_test.csv"
IMDB_TRAIN   = "data/clean/imdb_train.csv"
IMDB_TEST    = "data/clean/imdb_test.csv"
OUTPUT_CSV   = "data/clean/text_stats.csv"

AGNEWS_LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
IMDB_LABEL_NAMES   = {0: "Negative", 1: "Positive"}


# ═══════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════

def word_lengths(texts: pd.Series) -> np.ndarray:
    """Return array of word counts per text."""
    return texts.fillna("").apply(lambda t: len(str(t).split())).values


def length_stats_row(dataset: str, split: str, lengths: np.ndarray) -> dict:
    """Build one row of statistics as a dict."""
    n = len(lengths)
    return {
        "dataset":       dataset,
        "split":         split,
        "n_samples":     n,
        "mean":          round(lengths.mean(), 1),
        "median":        round(float(np.median(lengths)), 1),
        "std":           round(lengths.std(), 1),
        "min":           int(lengths.min()),
        "95th_pct":      round(float(np.percentile(lengths, 95)), 1),
        "max":           int(lengths.max()),
        "gt_128":        int((lengths > 128).sum()),
        "gt_128_pct":    round((lengths > 128).sum() / n * 100, 2),
        "gt_256":        int((lengths > 256).sum()),
        "gt_256_pct":    round((lengths > 256).sum() / n * 100, 2),
        "gt_512":        int((lengths > 512).sum()),
        "gt_512_pct":    round((lengths > 512).sum() / n * 100, 2),
    }


def class_dist_rows(dataset: str, split: str, df: pd.DataFrame,
                    label_col: str, label_names: dict) -> list:
    """Build class distribution rows."""
    dist = Counter(df[label_col])
    total = len(df)
    rows = []
    for k in sorted(dist.keys()):
        rows.append({
            "dataset":    dataset,
            "split":      split,
            "class_id":   k,
            "class_name": label_names.get(k, str(k)),
            "count":      dist[k],
            "percentage":  round(dist[k] / total * 100, 1),
        })
    return rows


def print_length_stats(lengths: np.ndarray, label: str = ""):
    """Print text length statistics for a given split."""
    print(f"\n  [{label}] Text Length (word count), n={len(lengths):,}")
    print(f"    Mean    : {lengths.mean():.1f}")
    print(f"    Median  : {np.median(lengths):.1f}")
    print(f"    Std     : {lengths.std():.1f}")
    print(f"    Min     : {lengths.min()}")
    print(f"    95th pct: {np.percentile(lengths, 95):.1f}")
    print(f"    Max     : {lengths.max()}")

    for threshold in [128, 256, 512]:
        over = (lengths > threshold).sum()
        pct  = over / len(lengths) * 100
        print(f"    >{threshold} words: {over:>6,} ({pct:.2f}%)")


def print_class_dist(df: pd.DataFrame, label_col: str,
                     label_names: dict, split_name: str):
    """Print class distribution for one split."""
    dist = Counter(df[label_col])
    total = len(df)
    print(f"  {split_name} ({total:,} samples):")
    for k in sorted(dist.keys()):
        name  = label_names.get(k, str(k))
        count = dist[k]
        pct   = count / total * 100
        print(f"    {name:<12s}: {count:>6,}  ({pct:.1f}%)")


# ═══════════════════════════════════════════════
# DATASET ANALYSIS
# ═══════════════════════════════════════════════

def analyze_agnews(length_rows: list, class_rows: list):
    """Analyze AG News: combine title + description as text for length stats."""
    print("\n" + "=" * 60)
    print("  AG News (4-class topic classification)")
    print("=" * 60)

    train_df = pd.read_csv(AGNEWS_TRAIN)
    test_df  = pd.read_csv(AGNEWS_TEST)

    train_df["text"] = train_df["title"].fillna("") + " " + train_df["description"].fillna("")
    test_df["text"]  = test_df["title"].fillna("")  + " " + test_df["description"].fillna("")

    # Class distribution
    print("\n  CLASS DISTRIBUTION")
    print("  " + "─" * 40)
    print_class_dist(train_df, "label", AGNEWS_LABEL_NAMES, "Train")
    print_class_dist(test_df,  "label", AGNEWS_LABEL_NAMES, "Test")
    class_rows.extend(class_dist_rows("AG News", "Train", train_df, "label", AGNEWS_LABEL_NAMES))
    class_rows.extend(class_dist_rows("AG News", "Test",  test_df,  "label", AGNEWS_LABEL_NAMES))

    # Text length statistics
    print("\n  TEXT LENGTH STATISTICS")
    print("  " + "─" * 40)
    train_lens = word_lengths(train_df["text"])
    test_lens  = word_lengths(test_df["text"])
    all_lens   = np.concatenate([train_lens, test_lens])

    print_length_stats(train_lens, "Train")
    print_length_stats(test_lens,  "Test")
    print_length_stats(all_lens,   "Overall")

    length_rows.append(length_stats_row("AG News", "Train",   train_lens))
    length_rows.append(length_stats_row("AG News", "Test",    test_lens))
    length_rows.append(length_stats_row("AG News", "Overall", all_lens))


def analyze_imdb(length_rows: list, class_rows: list):
    """Analyze IMDB: review column as text."""
    print("\n" + "=" * 60)
    print("  IMDB (2-class sentiment analysis)")
    print("=" * 60)

    train_df = pd.read_csv(IMDB_TRAIN)
    test_df  = pd.read_csv(IMDB_TEST)

    # Class distribution
    print("\n  CLASS DISTRIBUTION")
    print("  " + "─" * 40)
    print_class_dist(train_df, "label", IMDB_LABEL_NAMES, "Train")
    print_class_dist(test_df,  "label", IMDB_LABEL_NAMES, "Test")
    class_rows.extend(class_dist_rows("IMDB", "Train", train_df, "label", IMDB_LABEL_NAMES))
    class_rows.extend(class_dist_rows("IMDB", "Test",  test_df,  "label", IMDB_LABEL_NAMES))

    # Text length statistics
    print("\n  TEXT LENGTH STATISTICS")
    print("  " + "─" * 40)
    train_lens = word_lengths(train_df["review"])
    test_lens  = word_lengths(test_df["review"])
    all_lens   = np.concatenate([train_lens, test_lens])

    print_length_stats(train_lens, "Train")
    print_length_stats(test_lens,  "Test")
    print_length_stats(all_lens,   "Overall")

    length_rows.append(length_stats_row("IMDB", "Train",   train_lens))
    length_rows.append(length_stats_row("IMDB", "Test",    test_lens))
    length_rows.append(length_stats_row("IMDB", "Overall", all_lens))


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

def run_stats():
    print("=" * 60)
    print("  DSS5104 — TEXT STATISTICS REPORT")
    print("=" * 60)

    length_rows = []
    class_rows  = []

    if os.path.exists(AGNEWS_TRAIN) and os.path.exists(AGNEWS_TEST):
        analyze_agnews(length_rows, class_rows)
    else:
        print("\n[SKIP] AG News CSV not found")

    if os.path.exists(IMDB_TRAIN) and os.path.exists(IMDB_TEST):
        analyze_imdb(length_rows, class_rows)
    else:
        print("\n[SKIP] IMDB CSV not found")

    # ── Save to CSV ──
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    length_df = pd.DataFrame(length_rows)
    class_df  = pd.DataFrame(class_rows)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        # ── Section 1: Overview ──
        f.write("DSS5104 Text Classification — Dataset Statistics Report\n")
        f.write("\n")
        f.write("[Datasets]\n")
        f.write("AG News: 4-class topic classification (World / Sports / Business / Sci-Tech). Pre-split train/test from Kaggle. Text = title + description.\n")
        f.write("IMDB: 2-class sentiment analysis (Negative / Positive). 50K reviews from Kaggle. Stratified 90/10 train/test split (seed=42). HTML tags removed.\n")
        f.write("\n")

        # ── Section 2: Class Distribution ──
        f.write("[Class Distribution]\n")
        f.write("Both datasets are perfectly balanced — no class imbalance issue. Accuracy and macro-F1 are expected to be consistent.\n")
        class_df.to_csv(f, index=False)
        f.write("\n")

        # ── Section 3: Text Length Statistics ──
        f.write("[Text Length Statistics (word count)]\n")
        f.write("AG News texts are very short (mean ~38 words). No sample exceeds 256 words. Transformer 512-token limit has zero impact. TF-IDF and n-gram features can capture topic-distinctive vocabulary effectively.\n")
        f.write("IMDB reviews are much longer (mean ~229 words) with high variance (std ~170). About 7.2% of reviews exceed 512 words and will be truncated by BERT. Sentiment analysis requires understanding context and word order — advantages for Transformer and BiLSTM over bag-of-words.\n")
        length_df.to_csv(f, index=False)
        f.write("\n")

        # ── Section 4: Implications ──
        f.write("[Implications for Model Selection]\n")
        f.write("AG News: Short texts with clear topic keywords favor classical methods (TF-IDF + LR/SVM). Transformer pre-training may offer limited advantage.\n")
        f.write("IMDB: Long texts requiring contextual understanding favor Transformers. ~7% truncation at 512 tokens is a minor concern but worth noting in the report.\n")
        f.write("Data Efficiency: AG News — TF-IDF likely competitive even at low fractions. IMDB — Transformer pre-trained knowledge may shine at 1-10% data.\n")

    print(f"\n  Saved → {OUTPUT_CSV}")
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    run_stats()
