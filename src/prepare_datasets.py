"""
prepare_datasets.py — Data Remapping & Splitting
DSS5104 Text Classification Assignment

AG News:
  - Raw columns: Class Index (1-4), Title, Description
  - Remap labels 1-4 → 0-3  (0=World, 1=Sports, 2=Business, 3=Sci/Tech)
  - Keep original train/test split
  - Output: ag_news_train.csv, ag_news_test.csv  (columns: label, title, description)

IMDB:
  - Raw columns: review, sentiment (positive/negative)
  - Remove HTML tags (<br />, etc.)
  - Map sentiment → 0 (negative) / 1 (positive)
  - Stratified split into train (90%) / test (10%)
  - Output: imdb_train.csv, imdb_test.csv  (columns: review, label)

Usage:
  python prepare_datasets.py
"""

import os
import re
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════

AGNEWS_RAW_TRAIN = "data/ag_news/train.csv"
AGNEWS_RAW_TEST  = "data/ag_news/test.csv"
IMDB_RAW_PATH    = "data/imdb/IMDB Dataset.csv"
OUTPUT_DIR       = "data/clean"

SEED = 42
IMDB_TEST_RATIO = 0.1

AGNEWS_LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
IMDB_LABEL_NAMES   = {0: "Negative", 1: "Positive"}

# ═══════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════

def remove_html_tags(text: str) -> str:
    """Remove HTML tags only, keep everything else intact."""
    return re.sub(r"<[^>]+>", " ", str(text)).strip()


def process_agnews(train_path: str, test_path: str):
    """
    Load AG News, remap labels 1-4 → 0-3, keep original split.
    Returns (train_df, test_df) with columns [label, title, description].
    """
    train_df = pd.read_csv(train_path, header=0)
    test_df  = pd.read_csv(test_path, header=0)

    train_df.columns = ["label", "title", "description"]
    test_df.columns  = ["label", "title", "description"]

    # Remap labels: 1-4 → 0-3
    train_df["label"] = train_df["label"] - 1
    test_df["label"]  = test_df["label"] - 1

    return train_df, test_df


def process_imdb(path: str, test_ratio: float = 0.1, seed: int = 42):
    """
    Load IMDB, remove HTML tags, map sentiment → 0/1,
    stratified split into train/test.
    Returns (train_df, test_df) with columns [review, label].
    """
    df = pd.read_csv(path)

    # Remove HTML tags from reviews
    df["review"] = df["review"].apply(remove_html_tags)

    # Map sentiment → numeric label
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # Stratified train/test split
    train_df, test_df = train_test_split(
        df[["review", "label"]],
        test_size=test_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return train_df, test_df


def print_summary(name: str, train_df, test_df, label_col: str, label_names: dict):
    """Print dataset summary statistics."""
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Train : {len(train_df):>6,}")
    print(f"  Test  : {len(test_df):>6,}")

    for split_name, df in [("Train", train_df), ("Test", test_df)]:
        dist = Counter(df[label_col])
        dist_str = "  ".join(f"{label_names[k]}={v}" for k, v in sorted(dist.items()))
        print(f"  {split_name} labels: {dist_str}")


# ═══════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════

def run_prepare():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("DSS5104 — DATA PREPARATION")
    print("=" * 50)

    # ── AG News ──
    if os.path.exists(AGNEWS_RAW_TRAIN) and os.path.exists(AGNEWS_RAW_TEST):
        ag_train, ag_test = process_agnews(AGNEWS_RAW_TRAIN, AGNEWS_RAW_TEST)

        ag_train_path = os.path.join(OUTPUT_DIR, "ag_news_train.csv")
        ag_test_path  = os.path.join(OUTPUT_DIR, "ag_news_test.csv")
        ag_train.to_csv(ag_train_path, index=False)
        ag_test.to_csv(ag_test_path, index=False)

        print_summary("AG News", ag_train, ag_test, "label", AGNEWS_LABEL_NAMES)
        print(f"  Saved → {ag_train_path}")
        print(f"  Saved → {ag_test_path}")
    else:
        print(f"\n[SKIP] AG News not found at {AGNEWS_RAW_TRAIN}")

    # ── IMDB ──
    if os.path.exists(IMDB_RAW_PATH):
        imdb_train, imdb_test = process_imdb(
            IMDB_RAW_PATH, test_ratio=IMDB_TEST_RATIO, seed=SEED
        )

        imdb_train_path = os.path.join(OUTPUT_DIR, "imdb_train.csv")
        imdb_test_path  = os.path.join(OUTPUT_DIR, "imdb_test.csv")
        imdb_train.to_csv(imdb_train_path, index=False)
        imdb_test.to_csv(imdb_test_path, index=False)

        print_summary("IMDB", imdb_train, imdb_test, "label", IMDB_LABEL_NAMES)
        print(f"  Saved → {imdb_train_path}")
        print(f"  Saved → {imdb_test_path}")
    else:
        print(f"\n[SKIP] IMDB not found at {IMDB_RAW_PATH}")

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    run_prepare()
