"""
data_utils.py — Member B: Data Loading and Preprocessing
DSS5104 Text Classification Assignment
"""

import re
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# 1. Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """General text cleaning: strip HTML tags, lowercase, remove non-alpha characters, collapse whitespace"""
    text = re.sub(r"<[^>]+>", " ", str(text))          # strip HTML
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())   # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# 2. Vocabulary
# ─────────────────────────────────────────────

class Vocabulary:
    PAD_TOKEN = "<PAD>"   # index 0
    UNK_TOKEN = "<UNK>"   # index 1

    def __init__(self, max_size: int = 30_000, min_freq: int = 2):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def build(self, texts: List[str]) -> None:
        counter: Counter = Counter()
        for text in texts:
            counter.update(text.split())

        # Keep high-frequency words, truncate to max_size
        vocab_words = [
            w for w, c in counter.most_common(self.max_size)
            if c >= self.min_freq
        ]
        for word in vocab_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"[Vocab] size = {len(self.word2idx):,}  "
              f"(max_size={self.max_size}, min_freq={self.min_freq})")

    def encode(self, text: str, max_len: int = 256) -> List[int]:
        tokens = text.split()[:max_len]
        indices = [self.word2idx.get(t, 1) for t in tokens]
        # Pad on the right
        indices += [0] * (max_len - len(indices))
        return indices

    def __len__(self) -> int:
        return len(self.word2idx)


# ─────────────────────────────────────────────
# 3. Data Loading
# ─────────────────────────────────────────────

def load_agnews(
    train_path: str,
    test_path: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load AG News dataset.
    Column order: Class Index (1-4), Title, Description
    Returns: (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    Labels converted to 0-indexed (0,1,2,3)
    """
    train_df = pd.read_csv(train_path, header=0)
    test_df  = pd.read_csv(test_path,  header=0)

    for df in [train_df, test_df]:
        df.columns = ["label", "title", "description"]
        df["text"]  = df["title"].fillna("") + " " + df["description"].fillna("")
        df["text"]  = df["text"].apply(clean_text)
        df["label"] = df["label"] - 1   # convert to 0-indexed

    # Split validation set from training set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=val_ratio,
        stratify=train_df["label"].tolist(),
        random_state=seed,
    )

    test_texts  = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()

    _print_split_info("AG News", train_labels, val_labels, test_labels)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def load_imdb(
    path: str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load IMDB dataset (50K).
    Columns: review, sentiment (positive/negative)
    Returns: (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """
    df = pd.read_csv(path)
    df["text"]  = df["review"].apply(clean_text)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    texts  = df["text"].tolist()
    labels = df["label"].tolist()

    # Split test set first
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_ratio, stratify=labels, random_state=seed
    )
    # Then split validation set
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_ratio / (1 - test_ratio),
        stratify=train_val_labels,
        random_state=seed,
    )

    _print_split_info("IMDB", train_labels, val_labels, test_labels)
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels


def _print_split_info(name, train_labels, val_labels, test_labels):
    print(f"\n[{name}] train={len(train_labels):,}  "
          f"val={len(val_labels):,}  test={len(test_labels):,}")
    classes = sorted(set(train_labels))
    c = Counter(train_labels)
    print(f"  class distribution (train): "
          + "  ".join(f"cls{k}={c[k]}" for k in classes))


# ─────────────────────────────────────────────
# 4. Stratified Subsampling (for data efficiency experiments)
# ─────────────────────────────────────────────

def stratified_subsample(
    texts: List[str],
    labels: List[int],
    fraction: float,
    seed: int = 42,
) -> Tuple[List[str], List[int]]:
    """Stratified subsample of training set by fraction"""
    if fraction >= 1.0:
        return texts, labels

    n = max(1, int(len(texts) * fraction))
    n = max(n, len(set(labels)))   # ensure at least 1 sample per class

    idx, _ = train_test_split(
        range(len(texts)),
        train_size=n,
        stratify=labels,
        random_state=seed,
    )
    return [texts[i] for i in idx], [labels[i] for i in idx]


# ─────────────────────────────────────────────
# 5. Text Length Statistics (for report)
# ─────────────────────────────────────────────

def text_length_stats(texts: List[str], name: str = "") -> None:
    lengths = [len(t.split()) for t in texts]
    arr = np.array(lengths)
    print(f"\n[Length stats] {name}")
    print(f"  mean={arr.mean():.1f}  median={np.median(arr):.1f}  "
          f"95th={np.percentile(arr, 95):.1f}  max={arr.max()}")
