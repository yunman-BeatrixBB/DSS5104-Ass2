"""
trainer.py — Member B: Training Framework
DSS5104 Text Classification Assignment
"""

import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)

from data_utils import Vocabulary


# ─────────────────────────────────────────────
# 1. PyTorch Dataset
# ─────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab: Vocabulary,
        max_len: int = 256,
    ):
        self.encodings = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.long),
        )


def make_loader(
    texts: List[str],
    labels: List[int],
    vocab: Vocabulary,
    batch_size: int = 64,
    shuffle: bool = False,
    max_len: int = 256,
) -> DataLoader:
    ds = TextDataset(texts, labels, vocab, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=False)


# ─────────────────────────────────────────────
# 2. Single Epoch Training
# ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        # Gradient clipping: prevent exploding gradients in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    """Returns (accuracy, macro_f1, y_true, y_pred)"""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(y.tolist())

    acc = accuracy_score(targets, preds)
    f1  = f1_score(targets, preds, average="macro", zero_division=0)
    return acc, f1, targets, preds


# ─────────────────────────────────────────────
# 4. Full Training Loop (with early stopping)
# ─────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int = 10,
    patience: int = 3,
    verbose: bool = True,
) -> Tuple[nn.Module, float]:
    """
    Train model with early stopping on val macro-F1.
    Returns: (best_model, train_seconds)
    """
    best_val_f1 = -1.0
    best_state  = None
    patience_cnt = 0
    t0 = time.time()

    for epoch in range(1, num_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        if verbose:
            print(f"  Epoch {epoch:2d}/{num_epochs} | "
                  f"loss={loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                if verbose:
                    print(f"  Early stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, time.time() - t0


# ─────────────────────────────────────────────
# 5. Error Analysis
# ─────────────────────────────────────────────

def error_analysis(
    texts: List[str],
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    n_examples: int = 25,
) -> None:
    """
    Print n_examples misclassified samples and output classification report.
    """
    errors = [
        (texts[i], y_true[i], y_pred[i])
        for i in range(len(y_true))
        if y_true[i] != y_pred[i]
    ]
    print(f"\n[Error Analysis] Total errors: {len(errors)} / {len(y_true)}")

    if label_names is None:
        label_names = [str(c) for c in sorted(set(y_true))]

    print(f"\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))

    print(f"\n--- {n_examples} Misclassified Examples ---")
    for i, (text, true, pred) in enumerate(errors[:n_examples]):
        # Truncate display
        snippet = text[:120] + ("..." if len(text) > 120 else "")
        true_name = label_names[true] if true < len(label_names) else str(true)
        pred_name = label_names[pred] if pred < len(label_names) else str(pred)
        print(f"  [{i+1:2d}] TRUE={true_name:<12} PRED={pred_name:<12} TEXT: {snippet}")


# ─────────────────────────────────────────────
# 6. Single Seed Run
# ─────────────────────────────────────────────

def run_single_seed(
    model: nn.Module,
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    test_texts: List[str],
    test_labels: List[int],
    vocab: Vocabulary,
    device: torch.device,
    lr: float = 1e-3,
    num_epochs: int = 10,
    batch_size: int = 64,
    max_len: int = 256,
    verbose: bool = True,
) -> Dict:
    """
    Train and evaluate for one seed. Returns result dictionary.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = make_loader(train_texts, train_labels, vocab, batch_size, shuffle=True,  max_len=max_len)
    val_loader   = make_loader(val_texts,   val_labels,   vocab, batch_size, shuffle=False, max_len=max_len)
    test_loader  = make_loader(test_texts,  test_labels,  vocab, batch_size, shuffle=False, max_len=max_len)

    model, train_time = train_model(
        model, train_loader, val_loader,
        optimizer, criterion, device,
        num_epochs=num_epochs, verbose=verbose,
    )

    # Inference time
    t_infer = time.time()
    test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, device)
    infer_time = time.time() - t_infer

    return {
        "accuracy":   test_acc,
        "macro_f1":   test_f1,
        "train_time": train_time,
        "infer_time": infer_time,
        "y_true":     y_true,
        "y_pred":     y_pred,
        "model":      model,
    }
