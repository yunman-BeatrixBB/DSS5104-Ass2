"""
models.py — Member B: Neural Network Model Definitions
DSS5104 Text Classification Assignment

Three models:
  1. FastTextClassifier  — averaged word embeddings + linear layer
  2. TextCNN             — multi-scale convolutional filters + max-over-time pooling
  3. BiLSTMClassifier    — bidirectional LSTM + final hidden state concatenation

All models share the same interface:
  forward(x: LongTensor[B, L]) -> logits: FloatTensor[B, C]
"""

from typing import List
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# 1. FastText
# ─────────────────────────────────────────────

class FastTextClassifier(nn.Module):
    """
    Simplified implementation of FastText (Joulin et al., 2016).

    Mechanism:
      - Map each word to a D-dimensional embedding
      - Average all word embeddings in the sequence (EmbeddingBag mode='mean')
      - A linear layer maps to the number of classes

    Pros: extremely fast training, low memory, strong baseline.
    Cons: completely ignores word order.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        # EmbeddingBag automatically averages embeddings within a bag
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode="mean", padding_idx=padding_idx
        )
        self.fc = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)  —  B=batch_size, L=seq_len
        embedded = self.embedding(x)    # (B, D)
        return self.fc(embedded)        # (B, C)


# ─────────────────────────────────────────────
# 2. TextCNN
# ─────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    TextCNN implementation (Kim, 2014).

    Mechanism:
      - Word embeddings -> (B, L, D), transposed to (B, D, L) for Conv1d
      - Multiple 1D convolutions with different kernel sizes, each capturing n-gram features
      - Max-over-time pooling after each convolution -> scalar feature
      - Concatenate features from all kernels -> Dropout -> FC classifier

    Why multiple kernel sizes:
      - kernel=2 captures bigrams
      - kernel=3 captures trigrams
      - kernel=4 captures 4-grams
      Combined, they cover local features at different granularities
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        filter_sizes: List[int] = None,
        num_filters: int = 128,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()
        if filter_sizes is None:
            filter_sizes = [2, 3, 4]

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # One Conv1d per kernel size
        # in_channels=embed_dim (treat embedding dim as channels)
        # out_channels=num_filters
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self._init_weights()

    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        embedded = self.embedding(x)               # (B, L, D)
        embedded = embedded.permute(0, 2, 1)       # (B, D, L)  Conv1d expects channels in dim 1

        pooled_features = []
        for conv in self.convs:
            # conv: (B, D, L) -> (B, num_filters, L - fs + 1)
            activated = torch.relu(conv(embedded))
            # max-over-time pooling: take max over the time dimension
            pooled = activated.max(dim=-1).values  # (B, num_filters)
            pooled_features.append(pooled)

        # Concatenate features from all kernels
        cat = torch.cat(pooled_features, dim=1)    # (B, num_filters * len(filter_sizes))
        out = self.dropout(cat)
        return self.fc(out)                        # (B, C)


# ─────────────────────────────────────────────
# 3. BiLSTM
# ─────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM text classifier.

    Mechanism:
      - Word embeddings -> (B, L, D)
      - Fed into bidirectional LSTM: forward pass L->R, backward pass R->L
      - Concatenate last forward hidden state + last backward hidden state -> (B, 2*H)
      - Dropout -> FC classifier

    Why bidirectional:
      - Unidirectional LSTM can only see previous words
      - Bidirectional allows each step to attend to both past and future context

    When num_layers > 1, inter-layer dropout is applied (built into nn.LSTM).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.input_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            # PyTorch LSTM only applies dropout between layers when num_layers > 1
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout)
        # bidirectional -> hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L)
        embedded = self.input_dropout(self.embedding(x))  # (B, L, D)

        # hidden: (num_layers * 2, B, H)  — *2 because bidirectional
        _, (hidden, _) = self.lstm(embedded)

        # Take the last layer's forward and backward hidden states
        # hidden[-2]: last layer forward  (B, H)
        # hidden[-1]: last layer backward (B, H)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, 2H)

        out = self.output_dropout(last_hidden)
        return self.fc(out)                                        # (B, C)


# ─────────────────────────────────────────────
# 4. Model Factory (for convenient use in experiment loops)
# ─────────────────────────────────────────────

def build_model(
    model_name: str,
    vocab_size: int,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    model_name: 'fasttext' | 'textcnn' | 'bilstm'
    kwargs: model-specific hyperparameters
    """
    name = model_name.lower()
    if name == "fasttext":
        return FastTextClassifier(
            vocab_size=vocab_size,
            embed_dim=kwargs.get("embed_dim", 100),
            num_classes=num_classes,
        )
    elif name == "textcnn":
        return TextCNN(
            vocab_size=vocab_size,
            embed_dim=kwargs.get("embed_dim", 100),
            num_classes=num_classes,
            filter_sizes=kwargs.get("filter_sizes", [2, 3, 4]),
            num_filters=kwargs.get("num_filters", 128),
            dropout=kwargs.get("dropout", 0.5),
        )
    elif name == "bilstm":
        return BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=kwargs.get("embed_dim", 100),
            num_classes=num_classes,
            hidden_dim=kwargs.get("hidden_dim", 128),
            num_layers=kwargs.get("num_layers", 2),
            dropout=kwargs.get("dropout", 0.3),
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
