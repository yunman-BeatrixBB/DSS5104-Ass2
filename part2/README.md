# Member B — Neural Network Text Classification
## DSS5104 Assignment

###  Scope
- TextCNN (multi-scale convolution + max-pooling)
- BiLSTM (bidirectional LSTM)
- Data efficiency experiments (6 ratios × 3 seeds)

### File Structure
```
member_b/
├── data_utils.py      # Data loading, text cleaning, vocabulary, subsampling
├── models.py          # FastText / TextCNN / BiLSTM model definitions
├── trainer.py         # Training loop, evaluation, error analysis
├── experiments.py     # Main experiment script
├── visualize.py       # Result visualization
├── requirements.txt
└── results/           # Auto-generated

```

### Environment Setup

```bash
pip install -r requirements.txt
```

> For Mac M3 users: PyTorch will automatically use MPS acceleration.
> For Colab users: CUDA is enabled automatically.

### Data Placement

Place the data files under the corresponding subdirectories in `member_b/data/`:

```
member_b/
└── data/
    ├── ag_news/
    │   ├── train.csv
    │   └── test.csv
    └── imdb/
        └── IMDB Dataset.csv    
```

### Run Experiments

**AG News（topic classification）：**
```bash
python experiments.py --dataset agnews
```

**IMDB（sentiment analysis）：**
```bash
python experiments.py --dataset imdb
```

**un only selected models (for debugging):**
```bash
python experiments.py --dataset agnews --models textcnn --skip_hp_search
```

### Visualization

```bash
python visualize.py --result_file results/member_b_agnews.json
python visualize.py --result_file results/member_b_imdb.json
```

### Hyperparameter Settings

| model     | lr    | epochs | batch | embed_dim | other            |
|-----------|-------|--------|-------|-----------|------------------|
| FastText  | 1e-3  | 10     | 128   | 100       | —                |
| TextCNN   | 1e-3  | 10     | 64    | 100       | filters=[2,3,4]  |
| BiLSTM    | 1e-3  | 10     | 64    | 100       | hidden=128, L=2  |

Hyperparameters are selected based on validation set macro-F1. Candidate values can be found in experiments py:hyperparameter_search()

### Random Seeds

All experiments are run with 3 seeds (42, 123, 456), and reported as mean ± std.

### Output Description

- `results/member_b_{dataset}.json` — all numerical results

- `results/member_b_{dataset}_curve.png` — data efficiency learning curve

JSON ：
```json
{
  "dataset": "agnews",
  "full_data": {
    "fasttext": {"acc_mean": ..., "acc_std": ..., "f1_mean": ..., ...},
    ...
  },
  "efficiency": {
    "fasttext": {
      "1.0": {"f1_mean": ..., "f1_std": ...},
      "0.5": {...},
      ...
    }
  }
}
```
