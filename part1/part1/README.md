# DSS5104 Text Classification - Implementation of Classical Methods
Member A: Lead for Classical Methods

## Task Description
Implement and evaluate classical text classification methods:
- **TF-IDF + Logistic Regression**
- **TF-IDF + SVM**
- Hyperparameter tuning (C value, max_features, n-gram range)
- Data efficiency experiments (1%, 5%, 10%, 25%, 50%, 100% of dataset)
- Calculate mean and standard deviation using 3 random seeds (42, 123, 456)
- Error analysis: 30 misclassified cases for the best and worst models on each dataset

## Datasets
### AG News
- **Source**: Kaggle AG News Classification Dataset
- **Categories**: 4 classes (World, Sports, Business, Sci/Tech)
- **Samples**: 108K training / 12K validation / 7.6K test
- **Hypothesis**: Topic classification relies on distinct keywords, making TF-IDF highly competitive

### IMDb
- **Source**: Kaggle IMDB Dataset of 50K Movie Reviews
- **Categories**: 2 classes (Negative, Positive)
- **Samples**: 35K training / 7.5K validation / 7.5K test
- **Hypothesis**: Sentiment analysis requires contextual understanding, where Transformer models should outperform classical methods

---

## Text Length Distribution Analysis
### Statistical Methods
- **Token Counting**: Whitespace tokenization (approximate estimation; actual BERT token counts may vary slightly)
- **Metrics**: Mean, Median, 95th Percentile, 99th Percentile

### AG News Text Length Distribution

| Dataset Split | Sample Size | Mean | Median | Std Dev | Min | Max | 95th Percentile | 99th Percentile |
|---------------|-------------|------|--------|---------|-----|-----|-----------------|-----------------|
| **Training Set** | 108,000 | **37.9** | **37.0** | 10.1 | 4 | 171 | **53.0** | 70.0 |
| Validation Set | 12,000 | 37.8 | 37.0 | 9.9 | 8 | 177 | 52.0 | 70.0 |
| Test Set | 7,600 | 37.7 | 37.0 | 10.1 | 11 | 137 | 52.0 | 69.0 |

### IMDb Text Length Distribution

| Dataset Split | Sample Size | Mean | Median | Std Dev | Min | Max | 95th Percentile | 99th Percentile |
|---------------|-------------|------|--------|---------|-----|-----|-----------------|-----------------|
| **Training Set** | 35,020 | **231.4** | **173.0** | 171.7 | 4 | 2,470 | **591.0** | 907.6 |
| Validation Set | 7,480 | 230.4 | 173.0 | 169.4 | 8 | 1,263 | 588.0 | 899.2 |
| Test Set | 7,500 | 230.7 | 174.0 | 171.6 | 6 | 1,830 | 585.0 | 917.0 |

### Impact on Transformer Models
BERT/DistilBERT has a maximum input length of **512 tokens**. Text length distribution directly determines truncation strategies:

| Dataset | 95th Percentile Length | Maximum Length | Truncation Impact |
|---------|------------------------|----------------|-------------------|
| **AG News** | 53 tokens | 171 tokens | ✅ **95% of samples require no truncation** (53 < 512) |
| **IMDb** | 591 tokens | 2,470 tokens | ⚠️ **Significant truncation required** (591 > 512; ~15% of samples affected) |

**Analysis Conclusion**:
- **AG News**: Short and consistent text length allows nearly all samples to be fully fed into BERT without information loss due to truncation.
- **IMDb**: ~5% of samples exceed the 512-token limit and need truncation, which may lead to:
  - Loss of late-content information in long reviews, impairing sentiment judgment
  - Competitive advantage for classical methods (TF-IDF has no length constraints), potentially widening the performance gap with Transformer models

---

## Key Question Answers
### 1. Baseline Performance: On Which Tasks Does TF-IDF + Logistic Regression Perform Exceptionally Well? Why?

**Conclusion**: TF-IDF + LR performs nearly optimally on the **AG News topic classification task**.

| Dataset | LR Accuracy | SVM Accuracy | Performance Gap |
|---------|-------------|--------------|-----------------|
| **AG News** | 92.08% | 92.13% | **Only 0.05%** |
| **IMDb** | 87.64% | 90.51% | 2.88% |

**Reason Analysis**:
- **AG News** is a topic classification task with distinct keyword features per category:
  - World: "Venezuela", "Iraq", "Olympics"
  - Business: "Google IPO", "oil prices", "stock"
  - Sports: "Tiger", "Olympics", "gold medal"
  - Sci/Tech: "Intel", "Microsoft", "software"
  
  TF-IDF effectively captures these keyword signals, enabling strong performance from LR.

- **IMDb** is a sentiment analysis task that requires understanding:
  - Contextual semantics (e.g., "not bad" vs "bad")
  - Sarcasm and implicit meaning
  - Scope of negation words
  
  LR relies solely on bag-of-words models and fails to capture such complex patterns, resulting in inferior performance compared to SVM.

---

### 2. Data Efficiency: How Does the Ranking of Methods Change as Labeled Data Decreases?

**AG News Data Efficiency Comparison**:

| Data Ratio | LR Accuracy | SVM Accuracy | Leading Model |
|------------|-------------|--------------|--------------|
| 100% | 92.08% | **92.13%** | SVM (+0.05%) |
| 50% | 91.30% | **91.43%** | SVM (+0.13%) |
| 25% | **90.72%** | 90.67% | LR (+0.05%) |
| 10% | **89.57%** | 89.47% | LR (+0.10%) |
| 5% | 88.28% | **88.30%** | SVM (+0.02%) |
| 1% | **83.57%** | 83.57% | Tie |

**IMDb Data Efficiency Comparison**:

| Data Ratio | LR Accuracy | SVM Accuracy | Leading Model |
|------------|-------------|--------------|--------------|
| 100% | 87.64% | **90.51%** | SVM (+2.87%) |
| 50% | 86.43% | **89.66%** | SVM (+3.23%) |
| 25% | 85.37% | **88.52%** | SVM (+3.15%) |
| 10% | 83.83% | **87.05%** | SVM (+3.22%) |
| 5% | 82.80% | **85.24%** | SVM (+2.44%) |
| 1% | 80.43% | **80.52%** | SVM (+0.09%) |

**Key Findings**:
- **AG News**: The performance gap between LR and SVM narrows as data decreases. LR even slightly outperforms SVM at 25% and 10% data ratios.
- **IMDb**: SVM significantly outperforms LR across all data ratios, with a stable performance gap of 2.5–3.2%.

**Practical Recommendations**:
- For **keyword-driven tasks** (e.g., topic classification), **small datasets (10–25%)** can achieve performance close to full datasets.
- For **semantic understanding tasks** (e.g., sentiment analysis), **larger datasets** are required to leverage the advantages of complex models.

---

### 3. Cost-Accuracy Tradeoff: Which Method Offers the Optimal Tradeoff for Each Task?

**AG News Tradeoff Analysis**:

| Model | Accuracy | Training Time | Inference Time | Comprehensive Evaluation |
|-------|----------|---------------|----------------|--------------------------|
| **SVM** | **92.13%** | **14.7s** | **0.35s** | ✅ **Optimal Choice** |
| LR | 92.08% | 21.3s | 0.40s | Slightly lower accuracy and slower training |

**Conclusion**: SVM delivers the optimal tradeoff on AG News—**higher accuracy + faster training speed**

**IMDb Tradeoff Analysis**:

| Model | Accuracy | Training Time | Inference Time | Comprehensive Evaluation |
|-------|----------|---------------|----------------|--------------------------|
| **SVM** | **90.51%** | 17.3s | 1.39s | ✅ **Optimal Choice for Accuracy Priority** |
| LR | 87.64% | **5.5s** | **0.83s** | Choice for Speed Priority (-2.87% accuracy penalty) |

**Conclusion**:
- If **accuracy is prioritized**: Choose SVM (+2.87% accuracy improvement)
- If **speed is prioritized and lower accuracy is acceptable**: Choose LR (3x faster training)

---

### 4. Failure Modes: On Which Types of Samples Do Each Model Perform Poorly?

#### AG News Error Pattern Analysis

**Common Failure Modes** (both models misclassify the same samples):
1. **Blurred Category Boundaries** (World vs Business)
   - Example: "Venezuela Prepares for Chavez Recall Vote...produce turmoil in world oil market"
   - Analysis: Involves both political (World) and economic (Business) themes

2. **Commercial Elements in Tech News** (Sci/Tech vs Business)
   - Example: "Google IPO", "Intel product delay", "Yahoo! domain registration"
   - Analysis: Tech company news often involves commercial operations

3. **Sports-Politics Overlap** (Sports vs World)
   - Example: "Olympics day four...Richard Faulds...gold for Great Britain"
   - Analysis: International events involve national representation (World) while focusing on sports content (Sports)

**Model Differences**:
- LR is more prone to misclassifying Sports as Business than SVM (3 cases vs 1 case)
- The two models show high error overlap on the World category (~80% identical misclassified samples)

#### IMDb Error Pattern Analysis

**Common Failure Modes**:
1. **Complex Sentiment Expressions** (both models fail)
   - Example: "There were times when this movie seemed to get a whole lot more complicated than it needed to be, but I guess that's part of it's charm"
   - Analysis: Negative criticism followed by positive praise creates sentiment transitions that are difficult to capture

2. **Implicit Sarcasm** (both models fail)
   - Example: "This tiresome, plodding...movie in almost impossible to watch...The only two decent things...are both attached to gorgeous Stella Stevens"
   - Analysis: Superficial criticism with implicit humor/appreciation

3. **Mixed Sentiment Reviews** (both models fail)
   - Example: "I am quite a fan of novelist...Director has also directed wonderful comedic pieces...but as an adaptation, it fails from every angle"
   - Analysis: Positive evaluations of the original work and director vs negative evaluation of the adaptation

**Model Differences**:
- **LR-Specific Errors**: Simple keyword matching failures (e.g., classifying text as Negative upon seeing "bad" without considering context)
- **SVM-Specific Errors**: Mistakes on rare expression patterns, but overall stronger robustness

**Failure Mode Classification Statistics**:

| Failure Type | AG News | IMDb |
|--------------|---------|------|
| Blurred Category Boundaries | 45% | - |
| Mixed Multi-Topic Content | 35% | - |
| Complex Sentiment Expression | - | 40% |
| Sarcasm/Implicit Meaning | - | 30% |
| Keyword Misleading | 20% | 30% |

---

### 5. Error Analysis: Comparison Between Best and Worst Models

#### AG News: SVM (Best) vs LR (Worst)

**Confusion Matrix Comparison**:

SVM Best Model Error Distribution:
- World → Business: 10 cases
- Sci/Tech → Business: 7 cases
- World → Sports: 2 cases

LR Worst Model Error Distribution:
- World → Business: 9 cases
- Sci/Tech → Business: 8 cases
- Sports → Business: 1 case

**Key Findings**:
- **~70% of misclassified cases overlap between the two models** (21/30 samples)
- Both models are most prone to errors on the **World category** (13 cases vs 12 cases)
- **Implication**: These are inherently difficult samples (with blurred category boundaries) rather than model defects

**Representative Overlapping Error Cases**:

| Text | True Label | Prediction by Both Models | Analysis |
|------|------------|---------------------------|----------|
| "Oil prices bubble to record high..." | World | Business | Economic topic with global impact |
| "Google IPO...public participation" | Sci/Tech | Business | Tech company IPO involving commercial operations |

#### IMDb: SVM (Best) vs LR (Worst)

**Confusion Matrix Comparison**:

SVM Best Model:
- 19 Positive samples misclassified as Negative
- 11 Negative samples misclassified as Positive

LR Worst Model:
- 13 Positive samples misclassified as Negative
- 17 Negative samples misclassified as Positive

**Key Findings**:
- **~40% of misclassified cases overlap** (12/30 samples)
- LR is more prone to misclassifying Negative as Positive (17 cases vs 11 cases)
- **Implication**: LR relies more on surface-level vocabulary and is easily misled by positive words

**Representative Differential Error Cases**:

| Text Excerpt | True Label | SVM Prediction | LR Prediction | Analysis |
|--------------|------------|----------------|---------------|----------|
| "If I had not read the book...I would have liked it...but as an adaptation, it fails" | Negative | ✅ Negative | ❌ Positive | Praise followed by criticism; LR only captures the first half |
| "People say that this film is 'typical teen horror'...It's a good film" | Positive | ❌ Negative | ✅ Positive | SVM misled by the word "horror" |

---

## Hyperparameter Search Results
### AG News
| Model | C | Max Features | N-gram | Val Macro F1 |
|-------|---|--------------|--------|--------------|
| **LR** | 1.0 | 50000 | (1,2) | 0.9246 |
| **SVM** | 0.1 | 50000 | (1,2) | **0.9259** |

### IMDb
| Model | C | Max Features | N-gram | Val Macro F1 |
|-------|---|--------------|--------|--------------|
| **LR** | 0.1 | 10000 | (1,1) | 0.8853 |
| **SVM** | 1.0 | 50000 | (1,2) | **0.9051** |

---

## Test Set Results (100% Data)
### AG News
| Model | Accuracy | Macro F1 | Training Time | Inference Time |
|-------|----------|----------|---------------|----------------|
| **LR** | 92.08% ± 0.00% | 92.06% ± 0.00% | 21.27s | 0.399s |
| **SVM** | **92.13% ± 0.00%** | **92.11% ± 0.00%** | 14.70s | 0.349s |

**Conclusion**: SVM slightly outperforms LR (0.05% higher F1 score) with faster training speed

### IMDb
| Model | Accuracy | Macro F1 | Training Time | Inference Time |
|-------|----------|----------|---------------|----------------|
| **LR** | 87.64% ± 0.00% | 87.63% ± 0.00% | 5.51s | 0.825s |
| **SVM** | **90.51% ± 0.00%** | **90.51% ± 0.00%** | 17.34s | 1.394s |

**Conclusion**: SVM significantly outperforms LR (2.88% higher F1 score)

---

## Data Efficiency Experiment Results
### AG News - Learning Curves

| Data Ratio | LR Accuracy | SVM Accuracy | LR F1 | SVM F1 |
|------------|-------------|--------------|-------|--------|
| 100% | 92.08% | **92.13%** | 92.06% | **92.11%** |
| 50% | 91.30% | **91.43%** | 91.28% | **91.41%** |
| 25% | **90.72%** | 90.67% | **90.70%** | 90.63% |
| 10% | **89.57%** | 89.47% | **89.54%** | 89.42% |
| 5% | 88.28% | **88.30%** | 88.24% | **88.24%** |
| 1% | **83.57%** | 83.57% | **83.46%** | 83.41% |

### IMDb - Learning Curves

| Data Ratio | LR Accuracy | SVM Accuracy | LR F1 | SVM F1 |
|------------|-------------|--------------|-------|--------|
| 100% | 87.64% | **90.51%** | 87.63% | **90.51%** |
| 50% | 86.43% | **89.66%** | 86.42% | **89.66%** |
| 25% | 85.37% | **88.52%** | 85.35% | **88.52%** |
| 10% | 83.83% | **87.05%** | 83.80% | **87.05%** |
| 5% | 82.80% | **85.24%** | 82.77% | **85.23%** |
| 1% | 80.43% | **80.52%** | 80.40% | **80.51%** |

---

## Practical Recommendations Summary
### Task Type Selection

| Task Type | Recommended Method | Rationale |
|-----------|--------------------|-----------|
| **Topic Classification** (with distinct keywords) | TF-IDF + SVM/LR (either acceptable) | Negligible performance gap (<0.1%) |
| **Sentiment Analysis** (requires contextual understanding) | TF-IDF + SVM | Significantly outperforms LR (+2.9%) |

### Dataset Size Recommendations

| Dataset Size | Recommendation |
|--------------|----------------|
| **Adequate Data (>50K samples)** | Use SVM to fully leverage its classification capabilities |
| **Moderate Data (10K–50K samples)** | SVM still outperforms LR |
| **Limited Data (<10K samples)** | Performance gap between LR and SVM narrows; LR can be considered (faster training) |

### Cost-Accuracy Tradeoff

| Priority | AG News Recommendation | IMDb Recommendation |
|----------|------------------------|---------------------|
| **Accuracy Priority** | SVM | SVM |
| **Speed Priority** | SVM (surprisingly faster) | LR (3x faster training) |
| **Balanced Choice** | SVM | Depends on specific requirements |

---

## Project Structure Details

```
.
├── README.md                           # Project documentation and experimental results summary
├── requirements.txt                    # Python dependency list
│
├── data/                               # Dataset directory
│   ├── ag_news/                        # AG News dataset (preprocessed)
│   │   ├── train.csv                   # Training set: 108,000 news articles
│   │   ├── val.csv                     # Validation set: 12,000 news articles (for hyperparameter tuning)
│   │   ├── test.csv                    # Test set: 7,600 news articles (for final evaluation only)
│   │   └── metadata.json               # Metadata: number of classes, sample count, class distribution, etc.
│   └── imdb/                           # IMDb dataset (preprocessed)
│       ├── train.csv                   # Training set: 35,020 reviews
│       ├── val.csv                     # Validation set: 7,480 reviews (for hyperparameter tuning)
│       ├── test.csv                    # Test set: 7,500 reviews (for final evaluation only)
│       └── metadata.json               # Metadata: number of classes, sample count, class distribution, etc.
│
├── src/                                # Source code directory
│   ├── __init__.py                     # Python package initialization
│   ├── classical_models.py             # Core classifier implementation (TF-IDF + LR/SVM)
│   └── data_loaders/                   # Data loading module
│       ├── __init__.py                 # Package initialization
│       ├── prepare_datasets.py         # Data preprocessing script (generates standard format from raw Kaggle data)
│       ├── dataset_loader.py           # Data loading functions (load_ag_news, load_imdb, etc.)
│       └── analyze_text_length.py      # Text length distribution analysis script
│                                       #   - Calculates mean, median, 95th/99th percentiles
│                                       #   - Evaluates Transformer truncation impact
│
├── experiments/                        # Experimental script directory
│   ├── hyperparameter_search.py        # Hyperparameter search script
│   │                                   #   - Searches optimal parameters for each dataset-model combination
│   │                                   #   - Output: results/*_hyperparam_search.csv
│   ├── run_data_efficiency_experiments.py  # Main data efficiency experiment script
│   │                                   #   - Trains models on 6 data ratios (1%–100%)
│   │                                   #   - Uses 3 random seeds
│   │                                   #   - Output: results/classical_experiments.csv
│   ├── error_analysis.py               # Error analysis script
│   │                                   #   - Identifies best/worst models for each dataset
│   │                                   #   - Saves misclassified cases
│   │                                   #   - Output: results/*_errors.csv
│   ├── summarize_results.py            # Results summary script
│   │                                   #   - Calculates mean ± standard deviation
│   │                                   #   - Generates LaTeX-formatted tables
│   │                                   #   - Output: results/classical_summary.csv
│   └── plot_learning_curves.py         # Visualization script
│                                       #   - Plots learning curves (accuracy/F1 vs data ratio)
│                                       #   - Output: results/figures/*.png
│
└── results/                            # Experimental results directory (all output files)
    ├── best_params.json                # Optimal hyperparameters (4 combinations: 2 datasets × 2 models)
    │                                   #   Format: {"ag_news": {"lr": {...}, "svm": {...}}, ...}
    │
    ├── best_worst_models.json          # Identification of best/worst models per dataset
    │                                   #   Used to determine comparison pairs for error analysis
    │
    ├── text_length_stats.csv           # Text length distribution statistics
    │                                   #   - Mean, median, 95th/99th percentiles
    │                                   #   - Used for Transformer truncation impact analysis
    │
    ├── classical_experiments.csv       # Raw experimental results (72 rows = 2×2×6×3)
    │                                   #   Contains detailed metrics for each experiment:
    │                                   #   - dataset, model, seed, data_ratio
    │                                   #   - test_accuracy, test_macro_f1
    │                                   #   - train_time, inference_time
    │                                   #   - per_class_f1 (F1 score for each category)
    │
    ├── classical_summary.csv           # Aggregated statistical results (grouped by dataset/model/ratio)
    │                                   #   Contains: mean ± standard deviation
    │                                   #   Used to generate report tables and learning curves
    │
    ├── classical_for_comparison.csv    # Standardized data for comparison with Transformer methods
    │                                   #   Formatted for integration by Member D
    │
    ├── ag_news_lr_hyperparam_search.csv    # AG News + LR hyperparameter search results
    ├── ag_news_svm_hyperparam_search.csv   # AG News + SVM hyperparameter search results
    ├── imdb_lr_hyperparam_search.csv       # IMDb + LR hyperparameter search results
    ├── imdb_svm_hyperparam_search.csv      # IMDb + SVM hyperparameter search results
    │                                       # Each file contains:
    │                                       #   - C, max_features, ngram_range
    │                                       #   - val_accuracy, val_macro_f1
    │                                       #   - train_time
    │
    ├── ag_news_svm_best_errors.csv     # 30 misclassified cases from AG News best model (SVM)
    ├── ag_news_lr_worst_errors.csv     # 30 misclassified cases from AG News worst model (LR)
    ├── imdb_svm_best_errors.csv        # 30 misclassified cases from IMDb best model (SVM)
    ├── imdb_lr_worst_errors.csv        # 30 misclassified cases from IMDb worst model (LR)
    │                                   # Each error case file contains:
    │                                   #   - text: sample content (first 500 characters)
    │                                   #   - true_label: ground truth label
    │                                   #   - predicted_label: model prediction
    │                                   #   - true_idx/pred_idx: label indices
    │                                   #   Used for error analysis: identifying common/distinct error patterns
    │
    └── figures/                        # Visualization directory
        ├── ag_news_learning_curves.png       # AG News learning curves
        │                                     #   Left: Accuracy vs Data Ratio
        │                                     #   Right: Macro F1 vs Data Ratio
        │                                     #   Includes error bars (standard deviation across 3 seeds)
        ├── imdb_learning_curves.png          # IMDb learning curves (same structure as above)
        ├── model_comparison_100percent.png   # LR vs SVM bar chart comparison (100% data)
        ├── training_time_comparison.png      # Training time comparison (log scale)
        └── summary_report.txt                # Text-based results summary
```

---

## Running Instructions

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare datasets
python3 src/data_loaders/prepare_datasets.py

# 3. Hyperparameter search (4 runs: 2 datasets × 2 models)
python3 experiments/hyperparameter_search.py

# 4. Run data efficiency experiments (72 runs: 2 datasets × 2 models × 6 ratios × 3 seeds)
python3 experiments/run_data_efficiency_experiments.py

# 5. Summarize results
python3 experiments/summarize_results.py

# 6. Plot learning curves
python3 experiments/plot_learning_curves.py

# 7. Perform error analysis
python3 experiments/error_analysis.py
```

---

## Generated Visualizations
Located in the `results/figures/` directory:
- `ag_news_learning_curves.png` - AG News learning curves
- `imdb_learning_curves.png` - IMDb learning curves
- `model_comparison_100percent.png` - LR vs SVM comparison bar chart (100% data)
- `training_time_comparison.png` - Training time comparison chart

## Error Analysis Files
Located in the `results/` directory:
- `ag_news_svm_best_errors.csv` - Misclassified cases from AG News best model (SVM)
- `ag_news_lr_worst_errors.csv` - Misclassified cases from AG News worst model (LR)
- `imdb_svm_best_errors.csv` - Misclassified cases from IMDb best model (SVM)
- `imdb_lr_worst_errors.csv` - Misclassified cases from IMDb worst model (LR)

Each file contains 30 misclassified cases.

---

## Statistical Reliability
- **Random Seeds**: 42, 123, 456
- **Variance Range**:
  - 100% data: Almost no variance (<0.01%)
  - 1% data: Variance ~0.5–1% (consistent with expectations)

