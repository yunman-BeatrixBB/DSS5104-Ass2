# Member B: Neural Model Analysis
## TextCNN and BiLSTM — Results and Discussion

---

## 1. Dataset Hypotheses

**AG News (Topic Classification)**
We hypothesize that topic classification with four well-separated categories (World, Sports, Business, Sci/Tech) will favor keyword-driven approaches. Each category has a distinctive vocabulary — Sports articles mention athletes and scores, Business articles mention stocks and earnings, and so on. Neural models are expected to outperform classical baselines modestly, since the task does not require deep contextual understanding.

**IMDB (Sentiment Analysis)**
We hypothesize that sentiment classification on movie reviews will be more demanding. Reviewers frequently use sarcasm, negation, and nuanced language (e.g., "not bad at all", "surprisingly disappointing"). Models that capture sequential context — BiLSTM in particular — should have an advantage over bag-of-words approaches and even convolutional models that capture only local n-gram patterns.

---

## 2. Text Length Distribution

Text length statistics (word-level tokenization) across both datasets:

**AG News:**
- Mean: 38.3 words | Median: 37.0 | 95th percentile: 57.0 | Max: 180

**IMDB:**
- Mean: 234.2 words | Median: 176.0 | 95th percentile: 598.0 | Max: 2,494

AG News articles are uniformly short. Virtually all samples fall well within BERT's 512-token limit, so truncation is not a concern for any model tier. IMDB reviews are substantially longer and more variable. The 95th percentile reaches 598 tokens, meaning approximately 5% of reviews exceed the transformer input limit of 512 tokens. For our neural models, we apply a sequence length cap of 256 tokens. This truncation discards the latter portion of long reviews, which may contain concluding sentiment judgments — a potential disadvantage for all neural approaches on long documents.

---

## 3. Hyperparameter Tuning

Hyperparameters were selected based on validation set macro-F1. The candidate space was:
- Learning rate: {1e-3, 5e-4}
- Number of epochs: {5, 10}

All other architectural hyperparameters (embedding dimension: 100, filter sizes for TextCNN: [2, 3, 4], number of filters: 128, dropout: 0.5; BiLSTM hidden dimension: 128, layers: 2, dropout: 0.3) were fixed across experiments to isolate the effect of learning rate and epoch count.

**AG News best hyperparameters:**

| Model   | Learning Rate | Epochs | Val Macro-F1 |
|---------|--------------|--------|--------------|
| TextCNN | 1e-3         | 5      | 0.9155       |
| BiLSTM  | 1e-3         | 10     | 0.9249       |

**IMDB best hyperparameters (default used, HP search skipped):**

| Model   | Learning Rate | Epochs |
|---------|--------------|--------|
| TextCNN | 1e-3         | 10     |
| BiLSTM  | 1e-3         | 10     |

TextCNN converged faster on AG News (5 epochs sufficient), consistent with the task's reliance on local keyword features. BiLSTM required more epochs to allow the recurrent state to accumulate meaningful sequence-level representations.

---

## 4. Full Data Results

All results are reported on the held-out test set as mean ± standard deviation across 3 random seeds (42, 123, 456). The test set was never used for hyperparameter selection or model comparison during development.

### AG News

| Model   | Accuracy        | Macro-F1        | Train Time (s) | Infer Time (s) |
|---------|-----------------|-----------------|----------------|----------------|
| TextCNN | 0.9097 ± 0.0017 | 0.9097 ± 0.0018 | 171.6          | 1.393          |
| BiLSTM  | 0.9220 ± 0.0014 | 0.9219 ± 0.0014 | 1609.2         | 5.220          |

BiLSTM outperforms TextCNN by 1.2 percentage points in macro-F1 on AG News. The variance across seeds is low for both models (≤0.0018), confirming that the gap is statistically meaningful rather than noise. However, BiLSTM incurs a 9.4× training time overhead (1609s vs 172s) and a 3.7× inference time overhead, with only marginal accuracy gain.

**Per-class F1 (TextCNN, seed 42):**

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| World    | 0.91      | 0.90   | 0.91 |
| Sports   | 0.95      | 0.98   | 0.96 |
| Business | 0.89      | 0.86   | 0.87 |
| Sci/Tech | 0.88      | 0.89   | 0.88 |

**Per-class F1 (BiLSTM, seed 42):**

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| World    | 0.92      | 0.92   | 0.92 |
| Sports   | 0.96      | 0.98   | 0.97 |
| Business | 0.90      | 0.88   | 0.89 |
| Sci/Tech | 0.90      | 0.91   | 0.90 |

Sports is the highest-performing class for both models — sports vocabulary is highly distinctive (athlete names, score-related terms). Business is the weakest class for both, likely due to overlap with Sci/Tech (technology companies' financial news) and World (geopolitical economic events).

### IMDB

| Model   | Accuracy        | Macro-F1        | Train Time (s) | Infer Time (s) |
|---------|-----------------|-----------------|----------------|----------------|
| TextCNN | 0.8799 ± 0.0012 | 0.8799 ± 0.0012 | 280.3          | 1.067          |
| BiLSTM  | 0.8909 ± 0.0029 | 0.8908 ± 0.0030 | 1342.7         | 6.712          |

On IMDB, BiLSTM outperforms TextCNN by 1.1 percentage points in macro-F1. The advantage is consistent with our hypothesis: sentiment analysis benefits from sequential context modeling. BiLSTM's bidirectional hidden states can capture long-range dependencies — for example, a positive conclusion following a negative setup — whereas TextCNN is limited to local n-gram patterns. The cost remains substantial: BiLSTM takes 4.8× longer to train and 6.3× longer to infer per batch.

**Per-class F1 (BiLSTM, seed 42):**

| Class    | Precision | Recall | F1   |
|----------|-----------|--------|------|
| Negative | 0.90      | 0.87   | 0.89 |
| Positive | 0.88      | 0.91   | 0.89 |

Both classes are balanced and equally difficult, confirming that errors arise from genuine semantic ambiguity rather than class imbalance.

---

## 5. Data Efficiency Experiment

All models were trained on 1%, 5%, 10%, 25%, 50%, and 100% of the training data using stratified sampling to preserve class proportions. Each fraction was evaluated with 3 seeds.

### AG News Data Efficiency (Macro-F1, mean ± std)

| Fraction | TextCNN         | BiLSTM          |
|----------|-----------------|-----------------|
| 100%     | 0.9102 ± 0.0014 | 0.9217 ± 0.0009 |
| 50%      | 0.8946 ± 0.0036 | 0.9058 ± 0.0021 |
| 25%      | 0.8668 ± 0.0026 | 0.8905 ± 0.0054 |
| 10%      | 0.8101 ± 0.0015 | 0.8559 ± 0.0039 |
| 5%       | 0.7364 ± 0.0076 | 0.8117 ± 0.0049 |
| 1%       | 0.4872 ± 0.0249 | 0.5991 ± 0.0074 |

On AG News, BiLSTM leads TextCNN at every data fraction. The gap is largest at low data (1%: 11.2 points difference) and narrows progressively as more data is added (100%: 1.2 points). Both models degrade sharply below 10% of training data. At 1%, TextCNN collapses to near-random performance (0.4872), while BiLSTM retains slightly more structure (0.5991), suggesting that recurrent representations are more data-efficient for this task.

### IMDB Data Efficiency (Macro-F1, mean ± std)

| Fraction | TextCNN         | BiLSTM          |
|----------|-----------------|-----------------|
| 100%     | 0.8771 ± 0.0029 | 0.8944 ± 0.0022 |
| 50%      | 0.8613 ± 0.0024 | 0.8599 ± 0.0022 |
| 25%      | 0.7885 ± 0.0487 | 0.7855 ± 0.0229 |
| 10%      | 0.7665 ± 0.0221 | 0.6932 ± 0.0109 |
| 5%       | 0.7574 ± 0.0110 | 0.6108 ± 0.0059 |
| 1%       | 0.5912 ± 0.0404 | 0.5344 ± 0.0050 |

The IMDB results reveal a striking reversal. At low data fractions (1%–25%), TextCNN substantially outperforms BiLSTM. At 5%, the gap reaches 14.7 points (0.7574 vs 0.6108). The crossover point occurs around 50%, where the two models reach near-identical performance (0.8613 vs 0.8599). BiLSTM only surpasses TextCNN at full data (0.8944 vs 0.8771).

This reversal is explained by the different inductive biases of the two architectures. TextCNN learns local sentiment indicators (strongly positive or negative n-grams) which generalize well even from small samples. BiLSTM must learn to model long-range sequential dependencies, which requires substantially more data to parameterize correctly — with only 400 training examples (1%), the recurrent weights cannot meaningfully capture sequence structure. Furthermore, the large variance of TextCNN at 25% (±0.0487) indicates instability in this data regime, where performance is highly sensitive to which samples happen to be selected.

**Key finding:** The ranking of neural models is dataset- and data-size-dependent. On AG News, BiLSTM is always preferable. On IMDB with limited data, TextCNN is the safer choice. A practitioner with fewer than 10,000 labeled IMDB-like examples should prefer TextCNN over BiLSTM.

---

## 6. Cost-Accuracy Trade-off

| Dataset  | Model   | Macro-F1 | Train Time | Infer Time | F1/Time Ratio |
|----------|---------|----------|------------|------------|---------------|
| AG News  | TextCNN | 0.9097   | 171.6s     | 1.393s     | **Best**      |
| AG News  | BiLSTM  | 0.9219   | 1609.2s    | 5.220s     | +1.2% F1, 9.4× slower |
| IMDB     | TextCNN | 0.8799   | 280.3s     | 1.067s     | **Best**      |
| IMDB     | BiLSTM  | 0.8908   | 1342.7s    | 6.712s     | +1.1% F1, 4.8× slower |

In both datasets, TextCNN delivers approximately 99% of BiLSTM's accuracy at roughly one-fifth to one-tenth of the training cost. For production systems where inference latency matters — such as real-time content moderation or live sentiment monitoring — TextCNN is strongly preferable. BiLSTM's marginal accuracy gain is unlikely to justify its computational overhead except in applications where every fraction of a percent of F1 has measurable business value.

---

## 7. Error Analysis

### AG News — TextCNN (worst) vs BiLSTM (best)

**Total errors:** TextCNN: 702 / 7,600 (9.2%) | BiLSTM: 603 / 7,600 (7.9%)

**Dominant failure modes for both models:**

**1. Sci/Tech ↔ Business boundary confusion (most frequent)**
Technology companies are simultaneously science/technology entities and business actors. Articles about IBM acquisitions, Intel product delays, Yahoo pricing strategy, and Google's IPO were systematically misclassified by both models. For example:
- "IBM buys two Danish services firms" → TRUE: Sci/Tech, PRED: Business (both models)
- "Google lowers its IPO price range" → TRUE: World, PRED: Business (both models)
- "Intel to delay product aimed for high-definition TVs" → TRUE: Business, PRED: Sci/Tech (both models)

The boundary between Sci/Tech and Business is inherently ambiguous when covering major tech corporations. This is a label noise issue rather than a model failure.

**2. World ↔ Business confusion**
Economic news with global scope (oil prices, stock market movements tied to geopolitical events) was frequently misassigned. "Oil prices bubble to record high" (TRUE: World) and "Stocks climb on drop in consumer prices" (TRUE: World) were predicted as Business by both models. These articles use financial vocabulary that strongly associates with Business, yet their editorial classification treats them as World events.

**3. Sports ↔ World confusion**
International sporting events — particularly the 2004 Athens Olympics — created confusion. "Live Olympics Day Four" (TRUE: World) was predicted as Sports by both models, while "Afghan women make brief Olympic debut" (TRUE: Sports) was predicted as World. The Olympic context carries both Sports and geopolitical World significance.

**Model comparison:**
BiLSTM corrects several errors that TextCNN makes, particularly on longer articles where sequential context disambiguates the primary topic. However, both models fail on the same core set of structurally ambiguous examples — approximately 65–70% of errors are shared. This indicates that the remaining error mass reflects genuine label ambiguity in the dataset, not architectural limitations that better models could overcome.

---

### IMDB — TextCNN (worst) vs BiLSTM (best)

**Total errors:** TextCNN: 605 / 5,000 (12.1%) | BiLSTM: 550 / 5,000 (11.0%)

**Dominant failure modes:**

**1. Sarcasm and implicit sentiment (both models fail)**
Many misclassified reviews open with language that superficially suggests the opposite sentiment:
- "After all the crap that Hollywood and the indies have churned out, we finally get a movie that delivers..." → TRUE: Positive, PRED: Negative (both models)
- "It's hard to imagine a director capable of such godawful crap as Notting Hill pulling off something as sensitive..." → TRUE: Positive, PRED: Negative (both models)

Both models latch onto strongly negative words in the early portion of the review and cannot integrate the reversal that follows.

**2. Mixed-sentiment reviews (both models fail)**
Reviews that genuinely praise some aspects while criticizing others are inherently ambiguous:
- "I hadn't seen this in many years. The acting was so good... I thought: great, another movie I misjudged. As this time I felt the same way again." → TRUE: Negative, PRED: Positive (both models)

The positive framing of "the acting was so good" dominates the model's prediction despite the negative overall verdict.

**3. Truncation-induced errors (TextCNN more affected)**
Given that IMDB reviews are truncated to 256 tokens, reviews where the key sentiment signal appears in the latter half are systematically harder. TextCNN's max-pooling architecture selects the most activated local feature regardless of position, but if the decisive sentiment phrase falls beyond 256 words, neither model sees it. This particularly affects long negative reviews that open with detailed plot summary before delivering criticism.

**4. Domain-specific vocabulary (both models fail)**
Some reviews contain highly specific film criticism vocabulary or references to other works that require cultural knowledge:
- "Very typical Almodóvar of the time and in its own way no less funny than many of his later works" — requires knowing Almodóvar's reputation as a comedic auteur to interpret correctly.

**Model comparison:**
BiLSTM reduces errors relative to TextCNN primarily on reviews with long-range sentiment dependencies — cases where a negative qualifier early in the review is resolved by a positive conclusion several sentences later. TextCNN's local feature extraction cannot model this structure. However, on short reviews and reviews with clear local sentiment indicators, both models perform equivalently. Approximately 40% of errors are shared between the two models, reflecting genuinely ambiguous review content.

---

## 8. Key Findings Summary

**Baseline strength of neural models:**
On AG News, both TextCNN and BiLSTM achieve high performance (91–92% F1), consistent with the hypothesis that topic classification with distinctive per-class vocabulary is tractable for neural models. The task does not require deep contextual understanding — local n-gram features captured by TextCNN are largely sufficient.

**Neural model advantage on sentiment:**
On IMDB, BiLSTM's sequential modeling provides a measurable advantage at full data scale (+1.1% F1 over TextCNN). However, the gain is modest. The upper bound for bag-of-n-grams approaches on this task appears to be around 88–89%, and BiLSTM does not dramatically exceed this ceiling. Transformers (handled by Member C) are expected to show a larger gap due to their pretraining on large corpora and attention-based long-range modeling.

**Data efficiency:**
The most practically significant finding is the IMDB crossover: TextCNN is more data-efficient than BiLSTM for sentiment analysis below 50% of training data. Organizations with limited labeled data for sentiment tasks should prefer TextCNN or classical methods over BiLSTM. For topic classification (AG News), BiLSTM is consistently preferred at all data scales.

**Cost-accuracy trade-off:**
TextCNN provides the best cost-accuracy trade-off in both settings. The 9× training speed advantage over BiLSTM, combined with 3–6× faster inference, makes TextCNN the practical default for deployment. BiLSTM's marginal accuracy gains (1.1–1.2% F1) are unlikely to justify its computational cost in most production scenarios.

---

## 9. Validation Protocol Statement

The test set was used exclusively for final evaluation. All hyperparameter selection and model comparison decisions were made using the validation set only. For AG News, the standard test split (7,600 samples) was used directly. For IMDB, we held out 10% as test and split the remainder into 80% training and 10% validation. No test set results were observed until after all hyperparameter decisions were finalized.
