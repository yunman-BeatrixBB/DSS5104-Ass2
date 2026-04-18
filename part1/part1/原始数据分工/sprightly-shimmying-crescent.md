# DSS5104 文本分类作业 - 4人团队协作完整计划

## 一、数据集选择（建议方案）

### 数据集1: AG News (主题分类 - 传统方法应有竞争力)
- **类型**: 4类新闻主题分类（World, Sports, Business, Sci/Tech）
- **样本数**: 120K训练 / 7.6K测试
- **来源**: HuggingFace `ag_news`
- **为什么选它**: 主题分类有明显关键词特征，TF-IDF应该表现不错

### 数据集2: SST-2 (情感分析 - Transformer应有明显优势)
- **类型**: 电影评论二分类情感分析
- **样本数**: 67K训练 / 872测试
- **来源**: HuggingFace `glue` 子集 `sst2`
- **为什么选它**: 情感分析需要理解上下文和语义，Transformer应该优于传统方法

**备选方案**: 
- 如果SST-2太简单，可用 `tweet_eval` 的 `irony` 或 `hate` 子集（讽刺/仇恨言论检测，更难）
- 或用 `banking77`（意图检测，细粒度分类）

---

## 二、团队分工（4人）

### 成员A - 经典方法负责人
**负责内容**:
- TF-IDF + Logistic Regression
- TF-IDF + SVM
- 超参数调优（C值、n-gram范围）
- 经典方法的数据效率实验

### 成员B - 神经网络方法负责人
**负责内容**:
- FastText实现
- TextCNN实现
- BiLSTM实现（至少选1-2个）
- 神经网络的数据效率实验

### 成员C - Transformer方法负责人
**负责内容**:
- DistilBERT微调
- BERT-base/RoBERTa微调
- SetFit小样本实验（仅1%, 5%, 10%数据）
- Transformer的数据效率实验

### 成员D - 整合与报告负责人
**负责内容**:
- 统一的数据加载和预处理pipeline
- 实验框架搭建（随机种子、结果记录）
- 可视化（学习曲线、对比表格）
- 误差分析和报告撰写
- GitHub仓库管理

---

## 三、详细执行步骤

### Phase 1: 环境搭建与数据准备（第1周，2-3天）

#### 步骤1.1: 创建GitHub仓库
**负责人**: 成员D
- 创建repo，添加README
- 创建分支策略（main + 每人一个feature分支）
- 设置 `.gitignore`（忽略数据文件、模型checkpoint）

#### 步骤1.2: 统一环境配置
**负责人**: 成员D
```bash
# requirements.txt 模板
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
sentence-transformers>=2.2.0
setfit>=1.0.0
gensim>=4.3.0
nltk>=3.8.0
tqdm>=4.65.0
```

#### 步骤1.3: 数据加载模块
**负责人**: 成员D
```python
# data_loader.py 框架
from datasets import load_dataset
import numpy as np

def load_ag_news():
    """加载AG News数据集"""
    dataset = load_dataset("ag_news")
    return dataset

def load_sst2():
    """加载SST-2数据集"""
    dataset = load_dataset("glue", "sst2")
    return dataset

def subsample_dataset(dataset, ratio, seed=42):
    """分层抽样，保持类别比例"""
    # 实现分层抽样
    pass

def get_text_length_stats(dataset):
    """统计文本长度分布（用于讨论截断影响）"""
    pass
```

#### 步骤1.4: 实验框架
**负责人**: 成员D
```python
# experiment_framework.py
class ExperimentRunner:
    """统一实验运行器，确保可重复性"""
    def __init__(self, seed=42):
        self.seed = seed
        self.set_seed()
    
    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # ...
    
    def run_experiment(self, model, dataset, data_ratio):
        """运行单次实验，返回结果字典"""
        pass
```

---

### Phase 2: 模型实现（第1-2周）

#### 步骤2.1: 经典方法实现
**负责人**: 成员A
```python
# classical_models.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

class ClassicalTextClassifier:
    def __init__(self, model_type='lr', C=1.0, max_features=50000, ngram_range=(1,2)):
        self.model_type = model_type
        self.C = C
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = None
        self.model = None
    
    def fit(self, texts, labels):
        # TF-IDF + 模型训练
        pass
    
    def predict(self, texts):
        pass
    
    def evaluate(self, texts, labels):
        # 返回acc, macro_f1, per_class_f1, train_time, inference_time
        pass
```

**超参数搜索空间**:
- C: [0.1, 1, 10]
- max_features: [10000, 50000]
- ngram_range: [(1,1), (1,2)]

#### 步骤2.2: 神经网络方法实现
**负责人**: 成员B
```python
# neural_models.py
import torch
import torch.nn as nn

class FastTextClassifier(nn.Module):
    """FastText风格分类器"""
    pass

class TextCNN(nn.Module):
    """卷积神经网络文本分类器"""
    pass

class BiLSTMClassifier(nn.Module):
    """双向LSTM分类器"""
    pass
```

**超参数搜索空间**:
- Learning rate: [1e-3, 5e-4, 1e-4]
- Epochs: [5, 10, 20]
- Batch size: [32, 64]

#### 步骤2.3: Transformer方法实现
**负责人**: 成员C
```python
# transformer_models.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from setfit import SetFitModel

class TransformerClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
    
    def train(self, train_texts, train_labels, val_texts, val_labels, lr=2e-5, epochs=3):
        pass

class SetFitClassifier:
    """专门用于小样本实验"""
    def __init__(self, model_name='sentence-transformers/paraphrase-mpnet-base-v2'):
        self.model = SetFitModel.from_pretrained(model_name)
    
    def train(self, texts, labels):
        # SetFit使用对比学习，不需要大量数据
        pass
```

**超参数搜索空间**:
- Learning rate: [1e-5, 2e-5, 5e-5]
- Epochs: [3, 4, 5]
- Early stopping patience: 2

---

### Phase 3: 数据效率实验（第2周）

#### 步骤3.1: 定义数据比例
**负责人**: 成员D协调，所有人执行
```python
data_ratios = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
# SetFit 只在 [0.01, 0.05, 0.1] 上运行
```

#### 步骤3.2: 运行实验脚本
```python
# run_all_experiments.py
seeds = [42, 123, 456]  # 3个随机种子

for seed in seeds:
    for ratio in data_ratios:
        # 1. 经典方法
        for C in [0.1, 1, 10]:
            run_classical_experiment(seed, ratio, C)
        
        # 2. 神经网络方法
        run_neural_experiment(seed, ratio, lr=1e-3)
        
        # 3. Transformer方法
        run_transformer_experiment(seed, ratio, lr=2e-5)
        
        # 4. SetFit (仅小比例)
        if ratio <= 0.1:
            run_setfit_experiment(seed, ratio)
```

#### 步骤3.3: 结果记录格式
```python
results = {
    'dataset': 'ag_news',  # 或 'sst2'
    'model': 'tfidf_lr',
    'seed': 42,
    'data_ratio': 0.1,
    'hyperparams': {'C': 1.0, 'ngram_range': (1,2)},
    'test_accuracy': 0.85,
    'test_macro_f1': 0.84,
    'per_class_f1': [0.83, 0.85, 0.84, 0.86],
    'train_time_seconds': 12.5,
    'inference_time_seconds': 2.3,
    'model_size_mb': 0.5
}
```

---

### Phase 4: 结果分析与可视化（第3周）

#### 步骤4.1: 学习曲线绘制
**负责人**: 成员D
```python
# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(results_df, dataset_name):
    """
    绘制每种方法的准确率/F1 vs 数据量的曲线
    包含误差条（3个seed的标准差）
    """
    pass

def plot_comparison_table(results_df):
    """生成对比表格（最终报告用）"""
    pass

def plot_time_comparison(results_df):
    """训练时间/推理时间对比"""
    pass
```

#### 步骤4.2: 关键问题分析
**全员参与，成员D整理**:

1. **交叉点分析**: 找出TF-IDF和Transformer性能相等的临界点
2. **成本-精度权衡**: 计算性价比指标（如 准确率/训练时间）
3. **统计显著性**: 检查3个seed的方差，判断差异是否真实

---

### Phase 5: 误差分析（第3周）

#### 步骤5.1: 收集错误案例
**负责人**: 成员D
```python
# error_analysis.py
def analyze_errors(model, test_texts, test_labels, test_predictions, n_samples=30):
    """
    分析错误分类案例
    返回: 错误案例列表，包含文本、真实标签、预测标签
    """
    pass

def categorize_errors(error_cases):
    """
    人工标注错误类型：
    - 标签模糊/噪声
    - 文本过短
    - 讽刺/隐含意义
    - OOV词汇
    - 需要上下文理解
    """
    pass
```

#### 步骤5.2: 对比最佳和最差模型
**全员参与**:
- 对比最佳模型和最差模型在哪些样本上出错
- 讨论这说明各自的优势是什么

---

### Phase 6: 报告撰写（第4周）

#### 报告结构（10页以内）
**负责人**: 成员D主笔，其他人补充各自部分

```
1. Introduction (0.5页)
   - 简述文本分类的重要性和方法演进
   
2. Dataset Selection & Hypotheses (1页)
   - AG News: 为什么传统方法应该有竞争力
   - SST-2: 为什么Transformer应该有优势
   - 文本长度统计和类别分布
   
3. Methodology (1.5页)
   - 三层模型简述
   - 超参数搜索策略
   - 验证方案（明确说明测试集仅用于最终报告）
   
4. Results: Performance Comparison (2页)
   - 完整数据量下的对比表格
   - 学习曲线图（数据效率实验）
   - 训练/推理时间对比
   
5. Results: Data Efficiency Analysis (2页)
   - 不同数据比例下的性能变化
   - 交叉点分析
   - SetFit在小样本下的表现
   
6. Error Analysis (1.5页)
   - 错误案例展示（表格）
   - 错误类型分布
   - 不同模型的失效模式对比
   
7. Practical Recommendations (1页)
   - 回答关键问题
   - 为实际应用场景提供决策建议
   
8. Conclusion (0.5页)
```

---

## 四、关键检查点

### 每周同步会议要点

**Week 1 结束检查点**:
- [ ] GitHub仓库已创建，所有人有访问权限
- [ ] 数据加载模块工作正常
- [ ] 每人完成各自模型的基础实现

**Week 2 结束检查点**:
- [ ] 所有模型在完整数据上运行通过
- [ ] 数据效率实验至少完成50%
- [ ] 发现任何技术问题（如内存不足、运行时间过长）

**Week 3 结束检查点**:
- [ ] 所有实验完成（3个seed × 所有配置）
- [ ] 可视化图表已生成
- [ ] 误差分析完成

**Week 4 结束检查点**:
- [ ] 报告初稿完成
- [ ] 代码清理和文档完善
- [ ] 最终检查：所有结果可复现

---

## 五、风险管理

### 潜在问题及应对

| 风险 | 应对策略 |
|------|----------|
| Colab GPU时间不足 | 使用DistilBERT而非BERT-base；减少epoch；使用更小的验证集 |
| SetFit运行过慢 | 仅在1%, 5%, 10%数据上运行；使用更小的sentence-transformer |
| 团队成员进度不一致 | 每周同步；成员D负责协调；预留缓冲时间 |
| 模型性能与预期不符 | 记录真实结果，调整假设；作业重点是比较分析，不是追求SOTA |
| 随机种子导致方差大 | 增加seed数量（3-5个）；报告中讨论方差 |

---

## 六、GitHub仓库结构建议

```
text-classification-benchmark/
├── README.md                    # 运行说明
├── requirements.txt             # 依赖
├── data/                        # 数据（gitignore）
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # 数据加载
│   ├── classical_models.py     # 成员A
│   ├── neural_models.py        # 成员B
│   ├── transformer_models.py   # 成员C
│   ├── experiment_runner.py    # 成员D
│   └── utils.py
├── experiments/
│   ├── run_classical.py        # 成员A运行脚本
│   ├── run_neural.py           # 成员B运行脚本
│   ├── run_transformer.py      # 成员C运行脚本
│   └── run_all.py              # 统一运行
├── results/                     # 实验结果（gitignore大文件）
├── notebooks/
│   ├── analysis.ipynb          # 可视化分析
│   └── error_analysis.ipynb    # 误差分析
└── report/
    └── report.pdf              # 最终报告
```

---

## 七、协作工具推荐

1. **代码**: GitHub + Pull Request
2. **沟通**: WeChat/WhatsApp群组 + 每周视频会议
3. **文档**: Google Docs（共享报告草稿）
4. **实验追踪**: Weights & Biases 或简单的CSV文件
5. **开发环境**: Google Colab Pro（推荐至少1人购买，$10/月）

---

## 八、评估标准检查清单

提交前确认以下内容：

- [ ] 实现了3个层级的模型（经典至少2个，神经网络至少1个，Transformer至少2个）
- [ ] 使用了2个数据集（一个传统方法友好，一个Transformer友好）
- [ ] 数据效率实验包含至少6个数据比例（100%, 50%, 25%, 10%, 5%, 1%）
- [ ] SetFit仅在≤10%数据上评估
- [ ] 每个实验使用≥3个随机种子
- [ ] 报告了均值和标准差
- [ ] 包含超参数调优过程
- [ ] 包含训练时间和推理时间
- [ ] 误差分析包含20-30个错误案例
- [ ] 报告≤10页，无代码
- [ ] GitHub仓库包含README和requirements.txt
- [ ] 所有结果可复现
