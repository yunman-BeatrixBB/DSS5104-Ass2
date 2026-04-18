# 成员A专属执行计划：经典文本分类方法

## 你的核心任务

实现并评估以下经典方法：
1. **TF-IDF + Logistic Regression** (主要)
2. **TF-IDF + SVM** (对比)
3. **超参数调优**: C值、max_features、n-gram范围
4. **数据效率实验**: 在1%, 5%, 10%, 25%, 50%, 100%数据上运行
5. **3个随机种子**: 42, 123, 456 (计算均值和标准差)

---

## 数据集分析

### 数据集1: AG News (传统方法应该有竞争力)
- **Kaggle链接**: https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset
- **类别**: 4类 (World, Sports, Business, Sci/Tech)
- **样本**: 120K训练 / 7.6K测试
- **假设**: 主题分类有明显关键词，TF-IDF应该表现不错

### 数据集2: IMDb (Transformer应该有优势)
- **Kaggle链接**: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- **类别**: 2类 (Positive, Negative)
- **样本**: 50K (需要手动划分训练/验证/测试)
- **假设**: 情感分析需要理解上下文和语义

---

## 详细步骤：第一阶段 - 环境搭建 (Day 1-2)

### Step 1.1: 下载数据集

**AG News**:
```bash
# 从Kaggle下载后，文件结构应该是:
# train.csv: class, title, description
# test.csv: class, title, description

# 合并title和description作为输入文本
```

**IMDb**:
```bash
# IMDb Dataset.csv
# 列: review, sentiment
# sentiment: positive/negative
```

### Step 1.2: 创建你的开发文件

在GitHub仓库中创建以下文件:

```python
# src/classical_models.py
import numpy as np
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pickle

class ClassicalTextClassifier:
    """
    经典文本分类器：TF-IDF + Logistic Regression/SVM
    """
    def __init__(self, model_type='lr', C=1.0, max_features=50000, 
                 ngram_range=(1, 2), random_state=42):
        """
        参数:
            model_type: 'lr' (Logistic Regression) 或 'svm' (LinearSVC)
            C: 正则化参数
            max_features: TF-IDF最大特征数
            ngram_range: n-gram范围，(1,1)表示unigram，(1,2)表示uni+bigram
            random_state: 随机种子
        """
        self.model_type = model_type
        self.C = C
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state
        
        self.vectorizer = None
        self.model = None
        self.training_time = None
        self.inference_time = None
        
    def fit(self, texts, labels):
        """训练模型"""
        start_time = time.time()
        
        # 1. 构建TF-IDF向量器
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',  # 可选：去除停用词
            lowercase=True,
            min_df=2,  # 忽略出现次数少于2次的词
            max_df=0.95  # 忽略出现在95%以上文档中的词
        )
        
        # 2. 将文本转换为TF-IDF特征
        X = self.vectorizer.fit_transform(texts)
        
        # 3. 初始化分类器
        if self.model_type == 'lr':
            self.model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'  # 处理类别不平衡
            )
        elif self.model_type == 'svm':
            self.model = LinearSVC(
                C=self.C,
                max_iter=2000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        # 4. 训练
        self.model.fit(X, labels)
        
        self.training_time = time.time() - start_time
        
        return self
    
    def predict(self, texts):
        """预测"""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
    
    def predict_proba(self, texts):
        """预测概率（仅Logistic Regression支持）"""
        if self.model_type != 'lr':
            raise ValueError("predict_proba only available for Logistic Regression")
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)
    
    def evaluate(self, texts, labels, dataset_name=""):
        """
        评估模型，返回完整指标
        """
        # 推理时间
        start_time = time.time()
        predictions = self.predict(texts)
        self.inference_time = time.time() - start_time
        
        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        per_class_f1 = f1_score(labels, predictions, average=None).tolist()
        
        # 打印详细报告
        print(f"\n=== {dataset_name} Results ({self.model_type}, C={self.C}) ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Training Time: {self.training_time:.2f}s")
        print(f"Inference Time: {self.inference_time:.2f}s")
        print("\nPer-class F1:")
        print(classification_report(labels, predictions))
        
        return {
            'model_type': self.model_type,
            'C': self.C,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'train_time': self.training_time,
            'inference_time': self.inference_time,
            'predictions': predictions
        }
    
    def save(self, filepath):
        """保存模型"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'config': {
                    'model_type': self.model_type,
                    'C': self.C,
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range
                }
            }, f)
    
    def load(self, filepath):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.model = data['model']
            config = data['config']
            self.model_type = config['model_type']
            self.C = config['C']
            self.max_features = config['max_features']
            self.ngram_range = config['ngram_range']
        return self
```

---

## 详细步骤：第二阶段 - 数据加载 (Day 2-3)

### Step 2.1: AG News数据加载

```python
# src/data_loaders/ag_news_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_ag_news(data_dir='data/ag_news'):
    """
    加载AG News数据集
    假设文件: train.csv, test.csv
    """
    # 读取数据
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    # AG News的列: class, title, description
    # 合并title和description作为输入
    train_texts = (train_df['title'].fillna('') + ' ' + 
                   train_df['description'].fillna('')).str.strip()
    test_texts = (test_df['title'].fillna('') + ' ' + 
                  test_df['description'].fillna('')).str.strip()
    
    # 类别标签需要减1，因为原始数据是1-indexed
    train_labels = train_df['class'].values - 1
    test_labels = test_df['class'].values - 1
    
    # 从训练集划分验证集 (10%)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, 
        stratify=train_labels  # 分层抽样保持类别比例
    )
    
    return {
        'train': (train_texts.tolist(), train_labels),
        'val': (val_texts.tolist(), val_labels),
        'test': (test_texts.tolist(), test_labels),
        'num_classes': 4,
        'class_names': ['World', 'Sports', 'Business', 'Sci/Tech']
    }
```

### Step 2.2: IMDb数据加载

```python
# src/data_loaders/imdb_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_imdb(data_dir='data/imdb'):
    """
    加载IMDb数据集
    需要手动划分为 train/val/test = 70/15/15
    """
    df = pd.read_csv(f'{data_dir}/IMDB Dataset.csv')
    
    # 转换标签
    texts = df['review'].values
    labels = (df['sentiment'] == 'positive').astype(int).values
    
    # 首先划分出测试集15%
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # 然后从剩余数据中划分验证集 (15% / 85% ≈ 17.6%)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=0.176, random_state=42,
        stratify=train_val_labels
    )
    
    return {
        'train': (train_texts.tolist(), train_labels),
        'val': (val_texts.tolist(), val_labels),
        'test': (test_texts.tolist(), test_labels),
        'num_classes': 2,
        'class_names': ['Negative', 'Positive']
    }
```

### Step 2.3: 分层抽样函数（数据效率实验用）

```python
# src/data_loaders/subsample.py
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_subsample(texts, labels, ratio, random_state=42):
    """
    分层抽样，保持类别比例
    
    参数:
        texts: 文本列表
        labels: 标签数组
        ratio: 抽样比例 (0.01 = 1%)
        random_state: 随机种子
    返回:
        抽样后的(texts, labels)
    """
    if ratio >= 1.0:
        return texts, labels
    
    # 使用StratifiedShuffleSplit进行分层抽样
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-ratio, random_state=random_state)
    
    for keep_idx, _ in sss.split(texts, labels):
        sub_texts = [texts[i] for i in keep_idx]
        sub_labels = labels[keep_idx]
    
    return sub_texts, sub_labels
```

---

## 详细步骤：第三阶段 - 超参数搜索 (Day 4-5)

### Step 3.1: 超参数搜索脚本

```python
# experiments/hyperparameter_search.py
import sys
sys.path.append('src')

import json
import pandas as pd
from classical_models import ClassicalTextClassifier
from data_loaders.ag_news_loader import load_ag_news
from data_loaders.imdb_loader import load_imdb

def hyperparameter_search(dataset_name, data_loader, random_state=42):
    """
    对给定数据集进行超参数搜索
    """
    print(f"\n{'='*60}")
    print(f"Hyperparameter Search for {dataset_name}")
    print(f"{'='*60}")
    
    # 加载数据
    data = data_loader()
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    
    # 超参数搜索空间
    param_grid = {
        'model_type': ['lr', 'svm'],
        'C': [0.1, 1.0, 10.0],
        'max_features': [10000, 50000],
        'ngram_range': [(1, 1), (1, 2)]
    }
    
    results = []
    best_macro_f1 = 0
    best_params = None
    
    # 遍历所有组合（剪枝：可以先固定一些参数）
    for model_type in param_grid['model_type']:
        for C in param_grid['C']:
            for ngram_range in param_grid['ngram_range']:
                print(f"\nTrying: {model_type}, C={C}, ngram={ngram_range}")
                
                # 训练模型
                clf = ClassicalTextClassifier(
                    model_type=model_type,
                    C=C,
                    max_features=50000,  # 固定使用较大的词汇量
                    ngram_range=ngram_range,
                    random_state=random_state
                )
                clf.fit(train_texts, train_labels)
                
                # 在验证集上评估
                metrics = clf.evaluate(val_texts, val_labels, "Validation")
                
                result = {
                    'dataset': dataset_name,
                    'model_type': model_type,
                    'C': C,
                    'ngram_range': str(ngram_range),
                    'val_accuracy': metrics['accuracy'],
                    'val_macro_f1': metrics['macro_f1'],
                    'train_time': metrics['train_time']
                }
                results.append(result)
                
                # 更新最佳参数
                if metrics['macro_f1'] > best_macro_f1:
                    best_macro_f1 = metrics['macro_f1']
                    best_params = {
                        'model_type': model_type,
                        'C': C,
                        'max_features': 50000,
                        'ngram_range': ngram_range
                    }
                
                print(f"Val Macro F1: {metrics['macro_f1']:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{dataset_name}_hyperparam_search.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Best Params for {dataset_name}:")
    print(f"{best_params}")
    print(f"Best Val Macro F1: {best_macro_f1:.4f}")
    print(f"{'='*60}")
    
    return best_params, results_df

if __name__ == '__main__':
    # AG News
    ag_best_params, ag_results = hyperparameter_search('ag_news', load_ag_news)
    
    # IMDb
    imdb_best_params, imdb_results = hyperparameter_search('imdb', load_imdb)
    
    # 保存最佳参数
    with open('results/best_params.json', 'w') as f:
        json.dump({
            'ag_news': ag_best_params,
            'imdb': imdb_best_params
        }, f, indent=2)
```

---

## 详细步骤：第四阶段 - 数据效率实验 (Day 6-10)

### Step 4.1: 主实验脚本

```python
# experiments/run_classical_experiments.py
import sys
sys.path.append('src')

import json
import numpy as np
import pandas as pd
from classical_models import ClassicalTextClassifier
from data_loaders.ag_news_loader import load_ag_news
from data_loaders.imdb_loader import load_imdb
from data_loaders.subsample import stratified_subsample

def run_single_experiment(dataset_name, data_loader, data_ratio, 
                          hyperparams, seed=42):
    """
    运行单次实验
    
    返回结果字典
    """
    # 设置随机种子
    np.random.seed(seed)
    
    # 加载数据
    data = data_loader()
    train_texts_full, train_labels_full = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    # 分层抽样
    train_texts, train_labels = stratified_subsample(
        train_texts_full, train_labels_full, data_ratio, seed
    )
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}, Ratio: {data_ratio}, Seed: {seed}")
    print(f"Training samples: {len(train_texts)}")
    print(f"{'='*60}")
    
    # 训练模型
    clf = ClassicalTextClassifier(
        model_type=hyperparams['model_type'],
        C=hyperparams['C'],
        max_features=hyperparams['max_features'],
        ngram_range=hyperparams['ngram_range'],
        random_state=seed
    )
    clf.fit(train_texts, train_labels)
    
    # 在验证集上评估（用于监控，但不用于报告）
    val_metrics = clf.evaluate(val_texts, val_labels, "Validation")
    
    # 在测试集上评估（最终结果）
    test_metrics = clf.evaluate(test_texts, test_labels, "Test")
    
    # 构建结果字典
    result = {
        'dataset': dataset_name,
        'model': f"tfidf_{hyperparams['model_type']}",
        'seed': seed,
        'data_ratio': data_ratio,
        'train_size': len(train_texts),
        'hyperparams': json.dumps(hyperparams),
        
        # 验证集指标（仅用于开发）
        'val_accuracy': val_metrics['accuracy'],
        'val_macro_f1': val_metrics['macro_f1'],
        
        # 测试集指标（报告用）
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'per_class_f1': json.dumps(test_metrics['per_class_f1']),
        
        # 时间和资源
        'train_time': test_metrics['train_time'],
        'inference_time': test_metrics['inference_time'],
        'model_size_mb': 0.1  # 经典模型很小，估算值
    }
    
    return result

def run_all_experiments():
    """运行所有实验"""
    
    # 加载最佳超参数
    with open('results/best_params.json', 'r') as f:
        best_params = json.load(f)
    
    # 实验配置
    datasets = {
        'ag_news': load_ag_news,
        'imdb': load_imdb
    }
    
    data_ratios = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    seeds = [42, 123, 456]
    
    all_results = []
    
    # 遍历所有组合
    for dataset_name, data_loader in datasets.items():
        hyperparams = best_params[dataset_name]
        
        for data_ratio in data_ratios:
            for seed in seeds:
                try:
                    result = run_single_experiment(
                        dataset_name, data_loader, data_ratio,
                        hyperparams, seed
                    )
                    all_results.append(result)
                    
                    # 每完成一个实验就保存，防止丢失
                    pd.DataFrame(all_results).to_csv(
                        'results/classical_experiments.csv', index=False
                    )
                    
                except Exception as e:
                    print(f"Error in experiment: {e}")
                    continue
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Results saved to: results/classical_experiments.csv")
    print("="*60)
    
    return pd.DataFrame(all_results)

if __name__ == '__main__':
    results_df = run_all_experiments()
```

---

## 详细步骤：第五阶段 - 汇总结果 (Day 11-12)

### Step 5.1: 结果汇总脚本

```python
# experiments/summarize_results.py
import pandas as pd
import json

def summarize_results():
    """
    汇总实验结果，计算均值和标准差
    """
    df = pd.read_csv('results/classical_experiments.csv')
    
    # 按数据集、模型、数据比例分组，计算统计量
    summary = df.groupby(['dataset', 'model', 'data_ratio']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_macro_f1': ['mean', 'std'],
        'train_time': ['mean', 'std'],
        'inference_time': ['mean', 'std']
    }).reset_index()
    
    # 展平列名
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                       for col in summary.columns.values]
    
    # 保存汇总结果
    summary.to_csv('results/classical_summary.csv', index=False)
    
    print("Summary Statistics:")
    print(summary)
    
    # 为报告生成LaTeX表格
    print("\n\nLaTeX Table Format:")
    for dataset in summary['dataset'].unique():
        print(f"\n% {dataset} results")
        sub = summary[summary['dataset'] == dataset]
        for _, row in sub.iterrows():
            print(f"{row['data_ratio']*100:.0f}\% & "
                  f"{row['test_accuracy_mean']:.3f} $\\pm$ {row['test_accuracy_std']:.3f} & "
                  f"{row['test_macro_f1_mean']:.3f} $\\pm$ {row['test_macro_f1_std']:.3f} \\\\")
    
    return summary

if __name__ == '__main__':
    summarize_results()
```

---

## 详细步骤：第六阶段 - 错误分析数据准备 (Day 13-14)

### Step 6.1: 保存预测结果供分析

```python
# experiments/save_predictions_for_analysis.py
import sys
sys.path.append('src')

import json
import pandas as pd
from classical_models import ClassicalTextClassifier
from data_loaders.ag_news_loader import load_ag_news
from data_loaders.imdb_loader import load_imdb

def save_predictions(dataset_name, data_loader, hyperparams, seed=42):
    """
    保存最佳和最差模型的预测结果，供误差分析使用
    """
    # 加载数据
    data = data_loader()
    train_texts, train_labels = data['train']
    test_texts, test_labels = data['test']
    class_names = data['class_names']
    
    # 训练模型
    clf = ClassicalTextClassifier(
        model_type=hyperparams['model_type'],
        C=hyperparams['C'],
        max_features=hyperparams['max_features'],
        ngram_range=hyperparams['ngram_range'],
        random_state=seed
    )
    clf.fit(train_texts, train_labels)
    
    # 预测
    predictions = clf.predict(test_texts)
    
    # 保存前50个错误案例
    errors = []
    for i, (text, true, pred) in enumerate(zip(test_texts, test_labels, predictions)):
        if true != pred:
            errors.append({
                'index': i,
                'text': text[:500],  # 截取前500字符
                'true_label': class_names[true],
                'predicted_label': class_names[pred],
                'dataset': dataset_name,
                'model': f"tfidf_{hyperparams['model_type']}"
            })
        if len(errors) >= 50:
            break
    
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(f'results/{dataset_name}_classical_errors.csv', index=False)
    
    print(f"Saved {len(errors)} error cases to results/{dataset_name}_classical_errors.csv")
    return errors_df

if __name__ == '__main__':
    # 加载最佳参数
    with open('results/best_params.json', 'r') as f:
        best_params = json.load(f)
    
    # AG News
    save_predictions('ag_news', load_ag_news, best_params['ag_news'])
    
    # IMDb
    save_predictions('imdb', load_imdb, best_params['imdb'])
```

---

## 你的交付清单

### 代码文件（提交到GitHub）
- [ ] `src/classical_models.py` - 核心分类器实现
- [ ] `src/data_loaders/ag_news_loader.py` - AG News数据加载
- [ ] `src/data_loaders/imdb_loader.py` - IMDb数据加载
- [ ] `src/data_loaders/subsample.py` - 分层抽样
- [ ] `experiments/hyperparameter_search.py` - 超参数搜索
- [ ] `experiments/run_classical_experiments.py` - 主实验
- [ ] `experiments/summarize_results.py` - 结果汇总
- [ ] `experiments/save_predictions_for_analysis.py` - 错误分析数据

### 结果文件（用于报告）
- [ ] `results/ag_news_hyperparam_search.csv` - AG News超参数搜索结果
- [ ] `results/imdb_hyperparam_search.csv` - IMDb超参数搜索结果
- [ ] `results/best_params.json` - 每个数据集的最佳参数
- [ ] `results/classical_experiments.csv` - 所有实验原始结果
- [ ] `results/classical_summary.csv` - 汇总统计（均值±标准差）
- [ ] `results/ag_news_classical_errors.csv` - AG News错误案例
- [ ] `results/imdb_classical_errors.csv` - IMDb错误案例

### 提供给成员D的内容
- [ ] 学习曲线数据（用于绘制准确率/F1 vs 数据量）
- [ ] 与Transformer方法的对比表格
- [ ] 训练/推理时间对比
- [ ] 错误案例分析表格

---

## 关键注意事项

### 1. 随机种子设置
```python
import numpy as np
import random

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    # sklearn也使用numpy的随机种子
```

### 2. 分层抽样重要性
```python
from sklearn.model_selection import StratifiedShuffleSplit

# 必须保持类别比例，否则结果不可比
```

### 3. 类别不平衡处理
```python
# 在模型初始化时启用class_weight='balanced'
LogisticRegression(class_weight='balanced')
LinearSVC(class_weight='balanced')
```

### 4. 超参数搜索策略
- **先粗搜**: C ∈ {0.1, 1, 10}, ngram ∈ {(1,1), (1,2)}
- **报告用**: 使用验证集上macro_f1最好的参数
- **测试集**: 只在最后使用一次，用于报告最终指标

### 5. 运行时间预估（参考）

| 数据集 | 数据量 | 单次训练时间 | 总实验次数 | 预估总时间 |
|--------|--------|--------------|------------|------------|
| AG News | 120K | ~10s | 6比例×3种子×2模型≈36次 | ~10分钟 |
| IMDb | 35K | ~5s | 36次 | ~5分钟 |

经典方法非常快，你可以在本地轻松完成。

---

## 接下来做什么

1. **今天就做**: 
   - 从Kaggle下载两个数据集
   - 创建GitHub仓库（或让成员D创建后拉取）
   - 创建 `src/classical_models.py` 的基础框架

2. **本周完成**:
   - 数据加载模块
   - 单数据集的超参数搜索
   - 验证代码可以跑通

3. **与团队同步**:
   - 确认数据划分方式（成员D应该统一所有方法的数据划分）
   - 确认随机种子列表
   - 每周汇报进度

需要我详细展开任何具体部分的代码实现吗？例如：
- 如何安装依赖
- 如何处理Kaggle数据格式
- 如何调试超参数搜索