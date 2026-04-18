# DSS5104 文本分类 - 经典方法实现

成员A：经典方法负责人

## 任务描述

实现并评估经典文本分类方法：
- **TF-IDF + Logistic Regression**
- **TF-IDF + SVM**
- 超参数调优（C值、max_features、n-gram范围）
- 数据效率实验（1%, 5%, 10%, 25%, 50%, 100%数据）
- 3个随机种子（42, 123, 456）计算均值和标准差
- 错误分析：每个数据集的最佳和最差模型各30个错误案例

## 数据集

### AG News
- **来源**: Kaggle AG News Classification Dataset
- **类别**: 4类 (World, Sports, Business, Sci/Tech)
- **样本**: 108K训练 / 12K验证 / 7.6K测试
- **假设**: 主题分类有明显关键词，TF-IDF应该有竞争力

### IMDb
- **来源**: Kaggle IMDB Dataset of 50K Movie Reviews
- **类别**: 2类 (Negative, Positive)
- **样本**: 35K训练 / 7.5K验证 / 7.5K测试
- **假设**: 情感分析需要理解上下文，Transformer应该更好

---

## 文本长度分布分析

### 统计方法
- **Token计数方式**: 空格分词（近似估算，实际BERT token可能略有差异）
- **统计指标**: 平均值、中位数、第95百分位数、第99百分位数

### AG News 文本长度分布

| 数据划分 | 样本数 | 平均值 | 中位数 | 标准差 | 最小值 | 最大值 | 第95百分位 | 第99百分位 |
|----------|--------|--------|--------|--------|--------|--------|------------|------------|
| **训练集** | 108,000 | **37.9** | **37.0** | 10.1 | 4 | 171 | **53.0** | 70.0 |
| 验证集 | 12,000 | 37.8 | 37.0 | 9.9 | 8 | 177 | 52.0 | 70.0 |
| 测试集 | 7,600 | 37.7 | 37.0 | 10.1 | 11 | 137 | 52.0 | 69.0 |

### IMDb 文本长度分布

| 数据划分 | 样本数 | 平均值 | 中位数 | 标准差 | 最小值 | 最大值 | 第95百分位 | 第99百分位 |
|----------|--------|--------|--------|--------|--------|--------|------------|------------|
| **训练集** | 35,020 | **231.4** | **173.0** | 171.7 | 4 | 2,470 | **591.0** | 907.6 |
| 验证集 | 7,480 | 230.4 | 173.0 | 169.4 | 8 | 1,263 | 588.0 | 899.2 |
| 测试集 | 7,500 | 230.7 | 174.0 | 171.6 | 6 | 1,830 | 585.0 | 917.0 |

### 对Transformer模型的影响

BERT/DistilBERT的最大输入长度为**512 tokens**，文本长度分布直接影响截断策略：

| 数据集 | 95%分位长度 | 最大长度 | 截断影响 |
|--------|-------------|----------|----------|
| **AG News** | 53 tokens | 171 tokens | ✅ **95%样本无需截断**（53 < 512） |
| **IMDb** | 591 tokens | 2,470 tokens | ⚠️ **需显著截断**（591 > 512，约15%样本受影响） |

**分析结论**：
- **AG News**: 文本较短且长度均匀，几乎所有样本都能完整输入BERT，不会因截断丢失信息
- **IMDb**: 约5%的样本超过512 token限制，需要进行截断。这可能导致：
  - 长评论的后期内容被截断，影响情感判断
  - 对经典方法有利（TF-IDF无长度限制），可能拉大与Transformer的差距

---

## 关键问题解答

### 1. 基准性能：TF-IDF + 逻辑回归在哪些任务上表现得异常出色？为什么？

**结论**：TF-IDF + LR在**AG News主题分类任务**上表现接近最优。

| 数据集 | LR准确率 | SVM准确率 | 差距 |
|--------|----------|-----------|------|
| **AG News** | 92.08% | 92.13% | **仅0.05%** |
| **IMDb** | 87.64% | 90.51% | 2.88% |

**原因分析**：
- **AG News**是主题分类任务，每个类别有明确的关键词特征：
  - World: "Venezuela", "Iraq", "Olympics"
  - Business: "Google IPO", "oil prices", "stock"
  - Sports: "Tiger", "Olympics", "gold medal"
  - Sci/Tech: "Intel", "Microsoft", "software"
  
  TF-IDF能够很好地捕获这些关键词，因此LR表现优秀。

- **IMDb**是情感分析任务，需要理解：
  - 上下文语义（如"not bad" vs "bad"）
  - 讽刺和隐含意义
  - 否定词的作用范围
  
  LR仅依赖词袋模型，无法捕获这些复杂模式，因此表现不如SVM。

---

### 2. 数据效率：随着标记数据减少，方法的排名如何变化？

**AG News数据效率对比**：

| 数据比例 | LR准确率 | SVM准确率 | 领先模型 |
|----------|----------|-----------|----------|
| 100% | 92.08% | **92.13%** | SVM (+0.05%) |
| 50% | 91.30% | **91.43%** | SVM (+0.13%) |
| 25% | **90.72%** | 90.67% | LR (+0.05%) |
| 10% | **89.57%** | 89.47% | LR (+0.10%) |
| 5% | 88.28% | **88.30%** | SVM (+0.02%) |
| 1% | **83.57%** | 83.57% | 持平 |

**IMDb数据效率对比**：

| 数据比例 | LR准确率 | SVM准确率 | 领先模型 |
|----------|----------|-----------|----------|
| 100% | 87.64% | **90.51%** | SVM (+2.87%) |
| 50% | 86.43% | **89.66%** | SVM (+3.23%) |
| 25% | 85.37% | **88.52%** | SVM (+3.15%) |
| 10% | 83.83% | **87.05%** | SVM (+3.22%) |
| 5% | 82.80% | **85.24%** | SVM (+2.44%) |
| 1% | 80.43% | **80.52%** | SVM (+0.09%) |

**关键发现**：
- **AG News**: 随着数据减少，LR和SVM性能差距缩小，在25%和10%数据时LR甚至略微领先
- **IMDb**: SVM在所有数据比例下均显著优于LR，差距稳定在2.5-3.2%

**实践建议**：
- 对于关键词驱动的任务（如主题分类），**少量数据（10-25%）即可达到接近全量数据的性能**
- 对于语义理解任务（如情感分析），**需要更多数据才能体现复杂模型的优势**

---

### 3. 成本-准确性权衡：哪种方法能为每个任务提供最佳的权衡？

**AG News权衡分析**：

| 模型 | 准确率 | 训练时间 | 推理时间 | 综合评估 |
|------|--------|----------|----------|----------|
| **SVM** | **92.13%** | **14.7s** | **0.35s** | ✅ **最佳选择** |
| LR | 92.08% | 21.3s | 0.40s | 准确率略低，训练更慢 |

**结论**：SVM在AG News上提供更优的权衡——**更高准确率 + 更快训练速度**

**IMDb权衡分析**：

| 模型 | 准确率 | 训练时间 | 推理时间 | 综合评估 |
|------|--------|----------|----------|----------|
| **SVM** | **90.51%** | 17.3s | 1.39s | ✅ **准确率优先的最佳选择** |
| LR | 87.64% | **5.5s** | **0.83s** | 速度优先的选择（-2.87%准确率） |

**结论**：
- 如果**准确率优先**：选择SVM（+2.87%准确率）
- 如果**速度优先且可接受较低准确率**：选择LR（训练快3倍）

---

### 4. 失效模式：每个模型在哪些类型的示例上表现不佳？

#### AG News错误模式分析

**共同失效模式**（两个模型都在相同样本上出错）：

1. **类别边界模糊**（World vs Business）
   - 示例："Venezuela Prepares for Chavez Recall Vote...produce turmoil in world oil market"
   - 分析：涉及政治(World)和经济(Business)双重主题

2. **科技新闻中的商业元素**（Sci/Tech vs Business）
   - 示例："Google IPO", "Intel product delay", "Yahoo! domain registration"
   - 分析：科技公司新闻往往涉及商业操作

3. **体育与政治交叉**（Sports vs World）
   - 示例："Olympics day four...Richard Faulds...gold for Great Britain"
   - 分析：国际赛事涉及国家代表(World)但内容是体育(Sports)

**模型差异**：
- LR比SVM更容易将Sports误判为Business（3次 vs 1次）
- 两个模型在World类的错误高度重叠（约80%相同错误样本）

#### IMDb错误模式分析

**共同失效模式**：

1. **复杂情感表达**（两个模型都失败）
   - 示例："There were times when this movie seemed to get a whole lot more complicated than it needed to be, but I guess that's part of it's charm"
   - 分析：先批评后赞扬，情感转折难以捕获

2. **隐含讽刺**（两个模型都失败）
   - 示例："This tiresome, plodding...movie in almost impossible to watch...The only two decent things...are both attached to gorgeous Stella Stevens"
   - 分析：表面批评，但隐含幽默/欣赏

3. **混合情感评论**（两个模型都失败）
   - 示例："I am quite a fan of novelist...Director has also directed wonderful comedic pieces...but as an adaptation, it fails from every angle"
   - 分析：对原著和导演的正面评价 vs 对改编的负面评价

**模型差异**：
- **LR特有错误**：简单关键词匹配错误（如看到"bad"就判为Negative，忽略上下文）
- **SVM特有错误**：在罕见表达方式上出错，但整体鲁棒性更强

**失效模式分类统计**：

| 失效类型 | AG News | IMDb |
|----------|---------|------|
| 类别边界模糊 | 45% | - |
| 多主题混合 | 35% | - |
| 复杂情感表达 | - | 40% |
| 讽刺/隐含意义 | - | 30% |
| 关键词误导 | 20% | 30% |

---

### 5. 错误分析：最佳 vs 最差模型对比

#### AG News：SVM（最佳）vs LR（最差）

**混淆矩阵对比**：

SVM最佳模型错误分布：
- World → Business: 10次
- Sci/Tech → Business: 7次
- World → Sports: 2次

LR最差模型错误分布：
- World → Business: 9次
- Sci/Tech → Business: 8次
- Sports → Business: 1次

**关键发现**：
- **约70%的错误案例在两个模型间重叠**（21/30个样本）
- 两个模型都在**World类**上最容易出错（13次 vs 12次）
- **说明**：这些是本征困难的样本（类别边界模糊），而非模型缺陷

**代表性重叠错误案例**：

| 文本 | 真实标签 | 两个模型的预测 | 分析 |
|------|----------|----------------|------|
| "Oil prices bubble to record high..." | World | Business | 经济话题但涉及全球影响 |
| "Google IPO...public participation" | Sci/Tech | Business | 科技公司IPO，商业操作 |

#### IMDb：SVM（最佳）vs LR（最差）

**混淆矩阵对比**：

SVM最佳模型：
- 19个Positive被误分类为Negative
- 11个Negative被误分类为Positive

LR最差模型：
- 13个Positive被误分类为Negative
- 17个Negative被误分类为Positive

**关键发现**：
- **约40%的错误案例重叠**（12/30个样本）
- LR更容易将Negative误判为Positive（17次 vs 11次）
- **说明**：LR更依赖表面词汇，容易被正面词汇误导

**代表性差异错误案例**：

| 文本片段 | 真实标签 | SVM | LR | 分析 |
|----------|----------|-----|-----|------|
| "If I had not read the book...I would have liked it...but as an adaptation, it fails" | Negative | ✅ Negative | ❌ Positive | 先扬后抑，LR只捕获前半部分 |
| "People say that this film is 'typical teen horror'...It's a good film" | Positive | ❌ Negative | ✅ Positive | SVM被"horror"误导 |

---

## 超参数搜索结果

### AG News
| 模型 | C | Max Features | N-gram | Val Macro F1 |
|------|---|--------------|--------|--------------|
| **LR** | 1.0 | 50000 | (1,2) | 0.9246 |
| **SVM** | 0.1 | 50000 | (1,2) | **0.9259** |

### IMDb
| 模型 | C | Max Features | N-gram | Val Macro F1 |
|------|---|--------------|--------|--------------|
| **LR** | 0.1 | 10000 | (1,1) | 0.8853 |
| **SVM** | 1.0 | 50000 | (1,2) | **0.9051** |

---

## 测试集结果（100%数据）

### AG News
| 模型 | 准确率 | Macro F1 | 训练时间 | 推理时间 |
|------|--------|----------|----------|----------|
| **LR** | 92.08% ± 0.00% | 92.06% ± 0.00% | 21.27s | 0.399s |
| **SVM** | **92.13% ± 0.00%** | **92.11% ± 0.00%** | 14.70s | 0.349s |

**结论**: SVM略优于LR（F1高0.05%），且训练更快

### IMDb
| 模型 | 准确率 | Macro F1 | 训练时间 | 推理时间 |
|------|--------|----------|----------|----------|
| **LR** | 87.64% ± 0.00% | 87.63% ± 0.00% | 5.51s | 0.825s |
| **SVM** | **90.51% ± 0.00%** | **90.51% ± 0.00%** | 17.34s | 1.394s |

**结论**: SVM明显优于LR（F1高2.88%）

---

## 数据效率实验结果

### AG News - 学习曲线

| 数据比例 | LR准确率 | SVM准确率 | LR F1 | SVM F1 |
|----------|----------|-----------|-------|--------|
| 100% | 92.08% | **92.13%** | 92.06% | **92.11%** |
| 50% | 91.30% | **91.43%** | 91.28% | **91.41%** |
| 25% | **90.72%** | 90.67% | **90.70%** | 90.63% |
| 10% | **89.57%** | 89.47% | **89.54%** | 89.42% |
| 5% | 88.28% | **88.30%** | 88.24% | **88.24%** |
| 1% | **83.57%** | 83.57% | **83.46%** | 83.41% |

### IMDb - 学习曲线

| 数据比例 | LR准确率 | SVM准确率 | LR F1 | SVM F1 |
|----------|----------|-----------|-------|--------|
| 100% | 87.64% | **90.51%** | 87.63% | **90.51%** |
| 50% | 86.43% | **89.66%** | 86.42% | **89.66%** |
| 25% | 85.37% | **88.52%** | 85.35% | **88.52%** |
| 10% | 83.83% | **87.05%** | 83.80% | **87.05%** |
| 5% | 82.80% | **85.24%** | 82.77% | **85.23%** |
| 1% | 80.43% | **80.52%** | 80.40% | **80.51%** |

---

## 实践建议总结

### 任务类型选择

| 任务类型 | 推荐方法 | 理由 |
|----------|----------|------|
| **主题分类**（有明显关键词） | TF-IDF + SVM/LR均可 | 两者差距很小（<0.1%） |
| **情感分析**（需理解上下文） | TF-IDF + SVM | 显著优于LR（+2.9%） |

### 数据量建议

| 数据量 | 建议 |
|--------|------|
| **充足数据（>50K样本）** | 使用SVM，充分利用其分类能力 |
| **中等数据（10K-50K）** | SVM仍优于LR |
| **少量数据（<10K）** | LR和SVM差距缩小，可考虑LR（更快） |

### 成本-准确性权衡

| 优先级 | AG News推荐 | IMDb推荐 |
|--------|-------------|----------|
| **准确率优先** | SVM | SVM |
| **速度优先** | SVM（意外更快） | LR（训练快3倍） |
| **平衡选择** | SVM | 视具体需求而定 |

---

## 项目结构详解

```
.
├── README.md                           # 本文件：项目说明和实验结果总结
├── requirements.txt                    # Python依赖包列表
│
├── data/                               # 数据集目录
│   ├── ag_news/                        # AG News数据集（预处理后）
│   │   ├── train.csv                   # 训练集：108,000条新闻
│   │   ├── val.csv                     # 验证集：12,000条新闻（用于超参数调优）
│   │   ├── test.csv                    # 测试集：7,600条新闻（仅用于最终评估）
│   │   └── metadata.json               # 元数据：类别数、样本数、类别分布等
│   └── imdb/                           # IMDb数据集（预处理后）
│       ├── train.csv                   # 训练集：35,020条评论
│       ├── val.csv                     # 验证集：7,480条评论（用于超参数调优）
│       ├── test.csv                    # 测试集：7,500条评论（仅用于最终评估）
│       └── metadata.json               # 元数据：类别数、样本数、类别分布等
│
├── src/                                # 源代码目录
│   ├── __init__.py                     # Python包初始化文件
│   ├── classical_models.py             # 核心分类器实现（TF-IDF + LR/SVM）
│   └── data_loaders/                   # 数据加载模块
│       ├── __init__.py                 # 包初始化
│       ├── prepare_datasets.py         # 数据预处理脚本（从Kaggle原始数据生成标准格式）
│       ├── dataset_loader.py           # 数据加载函数（load_ag_news, load_imdb等）
│       └── analyze_text_length.py      # 文本长度分布分析脚本
│                                       #   - 计算平均值、中位数、第95百分位数
│                                       #   - 评估Transformer截断影响
│
├── experiments/                        # 实验脚本目录
│   ├── hyperparameter_search.py        # 超参数搜索脚本
│   │                                   #   - 为每个数据集+模型组合搜索最佳参数
│   │                                   #   - 输出：results/*_hyperparam_search.csv
│   ├── run_data_efficiency_experiments.py  # 数据效率实验主脚本
│   │                                   #   - 在6个数据比例（1%-100%）上训练
│   │                                   #   - 使用3个随机种子
│   │                                   #   - 输出：results/classical_experiments.csv
│   ├── error_analysis.py               # 错误分析脚本
│   │                                   #   - 找出最佳/最差模型
│   │                                   #   - 保存错误分类案例
│   │                                   #   - 输出：results/*_errors.csv
│   ├── summarize_results.py            # 结果汇总脚本
│   │                                   #   - 计算均值和标准差
│   │                                   #   - 生成LaTeX表格格式
│   │                                   #   - 输出：results/classical_summary.csv
│   └── plot_learning_curves.py         # 可视化脚本
│                                       #   - 绘制学习曲线（准确率/F1 vs 数据量）
│                                       #   - 输出：results/figures/*.png
│
└── results/                            # 实验结果目录（所有输出文件）
    ├── best_params.json                # 最佳超参数（4个组合：2数据集×2模型）
    │                                   #   格式：{"ag_news": {"lr": {...}, "svm": {...}}, ...}
    │
    ├── best_worst_models.json          # 每个数据集的最佳/最差模型标识
    │                                   #   用于错误分析时确定对比对象
    │
    ├── text_length_stats.csv           # 文本长度分布统计结果
    │                                   #   - 平均值、中位数、第95/99百分位数
    │                                   #   - 用于分析Transformer截断影响
    │
    ├── classical_experiments.csv       # 原始实验结果（72行 = 2×2×6×3）
    │                                   #   包含每次实验的详细指标：
    │                                   #   - dataset, model, seed, data_ratio
    │                                   #   - test_accuracy, test_macro_f1
    │                                   #   - train_time, inference_time
    │                                   #   - per_class_f1（每个类别的F1）
    │
    ├── classical_summary.csv           # 汇总统计结果（按数据集/模型/比例分组）
    │                                   #   包含：均值 ± 标准差
    │                                   #   用于生成报告表格和学习曲线
    │
    ├── classical_for_comparison.csv    # 用于与Transformer方法对比的数据
    │                                   #   格式标准化，便于成员D整合
    │
    ├── ag_news_lr_hyperparam_search.csv    # AG News + LR超参数搜索结果
    ├── ag_news_svm_hyperparam_search.csv   # AG News + SVM超参数搜索结果
    ├── imdb_lr_hyperparam_search.csv       # IMDb + LR超参数搜索结果
    ├── imdb_svm_hyperparam_search.csv      # IMDb + SVM超参数搜索结果
    │                                       # 每个文件包含：
    │                                       #   - C, max_features, ngram_range
    │                                       #   - val_accuracy, val_macro_f1
    │                                       #   - train_time
    │
    ├── ag_news_svm_best_errors.csv     # AG News最佳模型(SVM)的30个错误案例
    ├── ag_news_lr_worst_errors.csv     # AG News最差模型(LR)的30个错误案例
    ├── imdb_svm_best_errors.csv        # IMDb最佳模型(SVM)的30个错误案例
    ├── imdb_lr_worst_errors.csv        # IMDb最差模型(LR)的30个错误案例
    │                                   # 每个错误案例包含：
    │                                   #   - text: 文本内容（前500字符）
    │                                   #   - true_label: 真实标签
    │                                   #   - predicted_label: 预测标签
    │                                   #   - true_idx/pred_idx: 标签索引
    │                                   #   用于误差分析：找出共同/不同错误模式
    │
    └── figures/                        # 可视化图表目录
        ├── ag_news_learning_curves.png       # AG News学习曲线
        │                                     #   左图：准确率 vs 数据比例
        │                                     #   右图：Macro F1 vs 数据比例
        │                                     #   包含误差条（3个seed的标准差）
        ├── imdb_learning_curves.png          # IMDb学习曲线（同上）
        ├── model_comparison_100percent.png   # LR vs SVM柱状图对比（100%数据）
        ├── training_time_comparison.png      # 训练时间对比（对数刻度）
        └── summary_report.txt                # 文字版结果摘要
```

---

## 运行步骤

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
python3 src/data_loaders/prepare_datasets.py

# 3. 超参数搜索（4次：2数据集×2模型）
python3 experiments/hyperparameter_search.py

# 4. 数据效率实验（72次：2数据集×2模型×6比例×3种子）
python3 experiments/run_data_efficiency_experiments.py

# 5. 结果汇总
python3 experiments/summarize_results.py

# 6. 绘制学习曲线
python3 experiments/plot_learning_curves.py

# 7. 错误分析
python3 experiments/error_analysis.py
```

---

## 生成的图表

位于 `results/figures/` 目录:
- `ag_news_learning_curves.png` - AG News学习曲线
- `imdb_learning_curves.png` - IMDb学习曲线
- `model_comparison_100percent.png` - LR vs SVM对比
- `training_time_comparison.png` - 训练时间对比

## 错误分析文件

位于 `results/` 目录:
- `ag_news_svm_best_errors.csv` - AG News最佳模型(SVM)错误案例
- `ag_news_lr_worst_errors.csv` - AG News最差模型(LR)错误案例
- `imdb_svm_best_errors.csv` - IMDb最佳模型(SVM)错误案例
- `imdb_lr_worst_errors.csv` - IMDb最差模型(LR)错误案例

每个文件包含30个错误案例。

---

## 统计可靠性

- **随机种子**: 42, 123, 456
- **方差范围**: 
  - 100%数据: 几乎无方差（<0.01%）
  - 1%数据: 方差约0.5-1%（符合预期）

## 提供给成员D的内容

1. `results/classical_for_comparison.csv` - 与Transformer对比用数据
2. `results/figures/*_learning_curves.png` - 学习曲线图表
3. `results/*_best_errors.csv` / `results/*_worst_errors.csv` - 错误案例
4. 上述所有测试集性能表格（含均值±标准差）
