"""
数据效率实验主脚本
在1%, 5%, 10%, 25%, 50%, 100%的训练数据上运行所有模型
使用3个随机种子计算均值和标准差
"""
import sys
sys.path.insert(0, 'src')

import json
import pandas as pd
import numpy as np
import os
from data_loaders import load_ag_news, load_imdb, stratified_subsample
from classical_models import ClassicalTextClassifier


def run_single_experiment(dataset_name, data_loader, model_type, data_ratio, hyperparams, seed=42):
    """
    运行单次实验

    参数:
        dataset_name: 数据集名称
        data_loader: 数据加载函数
        model_type: 模型类型 ('lr' 或 'svm')
        data_ratio: 数据比例 (0.01 = 1%)
        hyperparams: 超参数字典
        seed: 随机种子

    返回:
        dict: 实验结果
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name} | Model: {model_type.upper()} | Ratio: {data_ratio*100:.1f}% | Seed: {seed}")
    print(f"{'='*70}")

    # 设置随机种子
    np.random.seed(seed)

    # 加载数据
    data = data_loader()
    train_texts_full, train_labels_full = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    class_names = data['class_names']

    # 分层抽样
    train_texts, train_labels = stratified_subsample(
        train_texts_full, train_labels_full, data_ratio, seed
    )

    print(f"Training samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # 训练模型
    clf = ClassicalTextClassifier(
        model_type=model_type,
        C=hyperparams['C'],
        max_features=hyperparams['max_features'],
        ngram_range=hyperparams['ngram_range'],
        random_state=seed
    )

    clf.fit(train_texts, train_labels)

    # 在验证集上评估（用于监控）
    val_metrics = clf.evaluate(val_texts, val_labels, "Validation", verbose=False)

    # 在测试集上评估（最终结果）
    test_metrics = clf.evaluate(test_texts, test_labels, "Test", verbose=True)

    # 构建结果字典
    result = {
        'dataset': dataset_name,
        'model': f"tfidf_{model_type}",
        'seed': seed,
        'data_ratio': data_ratio,
        'train_size': len(train_texts),
        'hyperparams': json.dumps(hyperparams),

        # 验证集指标（开发用）
        'val_accuracy': val_metrics['accuracy'],
        'val_macro_f1': val_metrics['macro_f1'],

        # 测试集指标（报告用）
        'test_accuracy': test_metrics['accuracy'],
        'test_macro_f1': test_metrics['macro_f1'],
        'per_class_f1': json.dumps(test_metrics['per_class_f1']),

        # 时间和资源
        'train_time': test_metrics['train_time'],
        'inference_time': test_metrics['inference_time'],
        'throughput': test_metrics['throughput'],
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

    # 计算总实验数: 2数据集 × 2模型 × 6比例 × 3种子 = 72
    total_experiments = len(datasets) * 2 * len(data_ratios) * len(seeds)
    current = 0

    # 遍历所有组合
    for dataset_name, data_loader in datasets.items():
        for model_type in ['lr', 'svm']:
            hyperparams = best_params[dataset_name][model_type]

            for data_ratio in data_ratios:
                for seed in seeds:
                    current += 1
                    print(f"\n\n{'#'*70}")
                    print(f"# Progress: {current}/{total_experiments} experiments")
                    print(f"{'#'*70}")

                    try:
                        result = run_single_experiment(
                            dataset_name, data_loader, model_type, data_ratio,
                            hyperparams, seed
                        )
                        all_results.append(result)

                        # 每完成一个实验就保存，防止丢失
                        results_df = pd.DataFrame(all_results)
                        results_df.to_csv('results/classical_experiments.csv', index=False)
                        print(f"\nSaved to results/classical_experiments.csv")

                    except Exception as e:
                        print(f"\nERROR in experiment: {e}")
                        import traceback
                        traceback.print_exc()
                        continue

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"Results saved to: results/classical_experiments.csv")
    print(f"Total experiments: {len(all_results)}")

    return pd.DataFrame(all_results)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    results_df = run_all_experiments()
