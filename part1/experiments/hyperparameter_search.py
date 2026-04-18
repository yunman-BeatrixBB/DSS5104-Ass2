"""
超参数搜索脚本
对AG News和IMDb数据集，分别在LR和SVM上搜索最佳超参数
"""
import sys
sys.path.insert(0, 'src')

import json
import pandas as pd
import numpy as np
from data_loaders import load_ag_news, load_imdb
from classical_models import ClassicalTextClassifier


def hyperparameter_search_single_model(dataset_name, data_loader, model_type, random_state=42):
    """
    对给定数据集的单个模型进行超参数搜索

    搜索空间:
    - C: [0.1, 1.0, 10.0]
    - max_features: [10000, 50000]
    - ngram_range: [(1, 1), (1, 2)]
    """
    print(f"\n{'='*70}")
    print(f"Hyperparameter Search: {dataset_name.upper()} + {model_type.upper()}")
    print(f"{'='*70}")

    # 加载数据
    data = data_loader()
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    class_names = data['class_names']

    print(f"Train size: {len(train_texts)}")
    print(f"Val size: {len(val_texts)}")
    print(f"Classes: {class_names}")

    # 超参数搜索空间
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'max_features': [10000, 50000],
        'ngram_range': [(1, 1), (1, 2)]
    }

    results = []
    best_macro_f1 = 0
    best_params = None
    total_combinations = len(param_grid['C']) * len(param_grid['max_features']) * len(param_grid['ngram_range'])

    print(f"\nTotal combinations to try: {total_combinations}")
    print(f"{'='*70}")

    # 遍历所有超参数组合
    for C in param_grid['C']:
        for max_features in param_grid['max_features']:
            for ngram_range in param_grid['ngram_range']:
                print(f"\nTrying: {model_type.upper()}, C={C}, max_features={max_features}, ngram={ngram_range}")

                try:
                    # 训练模型
                    clf = ClassicalTextClassifier(
                        model_type=model_type,
                        C=C,
                        max_features=max_features,
                        ngram_range=ngram_range,
                        random_state=random_state
                    )
                    clf.fit(train_texts, train_labels)

                    # 在验证集上评估
                    metrics = clf.evaluate(val_texts, val_labels, "Validation", verbose=False)

                    result = {
                        'dataset': dataset_name,
                        'model_type': model_type,
                        'C': C,
                        'max_features': max_features,
                        'ngram_range': str(ngram_range),
                        'val_accuracy': metrics['accuracy'],
                        'val_macro_f1': metrics['macro_f1'],
                        'train_time': metrics['train_time']
                    }
                    results.append(result)

                    print(f"  Val Macro F1: {metrics['macro_f1']:.4f} | "
                          f"Accuracy: {metrics['accuracy']:.4f} | "
                          f"Time: {metrics['train_time']:.2f}s")

                    # 更新最佳参数
                    if metrics['macro_f1'] > best_macro_f1:
                        best_macro_f1 = metrics['macro_f1']
                        best_params = {
                            'model_type': model_type,
                            'C': C,
                            'max_features': max_features,
                            'ngram_range': ngram_range
                        }
                        print(f"  *** New best for {model_type.upper()}! ***")

                except Exception as e:
                    print(f"  ERROR: {e}")
                    continue

    # 保存详细结果
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_macro_f1', ascending=False)
    results_df.to_csv(f'results/{dataset_name}_{model_type}_hyperparam_search.csv', index=False)

    # 打印最佳参数
    print(f"\n{'='*70}")
    print(f"BEST PARAMETERS for {dataset_name.upper()} + {model_type.upper()}:")
    print(f"{'='*70}")
    print(f"C: {best_params['C']}")
    print(f"Max Features: {best_params['max_features']}")
    print(f"N-gram Range: {best_params['ngram_range']}")
    print(f"Validation Macro F1: {best_macro_f1:.4f}")
    print(f"{'='*70}")

    return best_params, results_df


def main():
    """主函数：对每个数据集的两个模型分别进行超参数搜索"""
    import os
    os.makedirs('results', exist_ok=True)

    all_best_params = {}
    datasets = {
        'ag_news': load_ag_news,
        'imdb': load_imdb
    }
    models = ['lr', 'svm']

    # 对每个数据集和每个模型进行搜索
    for dataset_name, data_loader in datasets.items():
        all_best_params[dataset_name] = {}

        for model_type in models:
            print("\n" + "="*70)
            print(f"STARTING: {dataset_name.upper()} + {model_type.upper()}")
            print("="*70)

            best_params, results = hyperparameter_search_single_model(
                dataset_name, data_loader, model_type
            )
            all_best_params[dataset_name][model_type] = best_params

    # 保存所有最佳参数
    with open('results/best_params.json', 'w') as f:
        json.dump(all_best_params, f, indent=2)

    # 打印汇总
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH COMPLETED")
    print("="*70)
    print("\nBest parameters saved to: results/best_params.json")
    print("\nSummary:")
    for dataset, models_params in all_best_params.items():
        print(f"\n{dataset.upper()}:")
        for model, params in models_params.items():
            print(f"  {model.upper()}:")
            print(f"    C: {params['C']}, max_features: {params['max_features']}, ngram: {params['ngram_range']}")


if __name__ == '__main__':
    main()
