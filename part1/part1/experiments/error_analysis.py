"""
错误分析脚本
找出每个数据集上表现最佳和最差的模型，保存它们的错误分类案例
"""
import sys
sys.path.insert(0, 'src')

import json
import pandas as pd
import numpy as np
from data_loaders import load_ag_news, load_imdb
from classical_models import ClassicalTextClassifier


def save_error_cases_for_model(dataset_name, data_loader, model_type, hyperparams,
                                seed=42, max_errors=30, model_label=""):
    """
    为指定模型保存错误分类案例

    参数:
        dataset_name: 数据集名称
        data_loader: 数据加载函数
        model_type: 模型类型 ('lr' 或 'svm')
        hyperparams: 超参数字典
        seed: 随机种子
        max_errors: 最多保存的错误案例数
        model_label: 模型标签（"best"或"worst"）
    """
    print(f"\n{'='*70}")
    print(f"Error Analysis: {dataset_name.upper()} - {model_type.upper()} ({model_label})")
    print(f"{'='*70}")

    # 加载数据
    data = data_loader()
    train_texts, train_labels = data['train']
    test_texts, test_labels = data['test']
    class_names = data['class_names']

    print(f"Training on {len(train_texts)} samples...")

    # 训练模型
    clf = ClassicalTextClassifier(
        model_type=model_type,
        C=hyperparams['C'],
        max_features=hyperparams['max_features'],
        ngram_range=hyperparams['ngram_range'],
        random_state=seed
    )
    clf.fit(train_texts, train_labels)

    # 预测
    predictions = clf.predict(test_texts)

    # 计算性能指标
    from sklearn.metrics import accuracy_score, f1_score
    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average='macro')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    # 收集错误案例
    errors = []
    for i, (text, true, pred) in enumerate(zip(test_texts, test_labels, predictions)):
        if true != pred:
            errors.append({
                'index': i,
                'text': text[:500] if len(text) > 500 else text,  # 截取前500字符
                'true_label': class_names[true],
                'predicted_label': class_names[pred],
                'true_idx': int(true),
                'pred_idx': int(pred),
                'dataset': dataset_name,
                'model': f"tfidf_{model_type}",
                'model_label': model_label,
                'accuracy': accuracy,
                'macro_f1': macro_f1
            })
        if len(errors) >= max_errors:
            break

    # 保存为CSV
    output_file = f'results/{dataset_name}_{model_type}_{model_label}_errors.csv'
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(output_file, index=False)

    print(f"Saved {len(errors)} error cases to {output_file}")

    # 打印一些错误案例
    print(f"\nSample Error Cases ({model_label} model):")
    print("-"*70)
    for i, error in enumerate(errors[:5]):
        print(f"\n{i+1}. True: {error['true_label']} | Pred: {error['predicted_label']}")
        print(f"Text: {error['text'][:200]}...")
        print("-"*70)

    return errors_df, accuracy, macro_f1


def analyze_error_patterns(dataset_name, model_type, model_label):
    """
    分析错误模式，按类别对统计
    """
    errors_df = pd.read_csv(f'results/{dataset_name}_{model_type}_{model_label}_errors.csv')

    print(f"\n{'='*70}")
    print(f"Error Pattern Analysis: {dataset_name.upper()} - {model_type.upper()} ({model_label})")
    print(f"{'='*70}")

    # 统计混淆矩阵
    confusion = pd.crosstab(
        errors_df['true_label'],
        errors_df['predicted_label'],
        margins=True
    )
    print("\nError Confusion Matrix:")
    print(confusion)

    # 按真实类别统计错误数
    print("\nErrors by True Class:")
    print(errors_df['true_label'].value_counts())

    return confusion


def find_best_and_worst_models(dataset_name):
    """
    从实验结果中找出最佳和最差模型
    """
    # 读取实验结果
    df = pd.read_csv('results/classical_experiments.csv')

    # 筛选该数据集在100%数据上的结果
    dataset_results = df[(df['dataset'] == dataset_name) & (df['data_ratio'] == 1.0)]

    # 按模型分组，计算平均macro_f1
    model_performance = dataset_results.groupby('model')['test_macro_f1'].mean().sort_values(ascending=False)

    print(f"\n{'='*70}")
    print(f"Model Performance on {dataset_name.upper()} (100% data, averaged over seeds)")
    print(f"{'='*70}")
    print(model_performance)

    best_model = model_performance.index[0].replace('tfidf_', '')
    worst_model = model_performance.index[-1].replace('tfidf_', '')

    print(f"\nBest model: {best_model.upper()} (Macro F1: {model_performance.iloc[0]:.4f})")
    print(f"Worst model: {worst_model.upper()} (Macro F1: {model_performance.iloc[-1]:.4f})")

    return best_model, worst_model


def main():
    """主函数"""
    # 加载最佳参数
    with open('results/best_params.json', 'r') as f:
        best_params = json.load(f)

    datasets = {
        'ag_news': load_ag_news,
        'imdb': load_imdb
    }

    best_worst_summary = {}

    for dataset_name, data_loader in datasets.items():
        print("\n" + "="*70)
        print(f"ANALYZING DATASET: {dataset_name.upper()}")
        print("="*70)

        # 找出最佳和最差模型
        best_model, worst_model = find_best_and_worst_models(dataset_name)

        best_worst_summary[dataset_name] = {
            'best_model': best_model,
            'worst_model': worst_model
        }

        # 保存最佳模型的错误案例
        save_error_cases_for_model(
            dataset_name, data_loader, best_model,
            best_params[dataset_name][best_model],
            seed=42, max_errors=30, model_label="best"
        )
        analyze_error_patterns(dataset_name, best_model, "best")

        # 保存最差模型的错误案例
        save_error_cases_for_model(
            dataset_name, data_loader, worst_model,
            best_params[dataset_name][worst_model],
            seed=42, max_errors=30, model_label="worst"
        )
        analyze_error_patterns(dataset_name, worst_model, "worst")

    # 保存最佳/最差模型汇总
    with open('results/best_worst_models.json', 'w') as f:
        json.dump(best_worst_summary, f, indent=2)

    print("\n" + "="*70)
    print("Error analysis completed!")
    print("="*70)
    print("\nSummary of Best/Worst models saved to: results/best_worst_models.json")


if __name__ == '__main__':
    main()
