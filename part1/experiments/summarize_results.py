"""
结果汇总脚本
计算均值和标准差，生成报告用表格
对比两个模型（LR和SVM）的性能
"""
import pandas as pd
import numpy as np
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
        'inference_time': ['mean', 'std'],
        'throughput': ['mean', 'std']
    }).reset_index()

    # 展平列名
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns.values]

    # 保存汇总结果
    summary.to_csv('results/classical_summary.csv', index=False)

    # 按数据集打印结果
    for dataset in summary['dataset'].unique():
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {dataset.upper()}")
        print(f"{'='*80}")

        # 分别打印LR和SVM的结果
        for model in ['tfidf_lr', 'tfidf_svm']:
            sub = summary[(summary['dataset'] == dataset) &
                         (summary['model'] == model)].sort_values('data_ratio', ascending=False)

            if len(sub) == 0:
                continue

            model_name = "Logistic Regression" if model == 'tfidf_lr' else "SVM"
            print(f"\n{model_name}:")
            print(f"{'Ratio':<8} {'Accuracy':<20} {'Macro F1':<20} {'Train Time':<15} {'Inf Time':<15}")
            print("-"*80)

            for _, row in sub.iterrows():
                ratio = row['data_ratio']
                acc_mean = row['test_accuracy_mean']
                acc_std = row['test_accuracy_std']
                f1_mean = row['test_macro_f1_mean']
                f1_std = row['test_macro_f1_std']
                train_time = row['train_time_mean']
                inf_time = row['inference_time_mean']

                print(f"{ratio*100:>6.0f}%  "
                      f"{acc_mean:.4f} ± {acc_std:.4f}    "
                      f"{f1_mean:.4f} ± {f1_std:.4f}    "
                      f"{train_time:>6.2f}s        "
                      f"{inf_time:>6.3f}s")

    # 找出每个数据集上表现最好和最差的模型
    print("\n\n" + "="*80)
    print("BEST AND WORST MODELS (100% data)")
    print("="*80)

    for dataset in summary['dataset'].unique():
        dataset_full = summary[(summary['dataset'] == dataset) &
                              (summary['data_ratio'] == 1.0)]

        # 按macro_f1排序
        best_idx = dataset_full['test_macro_f1_mean'].idxmax()
        worst_idx = dataset_full['test_macro_f1_mean'].idxmin()

        best_row = dataset_full.loc[best_idx]
        worst_row = dataset_full.loc[worst_idx]

        print(f"\n{dataset.upper()}:")
        print(f"  Best:  {best_row['model']} (Macro F1: {best_row['test_macro_f1_mean']:.4f})")
        print(f"  Worst: {worst_row['model']} (Macro F1: {worst_row['test_macro_f1_mean']:.4f})")

    return summary


def generate_latex_tables(summary):
    """
    生成LaTeX表格格式
    """
    print("\n\n" + "="*80)
    print("LATEX TABLE FORMAT")
    print("="*80)

    for dataset in summary['dataset'].unique():
        print(f"\n% {dataset.upper()} Results")

        for model in ['tfidf_lr', 'tfidf_svm']:
            sub = summary[(summary['dataset'] == dataset) &
                         (summary['model'] == model)].sort_values('data_ratio', ascending=False)

            if len(sub) == 0:
                continue

            model_name = "LR" if model == 'tfidf_lr' else "SVM"

            print(f"\n% {model_name}")
            print("\\begin{table}[h]")
            print("\\centering")
            print("\\begin{tabular}{c|cc|cc}")
            print("\\hline")
            print("Data Ratio & Accuracy & Macro F1 & Train Time & Inf Time \\\\")
            print("\\hline")

            for _, row in sub.iterrows():
                ratio = int(row['data_ratio'] * 100)
                acc_mean = row['test_accuracy_mean']
                acc_std = row['test_accuracy_std']
                f1_mean = row['test_macro_f1_mean']
                f1_std = row['test_macro_f1_std']
                train_time = row['train_time_mean']
                inf_time = row['inference_time_mean']

                print(f"{ratio}\\% & ${acc_mean:.3f} \\pm {acc_std:.3f}$ & "
                      f"${f1_mean:.3f} \\pm {f1_std:.3f}$ & "
                      f"{train_time:.1f}s & {inf_time:.2f}s \\\\")

            print("\\hline")
            print("\\end{tabular}")
            print(f"\\caption{{{dataset.upper()} - {model_name} Results}}")
            print("\\end{table}")


def export_for_comparison():
    """
    导出数据用于与Transformer方法对比
    """
    summary = pd.read_csv('results/classical_summary.csv')

    # 创建对比用CSV
    comparison_data = []

    for _, row in summary.iterrows():
        model_name = "TF-IDF + LR" if row['model'] == 'tfidf_lr' else "TF-IDF + SVM"
        comparison_data.append({
            'dataset': row['dataset'],
            'method': model_name,
            'data_ratio': row['data_ratio'],
            'accuracy_mean': row['test_accuracy_mean'],
            'accuracy_std': row['test_accuracy_std'],
            'macro_f1_mean': row['test_macro_f1_mean'],
            'macro_f1_std': row['test_macro_f1_std'],
            'train_time_mean': row['train_time_mean'],
            'train_time_std': row['train_time_std'],
            'inference_time_mean': row['inference_time_mean'],
            'throughput_mean': row['throughput_mean']
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('results/classical_for_comparison.csv', index=False)

    print("\n\nExported for comparison: results/classical_for_comparison.csv")

    return comparison_df


def generate_model_comparison_table():
    """
    生成LR vs SVM的直接对比表格
    """
    df = pd.read_csv('results/classical_experiments.csv')

    print("\n\n" + "="*80)
    print("LR vs SVM DIRECT COMPARISON (100% data)")
    print("="*80)

    # 只取100%数据的结果
    full_data = df[df['data_ratio'] == 1.0]

    comparison = []
    for dataset in full_data['dataset'].unique():
        dataset_results = full_data[full_data['dataset'] == dataset]

        for model in ['tfidf_lr', 'tfidf_svm']:
            model_results = dataset_results[dataset_results['model'] == model]

            if len(model_results) > 0:
                acc_mean = model_results['test_accuracy'].mean()
                acc_std = model_results['test_accuracy'].std()
                f1_mean = model_results['test_macro_f1'].mean()
                f1_std = model_results['test_macro_f1'].std()
                train_time_mean = model_results['train_time'].mean()
                inf_time_mean = model_results['inference_time'].mean()

                comparison.append({
                    'dataset': dataset,
                    'model': model,
                    'accuracy': f"{acc_mean:.4f} ± {acc_std:.4f}",
                    'macro_f1': f"{f1_mean:.4f} ± {f1_std:.4f}",
                    'train_time': f"{train_time_mean:.2f}s",
                    'inf_time': f"{inf_time_mean:.3f}s"
                })

    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))

    return comparison_df


if __name__ == '__main__':
    summary = summarize_results()
    generate_latex_tables(summary)
    comparison = export_for_comparison()
    generate_model_comparison_table()

    print("\n\n" + "="*80)
    print("Summary saved to: results/classical_summary.csv")
    print("="*80)
