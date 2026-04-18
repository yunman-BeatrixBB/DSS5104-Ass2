"""
绘制学习曲线
准确率/F1值 vs 训练集大小的关系图
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size'] = 11


def plot_learning_curves(results_csv='results/classical_experiments.csv',
                         output_dir='results/figures'):
    """
    绘制学习曲线：准确率和Macro F1 vs 数据比例

    参数:
        results_csv: 实验结果CSV文件路径
        output_dir: 输出图片目录
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 读取数据
    df = pd.read_csv(results_csv)

    # 计算每个数据集、模型、数据比例的均值和标准差
    summary = df.groupby(['dataset', 'model', 'data_ratio']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_macro_f1': ['mean', 'std'],
        'train_size': 'mean'
    }).reset_index()

    # 展平列名
    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns.values]

    # 为每个数据集绘制图表
    for dataset in summary['dataset'].unique():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        dataset_data = summary[summary['dataset'] == dataset]

        # 获取数据集名称显示
        dataset_display = dataset.upper()

        # 颜色映射
        colors = {'tfidf_lr': '#2E86AB', 'tfidf_svm': '#A23B72'}
        model_names = {'tfidf_lr': 'TF-IDF + Logistic Regression',
                       'tfidf_svm': 'TF-IDF + SVM'}

        # 左图：准确率
        ax1 = axes[0]
        for model in ['tfidf_lr', 'tfidf_svm']:
            model_data = dataset_data[dataset_data['model'] == model]
            if len(model_data) == 0:
                continue

            # 按数据比例排序
            model_data = model_data.sort_values('data_ratio')

            x = model_data['data_ratio'] * 100  # 转换为百分比
            y = model_data['test_accuracy_mean']
            yerr = model_data['test_accuracy_std']

            ax1.errorbar(x, y, yerr=yerr, marker='o', markersize=8,
                        linewidth=2, capsize=5, capthick=2,
                        label=model_names[model], color=colors[model])

        ax1.set_xlabel('Training Data Ratio (%)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title(f'{dataset_display} - Accuracy vs Training Data Size',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')  # 使用对数刻度更好地显示小比例数据
        ax1.set_xticks([1, 5, 10, 25, 50, 100])
        ax1.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])
        ax1.set_ylim([0.5, 1.0])

        # 右图：Macro F1
        ax2 = axes[1]
        for model in ['tfidf_lr', 'tfidf_svm']:
            model_data = dataset_data[dataset_data['model'] == model]
            if len(model_data) == 0:
                continue

            # 按数据比例排序
            model_data = model_data.sort_values('data_ratio')

            x = model_data['data_ratio'] * 100  # 转换为百分比
            y = model_data['test_macro_f1_mean']
            yerr = model_data['test_macro_f1_std']

            ax2.errorbar(x, y, yerr=yerr, marker='s', markersize=8,
                        linewidth=2, capsize=5, capthick=2,
                        label=model_names[model], color=colors[model])

        ax2.set_xlabel('Training Data Ratio (%)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title(f'{dataset_display} - Macro F1 vs Training Data Size',
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        ax2.set_xticks([1, 5, 10, 25, 50, 100])
        ax2.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])
        ax2.set_ylim([0.5, 1.0])

        plt.tight_layout()

        # 保存图片
        output_file = f'{output_dir}/{dataset}_learning_curves.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")

        plt.close()

    print(f"\nAll learning curves saved to {output_dir}/")


def plot_model_comparison_bar(results_csv='results/classical_experiments.csv',
                              output_dir='results/figures'):
    """
    绘制柱状图对比LR和SVM在不同数据比例上的性能
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(results_csv)

    # 只取100%数据的结果
    full_data = df[df['data_ratio'] == 1.0]

    # 计算均值和标准差
    summary = full_data.groupby(['dataset', 'model']).agg({
        'test_accuracy': ['mean', 'std'],
        'test_macro_f1': ['mean', 'std']
    }).reset_index()

    summary.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                       for col in summary.columns.values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = summary['dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.35

    model_names = {'tfidf_lr': 'LR', 'tfidf_svm': 'SVM'}
    colors = {'tfidf_lr': '#2E86AB', 'tfidf_svm': '#A23B72'}

    # 左图：准确率对比
    ax1 = axes[0]
    for i, model in enumerate(['tfidf_lr', 'tfidf_svm']):
        model_data = summary[summary['model'] == model]
        means = model_data['test_accuracy_mean'].values
        stds = model_data['test_accuracy_std'].values

        ax1.bar(x + i*width, means, width, yerr=stds,
               label=model_names[model], color=colors[model],
               capsize=5, alpha=0.8)

    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Comparison - Accuracy (100% Data)',
                 fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([d.upper() for d in datasets])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0.8, 0.95])

    # 右图：Macro F1对比
    ax2 = axes[1]
    for i, model in enumerate(['tfidf_lr', 'tfidf_svm']):
        model_data = summary[summary['model'] == model]
        means = model_data['test_macro_f1_mean'].values
        stds = model_data['test_macro_f1_std'].values

        ax2.bar(x + i*width, means, width, yerr=stds,
               label=model_names[model], color=colors[model],
               capsize=5, alpha=0.8)

    ax2.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Macro F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model Comparison - Macro F1 (100% Data)',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels([d.upper() for d in datasets])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.8, 0.95])

    plt.tight_layout()

    output_file = f'{output_dir}/model_comparison_100percent.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()


def plot_training_time_comparison(results_csv='results/classical_experiments.csv',
                                  output_dir='results/figures'):
    """
    绘制训练时间对比图
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(results_csv)

    # 计算每个数据集、模型、数据比例的平均训练时间
    summary = df.groupby(['dataset', 'model', 'data_ratio']).agg({
        'train_time': 'mean',
        'inference_time': 'mean'
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'tfidf_lr': '#2E86AB', 'tfidf_svm': '#A23B72'}
    model_names = {'tfidf_lr': 'TF-IDF + LR', 'tfidf_svm': 'TF-IDF + SVM'}

    for idx, dataset in enumerate(summary['dataset'].unique()):
        ax = axes[idx]
        dataset_data = summary[summary['dataset'] == dataset]

        for model in ['tfidf_lr', 'tfidf_svm']:
            model_data = dataset_data[dataset_data['model'] == model]
            if len(model_data) == 0:
                continue

            model_data = model_data.sort_values('data_ratio')

            x = model_data['data_ratio'] * 100
            y = model_data['train_time']

            ax.plot(x, y, marker='o', markersize=8, linewidth=2,
                   label=model_names[model], color=colors[model])

        ax.set_xlabel('Training Data Ratio (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{dataset.upper()} - Training Time',
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([1, 5, 10, 25, 50, 100])
        ax.set_xticklabels(['1%', '5%', '10%', '25%', '50%', '100%'])

    plt.tight_layout()

    output_file = f'{output_dir}/training_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()


def generate_summary_report(results_csv='results/classical_experiments.csv',
                            output_file='results/figures/summary_report.txt'):
    """
    生成文字摘要报告
    """
    df = pd.read_csv(results_csv)

    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSICAL METHODS - LEARNING CURVE SUMMARY\n")
        f.write("="*70 + "\n\n")

        for dataset in df['dataset'].unique():
            f.write(f"\n{dataset.upper()}\n")
            f.write("-"*70 + "\n")

            dataset_df = df[df['dataset'] == dataset]

            for model in ['tfidf_lr', 'tfidf_svm']:
                model_df = dataset_df[dataset_df['model'] == model]
                model_name = "Logistic Regression" if model == 'tfidf_lr' else "SVM"

                f.write(f"\n{model_name}:\n")

                # 按数据比例排序
                for ratio in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
                    ratio_df = model_df[model_df['data_ratio'] == ratio]

                    if len(ratio_df) > 0:
                        acc_mean = ratio_df['test_accuracy'].mean()
                        acc_std = ratio_df['test_accuracy'].std()
                        f1_mean = ratio_df['test_macro_f1'].mean()
                        f1_std = ratio_df['test_macro_f1'].std()
                        train_time = ratio_df['train_time'].mean()

                        f.write(f"  {ratio*100:>5.0f}% data: "
                               f"Acc={acc_mean:.4f}±{acc_std:.4f}, "
                               f"F1={f1_mean:.4f}±{f1_std:.4f}, "
                               f"Time={train_time:.2f}s\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Saved summary report: {output_file}")


if __name__ == '__main__':
    print("Generating learning curve plots...\n")

    # 绘制学习曲线
    plot_learning_curves()

    # 绘制模型对比柱状图
    plot_model_comparison_bar()

    # 绘制训练时间对比
    plot_training_time_comparison()

    # 生成文字报告
    generate_summary_report()

    print("\n" + "="*70)
    print("All plots generated successfully!")
    print("="*70)
