"""
文本长度分布分析
计算数据集中文本长度的统计信息（平均值、中位数、第95百分位数）
使用空格分词估算token数量
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import json
from src.data_loaders import load_ag_news, load_imdb


def analyze_text_length(texts, dataset_name, split_name):
    """
    分析文本长度分布

    参数:
        texts: 文本列表
        dataset_name: 数据集名称
        split_name: 数据划分名称（train/val/test）

    返回:
        dict: 统计信息
    """
    # 使用空格分词估算token数量
    lengths = [len(text.split()) for text in texts]
    lengths = np.array(lengths)

    stats = {
        'dataset': dataset_name,
        'split': split_name,
        'num_samples': len(lengths),
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        '95th_percentile': float(np.percentile(lengths, 95)),
        '99th_percentile': float(np.percentile(lengths, 99)),
    }

    return stats


def main():
    """主函数：分析两个数据集的文本长度分布"""

    all_stats = []

    print("=" * 70)
    print("TEXT LENGTH DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print("\nToken counting method: whitespace splitting")
    print("Note: This is an approximation. Actual BERT tokens may differ.\n")

    # 分析 AG News
    print("\n" + "=" * 70)
    print("AG NEWS DATASET")
    print("=" * 70)

    # 获取数据目录（相对于项目根目录）
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    ag_data = load_ag_news(data_dir=os.path.join(base_dir, 'part1', 'data', 'ag_news'))
    for split_name in ['train', 'val', 'test']:
        if split_name not in ag_data:
            continue
        texts, labels = ag_data[split_name]

        stats = analyze_text_length(texts, 'ag_news', split_name)
        all_stats.append(stats)

        print(f"\n{split_name.upper()} SET:")
        print(f"  Number of samples: {stats['num_samples']:,}")
        print(f"  Mean length: {stats['mean']:.1f} tokens")
        print(f"  Median length: {stats['median']:.1f} tokens")
        print(f"  Std deviation: {stats['std']:.1f}")
        print(f"  Min length: {stats['min']} tokens")
        print(f"  Max length: {stats['max']} tokens")
        print(f"  95th percentile: {stats['95th_percentile']:.1f} tokens")
        print(f"  99th percentile: {stats['99th_percentile']:.1f} tokens")

    # 分析 IMDb
    print("\n" + "=" * 70)
    print("IMDB DATASET")
    print("=" * 70)

    imdb_data = load_imdb(data_dir=os.path.join(base_dir, 'part1', 'data', 'imdb'))
    for split_name in ['train', 'val', 'test']:
        if split_name not in imdb_data:
            continue
        texts, labels = imdb_data[split_name]

        stats = analyze_text_length(texts, 'imdb', split_name)
        all_stats.append(stats)

        print(f"\n{split_name.upper()} SET:")
        print(f"  Number of samples: {stats['num_samples']:,}")
        print(f"  Mean length: {stats['mean']:.1f} tokens")
        print(f"  Median length: {stats['median']:.1f} tokens")
        print(f"  Std deviation: {stats['std']:.1f}")
        print(f"  Min length: {stats['min']} tokens")
        print(f"  Max length: {stats['max']} tokens")
        print(f"  95th percentile: {stats['95th_percentile']:.1f} tokens")
        print(f"  99th percentile: {stats['99th_percentile']:.1f} tokens")

    # 保存结果
    stats_df = pd.DataFrame(all_stats)
    results_dir = os.path.join(base_dir, 'part1', 'results')
    os.makedirs(results_dir, exist_ok=True)
    stats_df.to_csv(os.path.join(results_dir, 'text_length_stats.csv'), index=False)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(stats_df.to_string(index=False))

    # 讨论对Transformer的影响
    print("\n" + "=" * 70)
    print("IMPLICATIONS FOR TRANSFORMER MODELS")
    print("=" * 70)
    print("\nBERT/DistilBERT has a maximum input length of 512 tokens.")
    print("Text length distribution affects truncation strategy:\n")

    for dataset in ['ag_news', 'imdb']:
        dataset_stats = [s for s in all_stats if s['dataset'] == dataset]
        train_95th = [s for s in dataset_stats if s['split'] == 'train'][0]['95th_percentile']
        train_max = [s for s in dataset_stats if s['split'] == 'train'][0]['max']

        print(f"{dataset.upper()}:")
        print(f"  - 95% of training samples have ≤ {train_95th:.0f} tokens")
        print(f"  - Maximum length: {train_max} tokens")

        if train_95th <= 512:
            print(f"  → MOST samples fit within BERT's 512-token limit (no truncation needed for 95%)")
        else:
            print(f"  → SIGNIFICANT truncation needed ({train_95th/512:.1f}x > 512 limit)")
        print()

    print(f"\nResults saved to: results/text_length_stats.csv")

    return all_stats


if __name__ == '__main__':
    stats = main()
