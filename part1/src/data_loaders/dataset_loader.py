"""
统一数据加载器模块
提供AG News和IMDb的数据加载功能，以及分层抽样功能
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os

def load_ag_news(data_dir='data/ag_news'):
    """
    加载AG News数据集

    返回:
        dict: {
            'train': (texts, labels),
            'val': (texts, labels),
            'test': (texts, labels),
            'num_classes': int,
            'class_names': list
        }
    """
    # 加载数据
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')

    # 加载元数据
    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    return {
        'train': (train_df['text'].tolist(), train_df['label'].values),
        'val': (val_df['text'].tolist(), val_df['label'].values),
        'test': (test_df['text'].tolist(), test_df['label'].values),
        'num_classes': metadata['num_classes'],
        'class_names': metadata['class_names']
    }

def load_imdb(data_dir='data/imdb'):
    """
    加载IMDb数据集

    返回:
        dict: {
            'train': (texts, labels),
            'val': (texts, labels),
            'test': (texts, labels),
            'num_classes': int,
            'class_names': list
        }
    """
    # 加载数据
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')

    # 加载元数据
    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)

    return {
        'train': (train_df['text'].tolist(), train_df['label'].values),
        'val': (val_df['text'].tolist(), val_df['label'].values),
        'test': (test_df['text'].tolist(), test_df['label'].values),
        'num_classes': metadata['num_classes'],
        'class_names': metadata['class_names']
    }

def stratified_subsample(texts, labels, ratio, random_state=42):
    """
    分层抽样，保持类别比例

    参数:
        texts: 文本列表
        labels: 标签数组
        ratio: 抽样比例 (0.01 = 1%)
        random_state: 随机种子

    返回:
        sub_texts: 抽样后的文本列表
        sub_labels: 抽样后的标签数组
    """
    if ratio >= 1.0:
        return texts, labels

    # 转换为numpy数组
    labels = np.array(labels)

    # 使用StratifiedShuffleSplit进行分层抽样
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1-ratio,
        random_state=random_state
    )

    for keep_idx, _ in sss.split(texts, labels):
        sub_texts = [texts[i] for i in keep_idx]
        sub_labels = labels[keep_idx]

    return sub_texts, sub_labels

def get_text_length_stats(texts, tokenizer=None):
    """
    统计文本长度分布

    参数:
        texts: 文本列表
        tokenizer: 如果提供，用于计算token数量；否则计算字符数

    返回:
        dict: 统计信息
    """
    if tokenizer:
        lengths = [len(tokenizer(text)['input_ids']) for text in texts]
    else:
        # 简单估算：按空格分词
        lengths = [len(text.split()) for text in texts]

    lengths = np.array(lengths)

    return {
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        '95th_percentile': float(np.percentile(lengths, 95)),
        '99th_percentile': float(np.percentile(lengths, 99))
    }

if __name__ == '__main__':
    # 测试数据加载
    print("Testing AG News loader...")
    ag_data = load_ag_news()
    print(f"Train size: {len(ag_data['train'][0])}")
    print(f"Val size: {len(ag_data['val'][0])}")
    print(f"Test size: {len(ag_data['test'][0])}")
    print(f"Classes: {ag_data['class_names']}")

    # 测试分层抽样
    print("\nTesting stratified subsample...")
    texts, labels = ag_data['train']
    sub_texts, sub_labels = stratified_subsample(texts, labels, 0.1, seed=42)
    print(f"Original size: {len(texts)}")
    print(f"Subsampled size: {len(sub_texts)}")

    # 测试文本长度统计
    print("\nTesting text length stats...")
    stats = get_text_length_stats(texts[:1000])
    print(f"Mean length: {stats['mean']:.1f} words")
    print(f"95th percentile: {stats['95th_percentile']:.1f} words")
