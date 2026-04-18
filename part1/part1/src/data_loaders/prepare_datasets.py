"""
数据预处理脚本
处理AG News和IMDb数据集，生成标准格式的训练/验证/测试文件
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import json

def prepare_ag_news(data_dir='/Users/gabby/Desktop/5104/T1/archive (4)', output_dir='data/ag_news'):
    """
    处理AG News数据集
    - 合并Title和Description作为输入文本
    - 类别标签减1转换为0-indexed
    - 划分train/val/test
    """
    print("="*60)
    print("Processing AG News Dataset")
    print("="*60)

    # 读取原始数据
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')

    print(f"Original train size: {len(train_df)}")
    print(f"Original test size: {len(test_df)}")

    # 合并title和description
    train_df['text'] = (train_df['Title'].fillna('') + ' ' +
                        train_df['Description'].fillna('')).str.strip()
    test_df['text'] = (test_df['Title'].fillna('') + ' ' +
                       test_df['Description'].fillna('')).str.strip()

    # 类别转换为0-indexed (原始是1-4)
    train_df['label'] = train_df['Class Index'] - 1
    test_df['label'] = test_df['Class Index'] - 1

    # 获取测试集
    test_texts = test_df['text'].values
    test_labels = test_df['label'].values

    # 从原始训练集划分出训练集和验证集 (90/10)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].values,
        train_df['label'].values,
        test_size=0.1,
        random_state=42,
        stratify=train_df['label'].values
    )

    # 保存处理后的数据
    os.makedirs(output_dir, exist_ok=True)

    # 保存为CSV
    pd.DataFrame({'text': train_texts, 'label': train_labels}).to_csv(
        f'{output_dir}/train.csv', index=False
    )
    pd.DataFrame({'text': val_texts, 'label': val_labels}).to_csv(
        f'{output_dir}/val.csv', index=False
    )
    pd.DataFrame({'text': test_texts, 'label': test_labels}).to_csv(
        f'{output_dir}/test.csv', index=False
    )

    # 保存元数据
    metadata = {
        'num_classes': 4,
        'class_names': ['World', 'Sports', 'Business', 'Sci/Tech'],
        'train_size': len(train_texts),
        'val_size': len(val_texts),
        'test_size': len(test_texts),
        'class_distribution': {
            'train': pd.Series(train_labels).value_counts().to_dict(),
            'val': pd.Series(val_labels).value_counts().to_dict(),
            'test': pd.Series(test_labels).value_counts().to_dict()
        }
    }

    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProcessed train size: {len(train_texts)}")
    print(f"Processed val size: {len(val_texts)}")
    print(f"Processed test size: {len(test_texts)}")
    print(f"\nClass distribution in train: {metadata['class_distribution']['train']}")
    print(f"Saved to {output_dir}/")

    return metadata

def prepare_imdb(data_path='/Users/gabby/Desktop/5104/T1/IMDB Dataset.csv', output_dir='data/imdb'):
    """
    处理IMDb数据集
    - 转换sentiment为0/1标签
    - 划分train/val/test (70/15/15)
    """
    print("\n" + "="*60)
    print("Processing IMDb Dataset")
    print("="*60)

    # 读取数据
    df = pd.read_csv(data_path)
    print(f"Original size: {len(df)}")

    # 转换标签: positive=1, negative=0
    df['label'] = (df['sentiment'] == 'positive').astype(int)
    df['text'] = df['review']

    # 首先划分出测试集 15%
    train_val_df, test_df = train_test_split(
        df[['text', 'label']],
        test_size=0.15,
        random_state=42,
        stratify=df['label']
    )

    # 从剩余数据中划分验证集 17.6% (15/85)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.176,
        random_state=42,
        stratify=train_val_df['label']
    )

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)

    # 保存元数据
    metadata = {
        'num_classes': 2,
        'class_names': ['Negative', 'Positive'],
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'class_distribution': {
            'train': train_df['label'].value_counts().to_dict(),
            'val': val_df['label'].value_counts().to_dict(),
            'test': test_df['label'].value_counts().to_dict()
        }
    }

    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nProcessed train size: {len(train_df)}")
    print(f"Processed val size: {len(val_df)}")
    print(f"Processed test size: {len(test_df)}")
    print(f"\nClass distribution in train: {metadata['class_distribution']['train']}")
    print(f"Saved to {output_dir}/")

    return metadata

if __name__ == '__main__':
    # 处理两个数据集
    ag_metadata = prepare_ag_news()
    imdb_metadata = prepare_imdb()

    print("\n" + "="*60)
    print("Data preparation completed!")
    print("="*60)
