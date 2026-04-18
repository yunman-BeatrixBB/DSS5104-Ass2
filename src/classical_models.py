"""
经典文本分类模型实现
TF-IDF + Logistic Regression / SVM
"""
import numpy as np
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


class ClassicalTextClassifier:
    """
    经典文本分类器：TF-IDF + Logistic Regression / SVM

    支持的功能：
    - TF-IDF特征提取
    - Logistic Regression分类
    - Linear SVM分类
    - 自动记录训练和推理时间
    - 完整的评估指标
    """

    def __init__(self, model_type='lr', C=1.0, max_features=50000,
                 ngram_range=(1, 2), random_state=42, class_weight='balanced'):
        """
        初始化分类器

        参数:
            model_type: 'lr' (Logistic Regression) 或 'svm' (LinearSVC)
            C: 正则化参数
            max_features: TF-IDF最大特征数
            ngram_range: n-gram范围，(1,1)表示unigram，(1,2)表示uni+bigram
            random_state: 随机种子
            class_weight: 类别权重，'balanced'自动根据类别频率调整
        """
        self.model_type = model_type
        self.C = C
        self.max_features = max_features
        # 确保ngram_range是元组（处理从JSON加载的列表）
        if isinstance(ngram_range, list):
            ngram_range = tuple(ngram_range)
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.class_weight = class_weight

        self.vectorizer = None
        self.model = None
        self.training_time = None
        self.inference_time = None

    def fit(self, texts, labels):
        """
        训练模型

        参数:
            texts: 文本列表
            labels: 标签数组

        返回:
            self
        """
        start_time = time.time()

        # 1. 构建TF-IDF向量器
        print(f"Building TF-IDF with max_features={self.max_features}, ngram_range={self.ngram_range}")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,  # 忽略出现次数少于2次的词
            max_df=0.95,  # 忽略出现在95%以上文档中的词
            sublinear_tf=True  # 使用1+log(tf)替代原始tf
        )

        # 2. 将文本转换为TF-IDF特征
        print("Fitting TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF matrix shape: {X.shape}")

        # 3. 初始化分类器
        if self.model_type == 'lr':
            print(f"Training Logistic Regression (C={self.C})...")
            self.model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight=self.class_weight,
                solver='lbfgs'
            )
        elif self.model_type == 'svm':
            print(f"Training Linear SVM (C={self.C})...")
            self.model = LinearSVC(
                C=self.C,
                max_iter=3000,
                random_state=self.random_state,
                class_weight=self.class_weight,
                dual='auto'
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose 'lr' or 'svm'.")

        # 4. 训练
        self.model.fit(X, labels)

        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")

        return self

    def predict(self, texts):
        """
        预测标签

        参数:
            texts: 文本列表

        返回:
            predictions: 预测标签数组
        """
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, texts):
        """
        预测概率（仅Logistic Regression支持）

        参数:
            texts: 文本列表

        返回:
            probabilities: 概率数组，shape=(n_samples, n_classes)
        """
        if self.model_type != 'lr':
            raise ValueError("predict_proba only available for Logistic Regression (model_type='lr')")

        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, texts, labels, dataset_name="Test", verbose=True):
        """
        评估模型

        参数:
            texts: 文本列表
            labels: 真实标签
            dataset_name: 数据集名称（用于打印）
            verbose: 是否打印详细信息

        返回:
            dict: 包含所有评估指标的字典
        """
        # 推理时间
        start_time = time.time()
        predictions = self.predict(texts)
        self.inference_time = time.time() - start_time

        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average='macro')
        per_class_f1 = f1_score(labels, predictions, average=None).tolist()

        if verbose:
            print(f"\n{'='*60}")
            print(f"{dataset_name} Results ({self.model_type.upper()}, C={self.C})")
            print(f"{'='*60}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            print(f"Training Time: {self.training_time:.2f}s")
            print(f"Inference Time: {self.inference_time:.2f}s ({len(texts)} samples)")
            print(f"Throughput: {len(texts)/self.inference_time:.1f} samples/sec")
            print(f"\nPer-class F1: {per_class_f1}")

        return {
            'model_type': self.model_type,
            'C': self.C,
            'max_features': self.max_features,
            'ngram_range': str(self.ngram_range),
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'train_time': self.training_time,
            'inference_time': self.inference_time,
            'throughput': len(texts) / self.inference_time if self.inference_time > 0 else 0,
            'predictions': predictions
        }

    def get_feature_names(self, top_n=20):
        """
        获取最重要的特征（词）

        参数:
            top_n: 返回前N个重要特征

        返回:
            dict: 每个类别的top特征
        """
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        feature_names = np.array(self.vectorizer.get_feature_names_out())

        if self.model_type == 'lr':
            # Logistic Regression有coef_属性
            coefs = self.model.coef_
            if coefs.shape[0] == 1:  # 二分类
                coefs = np.vstack([-coefs, coefs])

            result = {}
            for i, coef in enumerate(coefs):
                top_positive = feature_names[np.argsort(coef)[-top_n:][::-1]]
                top_negative = feature_names[np.argsort(coef)[:top_n]]
                result[f'class_{i}'] = {
                    'top_positive': top_positive.tolist(),
                    'top_negative': top_negative.tolist()
                }
            return result
        else:
            # SVM也有coef_属性
            return {"note": "Feature importance available via coef_ attribute"}

    def save(self, filepath):
        """
        保存模型到文件

        参数:
            filepath: 保存路径
        """
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model,
                'config': {
                    'model_type': self.model_type,
                    'C': self.C,
                    'max_features': self.max_features,
                    'ngram_range': self.ngram_range,
                    'class_weight': self.class_weight
                }
            }, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        从文件加载模型

        参数:
            filepath: 模型文件路径

        返回:
            ClassicalTextClassifier: 加载好的模型实例
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        config = data['config']
        instance = cls(
            model_type=config['model_type'],
            C=config['C'],
            max_features=config['max_features'],
            ngram_range=config['ngram_range'],
            class_weight=config.get('class_weight', 'balanced')
        )

        instance.vectorizer = data['vectorizer']
        instance.model = data['model']

        return instance


def quick_test():
    """快速测试函数"""
    from data_loaders import load_ag_news

    print("Loading AG News dataset...")
    data = load_ag_news()
    train_texts, train_labels = data['train']
    test_texts, test_labels = data['test']

    # 只取一部分数据快速测试
    train_texts = train_texts[:1000]
    train_labels = train_labels[:1000]
    test_texts = test_texts[:500]
    test_labels = test_labels[:500]

    print(f"\nTraining samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")

    # 测试Logistic Regression
    print("\n" + "="*60)
    print("Testing Logistic Regression")
    print("="*60)
    clf_lr = ClassicalTextClassifier(model_type='lr', C=1.0, max_features=10000)
    clf_lr.fit(train_texts, train_labels)
    results_lr = clf_lr.evaluate(test_texts, test_labels)

    # 测试SVM
    print("\n" + "="*60)
    print("Testing SVM")
    print("="*60)
    clf_svm = ClassicalTextClassifier(model_type='svm', C=1.0, max_features=10000)
    clf_svm.fit(train_texts, train_labels)
    results_svm = clf_svm.evaluate(test_texts, test_labels)

    print("\n" + "="*60)
    print("Quick test completed!")
    print("="*60)


if __name__ == '__main__':
    quick_test()
