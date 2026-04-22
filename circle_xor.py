import sys
import os
import numpy as np
import pandas as pd


# 将src\data到Python模块搜索路径
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.append(src_path)
data_path = os.path.join(os.path.dirname(__file__), 'data')
sys.path.append(data_path)

from logistic_regression import LogisticRegression
from SVM import SVM

# ===================== 数据读取与模型评估函数 =====================
def load_dataset(file_path):
    """读取csv数据集，返回特征矩阵X和标签向量y"""
    # 拼接完整文件路径（确保csv在根目录下）
    full_path = os.path.join(os.path.dirname(__file__), file_path)
    df = pd.read_csv(full_path)
    X = df[['x1', 'x2']].values
    y = df['label'].values
    return X, y

def evaluate_model(model, X, y, model_name, dataset_name):
    """训练模型并输出准确率（训练集=测试集）"""
    model.fit(X, y)
    accuracy = model.score(X, y)
    print(f"【{dataset_name}数据集】{model_name} 准确率: {accuracy:.4f}")
    return accuracy

# ===================== 3. 主流程：执行分类与评估 =====================
if __name__ == "__main__":
    # ---------------------- 加载数据集 ----------------------
    circles_X, circles_y = load_dataset('./data/circles_dataset.csv')
    xor_X, xor_y = load_dataset('./data/xor_dataset.csv')

    # ---------------------- 环形数据集分类结果 ----------------------
    print("1. 环形数据集 circles_dataset 分类准确率")

    lr_circles = LogisticRegression(learning_rate=0.01, n_iterations=5000)
    evaluate_model(lr_circles, circles_X, circles_y, "逻辑回归", "circles")
    svm_linear_circles = SVM(kernel='linear', C=1.0, max_iter=1000)
    evaluate_model(svm_linear_circles, circles_X, circles_y, "线性核SVM", "circles")
    svm_rbf_circles = SVM(kernel='rbf', C=10.0, sigma=0.5, max_iter=1000)
    evaluate_model(svm_rbf_circles, circles_X, circles_y, "高斯核SVM", "circles")

    # ---------------------- 异或数据集分类结果 ----------------------
    print("2. 异或数据集 xor_dataset 分类准确率")

    lr_xor = LogisticRegression(learning_rate=0.01, n_iterations=5000)
    evaluate_model(lr_xor, xor_X, xor_y, "逻辑回归", "xor")
    svm_linear_xor = SVM(kernel='linear', C=1.0, max_iter=1000)
    evaluate_model(svm_linear_xor, xor_X, xor_y, "线性核SVM", "xor")
    svm_rbf_xor = SVM(kernel='rbf', C=10.0, sigma=0.5, max_iter=1000)
    evaluate_model(svm_rbf_xor, xor_X, xor_y, "高斯核SVM", "xor")