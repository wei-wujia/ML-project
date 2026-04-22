# logistic_regression_titanic_training.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from datetime import datetime

# 数据文件路径
TRAIN_DATA_PATH = "./data/titanic/titanic_train.csv"
TEST_DATA_PATH = "./data/titanic/titanic_test.csv"
# 模型保存根目录
MODEL_SAVE_DIR = "./models"
# 特征列配置（仅保留有效特征，过滤所有zero无效列）
FEATURE_COLS = ["Age", "Fare", "Sex", "sibsp", "Parch", "Pclass", "Embarked"]
# 标签列名（匹配你的数据集列名）
LABEL_COL = "2urvived"
# 随机种子保证结果可复现
RANDOM_SEED = 42
# 当前日期（用于pkl文件命名）
CURRENT_DATE = datetime.now().strftime("%Y%m%d")

# ===================== 1. 数据预处理函数 =====================
def preprocess_data(df, feature_cols, label_col=None, scaler=None, is_train=True):
    """
    数据预处理：缺失值填充、特征标准化
    :param df: 输入DataFrame
    :param feature_cols: 特征列名列表
    :param label_col: 标签列名
    :param scaler: 标准化器（预测时传入训练好的scaler）
    :param is_train: 是否为训练阶段
    :return: 标准化后的特征、标签、标准化器（训练阶段）
    """
    # 提取特征
    X = df[feature_cols].copy()
    
    # 缺失值填充
    X["Age"] = X["Age"].fillna(X["Age"].mean())       # 年龄用均值填充
    X["Fare"] = X["Fare"].fillna(X["Fare"].mean())    # 票价用均值填充
    X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])  # 登船港口用众数填充
    
    # 标签处理（仅训练阶段）
    y = None
    if label_col is not None and is_train:
        y = df[label_col].copy()
        y = y.astype(int)  # 确保标签为0/1二值整数
    
    # 特征标准化
    if is_train:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    else:
        if scaler is None:
            raise ValueError("预测阶段必须传入训练好的scaler")
        X_scaled = scaler.transform(X)
        return X_scaled, y

# ===================== 2. 数据加载 =====================
if __name__ == "__main__":
    # 校验文件是否存在
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"训练数据不存在：{TRAIN_DATA_PATH}，请检查路径")
    
    # 读取数据
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    print(f"训练数据加载完成，共{df_train.shape[0]}条样本，{df_train.shape[1]}列")
    
    # 预处理训练数据
    X_train, y_train, scaler = preprocess_data(
        df_train, FEATURE_COLS, LABEL_COL, is_train=True
    )
    
    # 划分训练集/验证集（用于模型效果评估）
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED, stratify=y_train
    )

    # ===================== 3. 逻辑回归模型训练 =====================
    # 初始化模型（解决二分类任务，适配数据收敛）
    lr_model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,  # 保证迭代收敛
        class_weight="balanced"  # 平衡样本权重，适配titanic样本不均衡
    )
    
    # 训练模型
    lr_model.fit(X_train_split, y_train_split)
    
    # 验证集评估
    y_val_pred = lr_model.predict(X_val_split)
    val_accuracy = accuracy_score(y_val_split, y_val_pred)
    val_report = classification_report(y_val_split, y_val_pred)
    
    print("="*50)
    print(f"验证集准确率：{val_accuracy:.4f}")
    print("验证集分类报告：")
    print(val_report)
    print("="*50)
    
    # 用全量训练数据训练最终上线模型
    final_model = LogisticRegression(
        random_state=RANDOM_SEED,
        max_iter=1000,
        class_weight="balanced"
    )
    final_model.fit(X_train, y_train)
    print("全量训练集模型训练完成")

    # ===================== 4. 模型保存（符合命名规范）=====================
    # 确保保存目录存在
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    # 模型文件命名：算法_数据_日期.pkl
    model_save_name = f"logistic_titanic_{CURRENT_DATE}.pkl"
    scaler_save_name = f"logistic_titanic_scaler_{CURRENT_DATE}.pkl"
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_save_name)
    scaler_save_path = os.path.join(MODEL_SAVE_DIR, scaler_save_name)
    
    # 保存模型和标准化器
    joblib.dump(final_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    
    print(f"模型已保存至：{model_save_path}")
    print(f"标准化器已保存至：{scaler_save_path}")