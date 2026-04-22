import sys
import os
# 获取当前脚本所在的models文件夹路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 往上退一层，获取ML-project项目根目录
project_root = os.path.dirname(script_dir)
# 把项目根目录加入Python搜索路径，解决找不到src的问题
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# 导入你src文件夹里写好的线性回归类
from src.linear_regression import LinearRegression

# 路径配置
CSV_DATA_PATH = os.path.join(project_root, "data", "house", "house_data.csv")  # 房价csv路径
MODELS_SAVE_DIR = os.path.join(project_root, "models")                          # 模型保存目录

# 模型命名（严格遵循项目规范：算法_数据_日期.pkl）
ALGO_NAME = "linear"
DATA_NAME = "house"
from datetime import datetime
DATE = datetime.now().strftime("%Y%m%d")
MODEL_FILENAME = f"{ALGO_NAME}_{DATA_NAME}_{DATE}.pkl"

# 线性回归超参数配置（和你src/linear_regression.py完全对齐，课堂可现场调整）
# 可选求解方式：normal_equation(最小二乘闭式解，默认) / gradient_descent(梯度下降)
SOLVE_METHOD = "normal_equation"
LEARNING_RATE = 0.001    # 仅梯度下降模式生效
N_ITERATIONS = 1000       # 仅梯度下降模式生效
BATCH_SIZE = 32            # 仅梯度下降模式生效，可自定义输入

# 数据集划分配置
TEST_SIZE = 0.2  # 测试集占比20%
RANDOM_STATE = 42  # 随机种子，保证结果可复现

# ====================== 核心函数：读取房价CSV数据 ======================
def load_house_data(csv_path):
    """
    读取house_data.csv，提取特征和房价标签
    :param csv_path: house_data.csv的完整路径
    :return: 特征矩阵X，标签向量y
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"未找到房价数据文件：{csv_path}\n请确认 data/house/house_data.csv 路径正确")
    
    print(f"正在读取房价数据：{csv_path}")
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 提取特征（x1-x4）和标签（y），和你提供的CSV结构完全对齐
    feature_cols = ['x1', 'x2', 'x3', 'x4']
    X = df[feature_cols].values.astype(np.float32)
    y = df['y'].values.astype(np.float32)
    
    print(f"数据读取完成！共 {len(X)} 条样本，特征维度：{X.shape[1]}")
    return X, y

# ====================== 主流程：训练+保存模型 ======================
if __name__ == "__main__":
    # 1. 读取房价CSV数据
    X_all, y_all = load_house_data(CSV_DATA_PATH)

    # 2. 自动划分训练集和测试集
    print(f"\n正在划分训练集和测试集（测试集占比：{TEST_SIZE*100}%）...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"数据集划分完成！训练集：{len(X_train)} 条，测试集：{len(X_test)} 条")

    # 3. 初始化线性回归模型（和你src里的类接口完全对齐）
    print(f"\n开始训练线性回归模型，求解方式：{SOLVE_METHOD}")
    if SOLVE_METHOD == "normal_equation":
        model = LinearRegression(method='normal_equation')
    else:
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=LEARNING_RATE,
            n_iterations=N_ITERATIONS,
            batch_size=BATCH_SIZE
        )

    # 4. 执行训练（输入为特征向量，完全符合项目要求）
    model.fit(X_train, y_train)

    # 5. 模型评估（线性回归用R²决定系数，和你代码里的score方法完全对齐）
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    print(f"训练完成！")
    print(f"训练集R²分数：{train_r2}")
    print(f"测试集R²分数：{test_r2}")

    # 6. 保存模型到models文件夹，严格遵循项目命名规范
    model_save_path = os.path.join(MODELS_SAVE_DIR, MODEL_FILENAME)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n模型已成功保存到：{model_save_path}")