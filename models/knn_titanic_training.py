import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 数据集路径
TRAIN_DATA_PATH = "./data/titanic/titanic_train.csv"
TEST_DATA_PATH = "./data/titanic/titanic_test.csv"
# 模型保存路径（和你其他模型放在同一目录）
MODEL_SAVE_PATH = "./models/knn_titanic.pkl"
SCALER_SAVE_PATH = "./models/knn_titanic_scaler.pkl"

# ===================== 2. 数据加载与初步查看 =====================
print("开始加载泰坦尼克号数据集...")
# 读取训练数据
df = pd.read_csv(TRAIN_DATA_PATH)
print(f"数据加载完成！总样本数：{df.shape[0]}，总列数：{df.shape[1]}")

# ===================== 3. 数据预处理（核心步骤）=====================
# 3.1 删除无用列：乘客ID、全0的zero列（无任何信息增益）
drop_columns = ["Passengerid"] + [col for col in df.columns if col.startswith("zero")]
df_clean = df.drop(columns=drop_columns, axis=1)

# 3.2 重命名标签列，方便后续处理
df_clean.rename(columns={"2urvived": "survived"}, inplace=True)

# 3.3 缺失值处理
# 数值型特征：年龄、票价用均值填充
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].mean())
df_clean["Fare"] = df_clean["Fare"].fillna(df_clean["Fare"].mean())
# 分类型特征：登船港口用众数填充
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])

# 3.4 分类特征编码
# 性别列保持数值映射（1=男性，0=女性，与原数据一致）
# 登船港口独热编码，消除分类特征的有序性影响
df_clean = pd.get_dummies(df_clean, columns=["Embarked"], drop_first=True)

# 3.5 划分特征矩阵X和标签y（二分类标签：1=生存，0=未生存）
X = df_clean.drop("survived", axis=1)
y = df_clean["survived"]

# 3.6 划分训练集/测试集（测试集占比20%，与你线性回归代码保持一致）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # stratify保证标签分布一致
)
print(f"数据集划分完成！训练集：{X_train.shape[0]}条，测试集：{X_test.shape[0]}条")

# 3.7 特征标准化（KNN核心要求：距离敏感算法必须做标准化，消除量纲影响）
scaler = StandardScaler()
# 仅在训练集拟合，避免数据泄露
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 4. KNN模型训练 =====================
print("开始训练KNN二分类模型...")
# 初始化KNN模型（n_neighbors可通过网格搜索调优，默认5）
knn_model = KNeighborsClassifier(n_neighbors=5)
# 模型拟合
knn_model.fit(X_train_scaled, y_train)
print("KNN模型训练完成！")

# ===================== 5. 模型效果评估 =====================
# 模型预测
y_train_pred = knn_model.predict(X_train_scaled)
y_test_pred = knn_model.predict(X_test_scaled)

# 计算准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"训练集准确率：{train_accuracy}")
print(f"测试集准确率：{test_accuracy}")
print("\n【测试集详细分类报告】")
print(classification_report(y_test, y_test_pred))
print("\n【测试集混淆矩阵】")
print(confusion_matrix(y_test, y_test_pred))
print("="*50)

# ===================== 6. 模型&标准化器保存为pkl文件 =====================
# 自动创建models文件夹（如果不存在）
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# 保存KNN模型和标准化器（预测新数据必须用同一个标准化器）
joblib.dump(knn_model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)

print(f"KNN模型已保存至：{MODEL_SAVE_PATH}")
print(f"特征标准化器已保存至：{SCALER_SAVE_PATH}")

# ===================== 7. 验证模型保存有效性 =====================
print("验证模型加载效果...")
loaded_model = joblib.load(MODEL_SAVE_PATH)
loaded_scaler = joblib.load(SCALER_SAVE_PATH)

# 用测试集验证加载后的模型
verify_pred = loaded_model.predict(loaded_scaler.transform(X_test))
verify_accuracy = accuracy_score(y_test, verify_pred)
print(f"模型加载后测试集准确率：{verify_accuracy}，模型保存&加载验证成功！")