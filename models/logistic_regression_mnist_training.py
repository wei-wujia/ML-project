import sys
import os

# 1. 获取当前脚本所在的绝对路径 (即 .../ML-project/models)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 往上退一层，获取 ML-project 根目录
project_root = os.path.dirname(script_dir)
# 3. 把 ML-project 根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pickle
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from src.logistic_regression import LogisticRegression

# ====================== 配置项 ======================
# 自动拼接路径，无论在哪运行都不会错
DATA_DIR = os.path.join(project_root, "data", "mnist")
MODELS_SAVE_DIR = os.path.join(project_root, "models")

# 模型命名
from datetime import datetime
MODEL_FILENAME = f"logistic_mnist_{datetime.now().strftime('%Y%m%d')}.pkl"

# 超参数
LEARNING_RATE = 0.001
N_ITERATIONS = 1000

# ====================== 读取 data/mnist/0 和 1 下的图片 ======================
def load_data():
    images = []
    labels = []
    
    # 遍历 0 和 1 文件夹
    for label in [0, 1]:
        folder_path = os.path.join(DATA_DIR, str(label))
        
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"找不到文件夹: {folder_path}")

        print(f"正在读取: {folder_path} ...")
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # 读取图片 -> 转灰度 -> 转numpy向量
                    img = Image.open(os.path.join(folder_path, filename)).convert('L')
                    if img.size != (28, 28):
                        img = img.resize((28, 28))
                    img_array = np.array(img, dtype=np.float32).flatten()
                    
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"跳过文件 {filename}: {e}")

    return np.array(images), np.array(labels)

# 1. 加载数据
X_all, y_all = load_data()

# 2. 划分训练集和测试集
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# ====================== 提取 HOG 特征 ======================
def extract_hog(imgs):
    features = []
    for img in imgs:
        # 把展平的向量还原回 28x28 再提 HOG
        f = hog(img.reshape(28, 28), orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2))
        features.append(f)
    return np.array(features)

print("正在提取 HOG 特征...")
X_train = extract_hog(X_train_raw)
X_test = extract_hog(X_test_raw)

# ====================== 训练 & 保存 ======================
print(f"开始训练逻辑回归 (迭代次数: {N_ITERATIONS})...")
model = LogisticRegression(learning_rate=LEARNING_RATE, n_iterations=N_ITERATIONS)
model.fit(X_train, y_train)

# 打印准确率
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"训练完成!")
print(f"训练集准确率: {train_acc}")
print(f"测试集准确率: {test_acc}")

# 保存模型
save_path = os.path.join(MODELS_SAVE_DIR, MODEL_FILENAME)
with open(save_path, 'wb') as f:
    pickle.dump(model, f)

print(f"\n模型已保存至: {save_path}")