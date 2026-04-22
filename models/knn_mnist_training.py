import os
import pickle
import numpy as np
from PIL import Image  # 需要安装Pillow库：pip install Pillow
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# 路径配置（完全适配你的项目结构）
DATA_PATH = "./data/mnist/"       # MNIST数据集所在路径（下面有0/1/2...9文件夹）
MODELS_SAVE_PATH = "./models/"    # 模型pkl保存路径
# 命名配置（严格遵循你的命名规范：算法_数据_日期.pkl）
ALGORITHM_NAME = "knn"
DATA_NAME = "mnist"
DATE = "20260421"  # 可修改为实际运行日期
# KNN模型超参数
N_NEIGHBORS = 5  # 近邻数，常用3/5/7
# 训练/测试集划分比例
TEST_SIZE = 0.2
RANDOM_STATE = 42

# 模型与标准化器的保存文件名
MODEL_SAVE_FILE = os.path.join(MODELS_SAVE_PATH, f"{ALGORITHM_NAME}_{DATA_NAME}_{DATE}.pkl")
SCALER_SAVE_FILE = os.path.join(MODELS_SAVE_PATH, f"{ALGORITHM_NAME}_{DATA_NAME}_scaler_{DATE}.pkl")

# ===================== 1. 初始化环境 =====================
os.makedirs(MODELS_SAVE_PATH, exist_ok=True)

# ===================== 2. 【新增】从本地文件夹加载MNIST图片 =====================
def load_mnist_from_folders(data_root):
    """
    从文件夹结构加载MNIST数据：
    data_root/
        ├── 0/  (存放所有数字0的图片)
        ├── 1/  (存放所有数字1的图片)
        ├── ...
        └── 9/  (存放所有数字9的图片)
    图片要求：28*28像素，支持png/jpg/bmp等格式
    """
    print(f"正在从文件夹加载MNIST数据：{data_root}")
    images = []
    labels = []
    
    # 遍历0-9共10个数字文件夹
    for digit in range(10):
        digit_folder = os.path.join(data_root, str(digit))
        if not os.path.exists(digit_folder):
            print(f"警告：未找到数字 {digit} 的文件夹 {digit_folder}，将跳过该类别")
            continue
            
        # 遍历文件夹下的所有图片文件
        for img_name in os.listdir(digit_folder):
            img_path = os.path.join(digit_folder, img_name)
            # 跳过非文件（如.DS_Store）和非图片后缀
            if not os.path.isfile(img_path) or img_name.lower().endswith(('.txt', '.csv', '.md')):
                continue
                
            try:
                # 读取图片：转为灰度图→强制resize为28*28→展平为784维向量
                img = Image.open(img_path).convert('L')  # 'L'模式转为灰度
                img = img.resize((28, 28))  # 确保尺寸统一为28*28
                img_array = np.array(img).flatten()  # 展平为 (784,) 的一维数组
                
                images.append(img_array)
                labels.append(digit)
            except Exception as e:
                print(f"跳过无法读取的图片：{img_path}，错误：{e}")
                continue

    if len(images) == 0:
        raise FileNotFoundError(f"未在 {data_root} 下找到任何有效图片，请检查文件夹结构！")
        
    # 转为numpy数组
    X = np.array(images)
    y = np.array(labels)
    print(f"本地图片加载完成！共 {len(X)} 张图片，类别数：{len(np.unique(y))}")
    return X, y

# 执行数据加载
try:
    X, y = load_mnist_from_folders(DATA_PATH)
    # 划分训练集和测试集（分层划分，保证每个类别比例一致）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
except Exception as e:
    print(f"本地图片加载失败：{e}，将尝试加载标准MNIST数据集...")
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int32)
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

print(f"训练集：{len(X_train)}条 | 测试集：{len(X_test)}条")

# ===================== 3. 数据标准化（KNN必做）=====================
print("正在进行数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================== 4. 训练KNN 10分类模型 =====================
print(f"正在训练KNN模型，近邻数n_neighbors={N_NEIGHBORS}...")
knn_model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)

# ===================== 5. 模型测试与效果评估 =====================
print("正在执行测试集预测与模型评估...")
y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"【测试集整体准确率】：{accuracy}")
print("【测试集混淆矩阵】")
print(conf_matrix)

# ===================== 6. 保存模型与标准化器为pkl文件 =====================
with open(MODEL_SAVE_FILE, "wb") as f:
    pickle.dump(knn_model, f)
with open(SCALER_SAVE_FILE, "wb") as f:
    pickle.dump(scaler, f)

print(f"KNN模型已保存至：{MODEL_SAVE_FILE}")
print(f"特征标准化器已保存至：{SCALER_SAVE_FILE}")

# ===================== 7. 模型加载验证 =====================
print("正在验证模型保存与加载效果...")
with open(MODEL_SAVE_FILE, "rb") as f:
    loaded_model = pickle.load(f)
with open(SCALER_SAVE_FILE, "rb") as f:
    loaded_scaler = pickle.load(f)

X_test_loaded_scaled = loaded_scaler.transform(X_test)
y_pred_loaded = loaded_model.predict(X_test_loaded_scaled)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)

print(f"模型加载后测试准确率：{loaded_accuracy}，模型保存&加载验证成功！")