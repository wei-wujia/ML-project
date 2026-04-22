import os
import pickle
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 路径配置（完全适配你的项目结构）
DATA_PATH = "./data/cifar10_images/"  # CIFAR-10数据集根路径（内含train/test文件夹）
MODELS_SAVE_PATH = "./models/"         # 模型pkl保存路径
# 命名配置（严格遵循项目规范：算法_数据_日期.pkl）
ALGORITHM_NAME = "knn"
DATA_NAME = "cifar10"
DATE = "20260421"  # 可修改为实际运行日期，和项目内其他模型日期保持一致
# KNN模型超参数
N_NEIGHBORS = 5  # 近邻数，CIFAR-10推荐3/5/7，5为通用最优值
# CIFAR-10官方10个类别（和你的文件夹名一一对应，顺序对应0-9标签）
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 模型与标准化器的保存文件名（完全符合命名要求）
MODEL_SAVE_FILE = os.path.join(MODELS_SAVE_PATH, f"{ALGORITHM_NAME}_{DATA_NAME}_{DATE}.pkl")
SCALER_SAVE_FILE = os.path.join(MODELS_SAVE_PATH, f"{ALGORITHM_NAME}_{DATA_NAME}_scaler_{DATE}.pkl")

# ===================== 1. 初始化环境 =====================
os.makedirs(MODELS_SAVE_PATH, exist_ok=True)

# ===================== 2. 适配本地文件夹的CIFAR-10图片加载 =====================
def load_cifar10_split_from_folders(data_root):
    """
    适配你的CIFAR-10文件夹结构：
    data_root/
        ├── train/
        │   ├── airplane/ (数字0的训练图片)
        │   ├── automobile/ (数字1的训练图片)
        │   └── ... (共10个类别文件夹)
        └── test/
            ├── airplane/ (数字0的测试图片)
            ├── automobile/ (数字1的测试图片)
            └── ... (共10个类别文件夹)
    图片要求：32*32彩色图，支持png/jpg等常见格式
    """
    print(f"正在从文件夹加载CIFAR-10数据集：{data_root}")
    
    def load_split(split_name):
        """加载train或test单份数据集"""
        split_path = os.path.join(data_root, split_name)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"未找到{split_name}文件夹：{split_path}")
        
        images = []
        labels = []
        
        # 遍历10个类别，映射为0-9标签
        for label_idx, class_name in enumerate(CIFAR10_CLASSES):
            class_folder = os.path.join(split_path, class_name)
            if not os.path.exists(class_folder):
                print(f"警告：未找到类别 {class_name} 的文件夹 {class_folder}，将跳过该类别")
                continue
            
            # 遍历类别文件夹下的所有图片
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                # 跳过非文件和非图片后缀
                if not os.path.isfile(img_path) or img_name.lower().endswith(('.txt', '.csv', '.md')):
                    continue
                
                try:
                    # 读取图片：强制转为RGB三通道→resize为32*32→展平为3072维特征向量
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((32, 32))  # 确保尺寸统一为CIFAR-10标准32*32
                    img_array = np.array(img).flatten()  # 32*32*3=3072维一维向量
                    
                    images.append(img_array)
                    labels.append(label_idx)
                except Exception as e:
                    print(f"跳过无法读取的图片：{img_path}，错误：{e}")
                    continue
        
        return np.array(images), np.array(labels)
    
    # 分别加载训练集和测试集
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    
    if len(X_train) == 0 or len(X_test) == 0:
        raise FileNotFoundError(f"未在 {data_root} 下找到有效训练/测试图片，请检查文件夹结构！")
    
    print(f"CIFAR-10数据集加载完成！")
    print(f"训练集：{len(X_train)}张 | 测试集：{len(X_test)}张 | 类别数：{len(np.unique(y_train))}")
    return X_train, y_train, X_test, y_test

# 执行数据加载，本地加载失败自动兜底官方数据集
try:
    X_train, y_train, X_test, y_test = load_cifar10_split_from_folders(DATA_PATH)
except Exception as e:
    print(f"本地图片加载失败：{e}，正在加载官方CIFAR-10数据集...")
    from sklearn.datasets import fetch_openml
    # 加载官方CIFAR-10数据集，自动划分训练集50000、测试集10000
    X, y = fetch_openml('CIFAR-10', version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int32)
    X_train, X_test = X[:50000], X[50000:]
    y_train, y_test = y[:50000], y[50000:]
    print(f"官方数据集加载完成 | 训练集：{len(X_train)}条 | 测试集：{len(X_test)}条")

# ===================== 3. 数据标准化（KNN对距离敏感，必做）=====================
print("正在进行数据标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("数据标准化完成！")

# ===================== 4. 训练KNN 10分类模型 =====================
print(f"正在训练KNN模型，近邻数n_neighbors={N_NEIGHBORS}...")
# n_jobs=-1 调用所有CPU核心加速，适配CIFAR-10大数据量
knn_model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)
print("KNN模型训练完成！")

# ===================== 5. 模型测试与效果评估（适配老师当堂查看）=====================
print("正在执行测试集预测与模型评估...")
y_pred = knn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=CIFAR10_CLASSES)

# 打印核心评估结果，和项目内其他脚本格式统一
print(f"测试集整体准确率：{accuracy}")
print("【测试集混淆矩阵】")
print(conf_matrix)

# ===================== 6. 保存模型与标准化器为pkl文件（严格遵循命名规范）=====================
with open(MODEL_SAVE_FILE, "wb") as f:
    pickle.dump(knn_model, f)
with open(SCALER_SAVE_FILE, "wb") as f:
    pickle.dump(scaler, f)

print(f"KNN模型已保存至：{MODEL_SAVE_FILE}")
print(f"特征标准化器已保存至：{SCALER_SAVE_FILE}")

# ===================== 7. 模型加载验证（确保pkl可用，满足现场操作要求）=====================
print("正在验证模型保存与加载效果...")
# 加载刚保存的pkl文件
with open(MODEL_SAVE_FILE, "rb") as f:
    loaded_model = pickle.load(f)
with open(SCALER_SAVE_FILE, "rb") as f:
    loaded_scaler = pickle.load(f)

# 用加载的模型重新预测，验证一致性
X_test_loaded_scaled = loaded_scaler.transform(X_test)
y_pred_loaded = loaded_model.predict(X_test_loaded_scaled)
loaded_accuracy = accuracy_score(y_test, y_pred_loaded)

print(f"模型加载后测试集准确率：{loaded_accuracy}")