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
from PIL import Image
from skimage.feature import hog
# 导入你src文件夹里写好的线性回归类
from src.linear_regression import LinearRegression

# 路径配置
DATA_DIR = os.path.join(project_root, "data", "cifar10_images")  # cifar10图片根目录
MODELS_SAVE_DIR = os.path.join(project_root, "models")            # 模型保存目录

# 模型命名（严格遵循项目规范：算法_数据_日期.pkl）
ALGO_NAME = "linear"
DATA_NAME = "cifar"
from datetime import datetime
DATE = datetime.now().strftime("%Y%m%d")
MODEL_FILENAME = f"{ALGO_NAME}_{DATA_NAME}_{DATE}.pkl"

# cifar10 类别-标签映射（固定对应，和官方cifar10完全一致）
CLASS_LABEL_MAP = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}

# 线性回归超参数配置（和你linear_regression.py完全对齐，可现场调整）
# 可选求解方式：normal_equation(最小二乘闭式解，默认) / gradient_descent(梯度下降)
SOLVE_METHOD = "normal_equation"
LEARNING_RATE = 0.001    # 仅梯度下降模式生效
N_ITERATIONS = 1000       # 仅梯度下降模式生效
BATCH_SIZE = 64            # 仅梯度下降模式生效，可自定义输入

# ====================== 核心函数：读取cifar10图片 ======================
def load_cifar_images(data_root, is_train=True):
    """
    读取cifar10图片数据
    :param data_root: cifar10_images根目录
    :param is_train: 是否读取训练集（train文件夹），False则读取测试集（test文件夹）
    :return: 图片特征数组、标签数组
    """
    images = []
    labels = []
    # 选择train/test文件夹
    split_folder = "train" if is_train else "test"
    split_dir = os.path.join(data_root, split_folder)

    # 检查文件夹是否存在
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"找不到文件夹：{split_dir}，请确认cifar10数据集路径正确")

    print(f"正在读取 {split_folder} 集图片...")
    # 遍历每个类别文件夹
    for class_name, label in CLASS_LABEL_MAP.items():
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"警告：跳过不存在的类别文件夹 {class_dir}")
            continue

        # 遍历文件夹下的所有图片
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(class_dir, filename)
                try:
                    # 读取图片→转灰度图→确保尺寸32x32（cifar10标准尺寸）
                    img = Image.open(img_path).convert('L')
                    if img.size != (32, 32):
                        img = img.resize((32, 32), Image.Resampling.LANCZOS)
                    # 转为numpy数组展平
                    img_array = np.array(img, dtype=np.float32).flatten()
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"跳过损坏文件 {filename}，错误：{e}")

    print(f"{split_folder}集读取完成！共 {len(images)} 张图片")
    return np.array(images), np.array(labels, dtype=np.float32)

# ====================== 核心函数：提取HOG特征（和项目其他算法统一）======================
def extract_cifar_hog_features(images, img_shape=(32, 32)):
    """
    为cifar10图片提取HOG特征，输入为特征向量，和项目要求完全对齐
    :param images: 展平的图片数组
    :param img_shape: 图片原始尺寸32x32
    :return: HOG特征数组
    """
    hog_features = []
    print("正在提取HOG特征...")
    for img in images:
        # 把展平的向量还原为2D图片
        img_2d = img.reshape(img_shape)
        # 提取HOG特征（适配32x32尺寸，和mnist的HOG逻辑统一）
        feature = hog(
            img_2d,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )
        hog_features.append(feature)
    print(f"HOG特征提取完成！特征维度：{np.array(hog_features).shape[1]}")
    return np.array(hog_features)

# ====================== 主流程：训练+保存模型 ======================
if __name__ == "__main__":
    # 1. 读取训练集和测试集
    X_train_raw, y_train = load_cifar_images(DATA_DIR, is_train=True)
    X_test_raw, y_test = load_cifar_images(DATA_DIR, is_train=False)

    # 2. 提取HOG特征（输入为特征向量，符合项目要求）
    X_train = extract_cifar_hog_features(X_train_raw)
    X_test = extract_cifar_hog_features(X_test_raw)

    # 3. 初始化线性回归模型（和你src里的类完全对齐）
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

    # 4. 执行训练
    model.fit(X_train, y_train)

    # 5. 模型评估（线性回归用R²决定系数，和你代码里的score方法对齐）
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