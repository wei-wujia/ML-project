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
from sklearn.model_selection import train_test_split
from skimage.feature import hog
# 导入你src文件夹里写好的SVM类
from src.SVM import SVM

# 路径配置
DATA_DIR = os.path.join(project_root, "data", "mnist")  # mnist 0/1文件夹所在路径
MODELS_SAVE_DIR = os.path.join(project_root, "models")    # 模型保存目录

# 模型命名（严格遵循项目规范：算法_数据_日期.pkl）
ALGO_NAME = "svm"
DATA_NAME = "mnist"
from datetime import datetime
DATE = datetime.now().strftime("%Y%m%d")
MODEL_FILENAME = f"{ALGO_NAME}_{DATA_NAME}_{DATE}.pkl"

# 二分类配置（0文件夹→标签0，1文件夹→标签1，和SVM类要求完全对齐）
CLASS_FOLDERS = {
    0: "0",
    1: "1"
}

# 数据集划分配置
TEST_SIZE = 0.2  # 测试集占比20%
RANDOM_STATE = 42  # 随机种子，保证结果可复现

# SVM超参数配置（和你src/SVM.py完全对齐，课堂可现场调整）
SVM_C = 1.0                  # 惩罚系数
SVM_KERNEL = 'linear'        # 核函数，默认线性核（标准SVM），可选'rbf'高斯核
SVM_MAX_ITER = 1000          # 最大迭代次数
SVM_TOL = 1e-3                # 收敛阈值
SVM_SIGMA = 1.0               # 仅rbf核生效的带宽参数

# ====================== 核心函数：读取mnist 0/1文件夹下的图片 ======================
def load_mnist_images(data_root, class_folders):
    """
    读取 data/mnist/0/ 和 1/ 下的28*28图片
    :param data_root: data/mnist 根目录
    :param class_folders: 标签-文件夹名映射
    :return: 图片特征数组、标签数组
    """
    images = []
    labels = []

    # 遍历每个类别文件夹
    for label, folder_name in class_folders.items():
        folder_path = os.path.join(data_root, folder_name)
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"未找到类别文件夹：{folder_path}\n请确认 data/mnist 下包含 0 和 1 两个文件夹")
        
        print(f"正在读取文件夹：{folder_path} (标签：{label})")
        # 遍历文件夹下的所有图片文件
        for filename in os.listdir(folder_path):
            # 仅处理常见图片格式
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, filename)
                try:
                    # 读取图片并转为灰度图
                    img = Image.open(img_path).convert('L')
                    # 确保图片是28*28
                    if img.size != (28, 28):
                        img = img.resize((28, 28), Image.Resampling.LANCZOS)
                    # 转为numpy数组并展平为一维向量
                    img_array = np.array(img, dtype=np.float32).flatten()
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"跳过损坏文件：{filename}，错误：{e}")

    if len(images) == 0:
        raise ValueError(f"在 {data_root} 下未找到任何有效图片，请检查文件夹结构")

    print(f"图片读取完成！共读取 {len(images)} 张图片")
    return np.array(images), np.array(labels)

# ====================== 核心函数：提取HOG特征（和项目其他算法完全统一）======================
def extract_mnist_hog_features(images, img_shape=(28,28)):
    """
    把mnist原始图像转为一维HOG特征向量，符合项目输入要求
    :param images: 展平的图片数组
    :param img_shape: 图片原始尺寸28*28
    :return: HOG特征数组
    """
    hog_features = []
    print("正在提取 HOG 特征...")
    for img in images:
        img_2d = img.reshape(img_shape)
        # 提取HOG特征，和项目内其他算法的参数完全对齐
        feature = hog(
            img_2d,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )
        hog_features.append(feature)
    print(f"HOG 特征提取完成！特征维度：{np.array(hog_features).shape[1]}")
    return np.array(hog_features)

# ====================== 主流程：训练+保存模型 ======================
if __name__ == "__main__":
    # 1. 读取 data/mnist/0/ 和 1/ 下的图片
    X_all, y_all = load_mnist_images(DATA_DIR, CLASS_FOLDERS)

    # 2. 自动划分训练集和测试集（分层抽样，保证类别比例一致）
    print(f"\n正在划分训练集和测试集（测试集占比：{TEST_SIZE*100}%）...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all
    )
    print(f"数据集划分完成！训练集：{len(X_train_raw)} 张，测试集：{len(X_test_raw)} 张")

    # 3. 提取HOG特征（输入为特征向量，完全符合项目要求）
    X_train = extract_mnist_hog_features(X_train_raw)
    X_test = extract_mnist_hog_features(X_test_raw)

    # 4. 初始化SVM模型（和你src/SVM.py的接口完全对齐）
    print(f"\n开始训练SVM模型，核函数：{SVM_KERNEL}，惩罚系数C={SVM_C}")
    model = SVM(
        C=SVM_C,
        kernel=SVM_KERNEL,
        max_iter=SVM_MAX_ITER,
        tol=SVM_TOL,
        sigma=SVM_SIGMA
    )

    # 5. 执行训练
    model.fit(X_train, y_train)

    # 6. 模型评估（控制台打印，方便课堂演示）
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"训练完成！")
    print(f"训练集准确率：{train_acc}")
    print(f"测试集准确率：{test_acc}")

    # 7. 保存模型到models文件夹，严格遵循项目命名规范
    model_save_path = os.path.join(MODELS_SAVE_DIR, MODEL_FILENAME)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\n模型已成功保存到：{model_save_path}")