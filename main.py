import argparse
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data")
SRC_PATH = os.path.join(ROOT_PATH, "src")
MODELS_PATH = os.path.join(ROOT_PATH, "models")
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
# 自动创建缺失的文件夹
for path in [MODELS_PATH, RESULTS_PATH]:
    if not os.path.exists(path):
        os.makedirs(path)

import sys
sys.path.append(SRC_PATH)
from KNN import KNN
from linear_regression import LinearRegression
from logistic_regression import LogisticRegression
from SVM import SVM
from ANN import ANN
# ===================== 命令行参数解析 =====================
def parse_args():
    '''
    1.线性回归
        #线性回归做house房价预测（最小二乘法）
        python main.py --algo linear --data house --process train
        #线性回归做house房价预测（梯度下降法）
        python main.py --algo linear --data house --process train --lr_method gradient_descent --learning_rate 0.001
    2.逻辑回归
        #逻辑回归做titanic二分类训练
        python main.py --algo logistic --data titanic --process train
        #逻辑回归做mnist选0和1两个数字二分类训练
        python main.py --algo logistic --data mnist --process train --mnist_class1 0 --mnist_class2 1
        #逻辑回归做cifar10自选ship和truck两个类别二分类训练
        python main.py --algo logistic --data cifar10 --process train --cifar10_class1 ship --cifar10_class2 truck
    3.KNN
        #KNN做titanic二分类
        python main.py --algo knn --data titanic --process train
        #KNN做mnist全10分类训练
        python main.py --algo knn --data mnist --process train
        #KNN做cifar10全10分类训练
        python main.py --algo knn --data cifar10 --process train
    4.SVM
        #SVM做titanic二分类训练
        python main.py --algo svm --data titanic --process train
        #SVM做mnist选0和1两个数字二分类训练
        python main.py --algo svm --data mnist --process train --mnist_class1 0 --mnist_class2 1
        #SVM做cifar10自选cat和dog两个类别二分类训练
        python main.py --algo svm --data cifar10 --process train --cifar10_class1 cat --cifar10_class2 dog
    5.ANN
        #ANN做house房价预测
        python main.py --algo ann --data house --process train
        #ANN做titanic二分类
        python main.py --algo ann --data titanic --process train
        #ANN做cifar10自选二分类
        python main.py --algo ann --data cifar10 --process train --cifar10_class1 cat --cifar10_class2 dog
        #ANN做cifar10十分类
        python main.py --algo ann --data cifar10 --process train --cifar10_multi

    使用已训练模型测试（自动加载最新模型）
        python main.py --algo {} --data {} --process test (--model path {})
    '''
    parser = argparse.ArgumentParser(description='机器学习项目统一调度入口')
    # 核心必填参数
    parser.add_argument('--algo', type=str, required=True, 
                    choices=['knn', 'linear', 'logistic', 'svm', 'ann'],
                    help='选择算法：knn/linear/logistic/svm/ann')
    parser.add_argument('--data', type=str, required=True,
                        choices=['titanic', 'house', 'mnist', 'cifar10'],
                        help='选择数据集：titanic/house/mnist/cifar10')
    parser.add_argument('--process', type=str, required=True,
                        choices=['train', 'test'],
                        help='执行流程：train(训练)/test(测试)')
    
    # 可选参数 - 二分类类别自定义（支持mnist/cifar10自选2类）
    parser.add_argument('--model_path', type=str, default=None,
                        help='测试模式专用：指定已训练模型pkl文件路径，不填则自动匹配默认命名文件')
    # mnist二分类自定义类别（10类中任选2个）
    parser.add_argument('--mnist_class1', type=str, default='0',
                        help='mnist二分类专用：指定第一个数字类别(0-9)，默认0')
    parser.add_argument('--mnist_class2', type=str, default='1',
                        help='mnist二分类专用：指定第二个数字类别(0-9)，默认1')
    # cifar10二分类自定义类别（10类中任选2个）
    # airplane automobile bird cat deer dog frog horse ship truck
    parser.add_argument('--cifar10_class1', type=str, default='airplane',
                        help='cifar10二分类专用：指定第一个类别名，默认airplane')
    parser.add_argument('--cifar10_class2', type=str, default='automobile',
                        help='cifar10二分类专用：指定第二个类别名，默认automobile')
    # 十分类开关
    parser.add_argument('--cifar10_multi', action='store_true', default=False,
                    help='cifar10启用10分类模式，开启后忽略class1/class2，加载全部10个类别')
    
    # 算法超参数可选配置（与原算法接口完全对齐）
    parser.add_argument('--n_neighbors', type=int, default=5, help='KNN近邻数')
    parser.add_argument('--C', type=float, default=1.0, help='SVM惩罚系数')
    parser.add_argument('--kernel', type=str, default='linear', help='SVM核函数')
    parser.add_argument('--lr_method', type=str, default='normal_equation', 
                        help='线性回归求解方式：normal_equation/gradient_descent')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='梯度下降学习率')
    parser.add_argument('--n_iterations', type=int, default=1000, help='梯度下降迭代轮数')
    parser.add_argument('--hidden_layers', type=str, default='64,32', 
                        help='ANN隐藏层结构，逗号分隔，如"64,32"表示两个隐藏层')
    parser.add_argument('--activation', type=str, default='relu', 
                        help='ANN隐藏层激活函数：relu/sigmoid/tanh')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='ANN Dropout比率，0.0-1.0之间，0表示不使用Dropout，推荐0.2-0.5')
    return parser.parse_args()


def load_titanic():
    """加载titanic（完全适配你的CSV：标签=2urvived，自动清无效列/NaN/方差0列）"""
    train_path = os.path.join(DATA_PATH, "titanic", "titanic_train.csv")
    test_path = os.path.join(DATA_PATH, "titanic", "titanic_test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练集：{train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试集：{test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 统一预处理函数
    def process(df):
        # 1. 删掉无用列：Passengerid + 所有 zero 列
        drop_cols = ["Passengerid"] + [c for c in df.columns if str(c).startswith("zero")]
        df = df.drop(columns=drop_cols, errors="ignore")

        # 2. 缺失值填充
        df["Age"] = df["Age"].fillna(df["Age"].median())
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

        # 3. 分类特征编码
        if "Sex" in df.columns:
            df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
        if "Embarked" in df.columns:
            df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

        # 4. 删掉方差为0的列（解决标准化警告）
        df = df.loc[:, df.var() > 0]

        # 5. 拆分 X、y（关键：保证 y 是一维数组！）
        y = df["2urvived"].values.ravel()  # ravel() 强制变成一维，杜绝0维
        X = df.drop(columns=["2urvived"], errors="ignore").values

        return X, y

    X_train, y_train = process(train_df)
    X_test, y_test = process(test_df)

    return X_train, X_test, y_train, y_test

def load_house():
    """加载house房价回归数据集"""
    df = pd.read_csv(os.path.join(DATA_PATH, "house", "house_data.csv"))
    # 去除缺失值，分离特征与标签（最后一列为房价）
    df = df.dropna()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_mnist(algo, class1, class2):
    """
    加载mnist数据集
    - 二分类算法(logistic/svm)：加载用户指定的2个数字类别，映射为0/1二分类标签
    - KNN：加载0-9全10个类别，支持10分类任务
    :param algo: 算法类型
    :param class1: 二分类第一个类别
    :param class2: 二分类第二个类别
    """
    data_path = os.path.join(DATA_PATH, "mnist")
    # 获取mnist所有存在的数字类别文件夹（0-9）
    all_classes = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and d.isdigit()]
    all_classes.sort()  # 按数字顺序排序

    # 二分类算法：仅加载用户指定的2个类别
    if algo in ['logistic', 'svm']:
        # 合法性校验
        if class1 not in all_classes or class2 not in all_classes:
            raise ValueError(f"mnist类别不存在，可选数字类别：{all_classes}")
        if class1 == class2:
            raise ValueError("mnist二分类的两个类别不能相同，请输入不同的数字")
        use_classes = [class1, class2]
        class_map = {class1: 0, class2: 1}  # 映射为0/1标签，适配原算法要求
    # KNN：加载全10分类
    else:
        use_classes = all_classes
        class_map = {cls: int(cls) for cls in use_classes}  # 标签为原数字0-9
    
    X = []
    y = []
    # 遍历选中的类别，加载图片数据
    for cls in use_classes:
        img_dir = os.path.join(data_path, cls)
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            # 读取灰度图，resize到28*28，展平为特征向量
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue  # 跳过损坏的图片
            img = cv2.resize(img, (28, 28))
            X.append(img.flatten())
            y.append(class_map[cls])
    
    X = np.array(X)
    y = np.array(y)
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_cifar10(use_multi_class=False, class1=None, class2=None):
    """
    加载cifar10数据集
    :param use_multi_class: True=加载全部10类（十分类），False=二分类
    :param class1: 二分类第一个类别
    :param class2: 二分类第二个类别
    """
    cifar_path = os.path.join(DATA_PATH, "cifar10_images")
    if not os.path.exists(cifar_path):
        raise FileNotFoundError(f"找不到cifar10根目录：{cifar_path}")

    # CIFAR-10 标准10类别列表
    all_classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    # 确定要加载的类别
    if use_multi_class:
        target_classes = all_classes
        print("十分类模式：加载CIFAR-10全部10个类别")
    else:
        # 二分类：校验并加载指定的2个类别
        c1 = class1 if class1 in all_classes else all_classes[0]
        c2 = class2 if class2 in all_classes else all_classes[1]
        target_classes = [c1, c2]
        print(f"二分类模式：加载类别 [{c1}] 和 [{c2}]")

    X = []
    y = []

    # 遍历train/test文件夹，合并后再切分保证分布一致
    for split in ["train", "test"]:
        split_path = os.path.join(cifar_path, split)
        if not os.path.exists(split_path):
            continue

        # 遍历每个类别文件夹
        for cls_idx, cls_name in enumerate(target_classes):
            cls_path = os.path.join(split_path, cls_name)
            if not os.path.exists(cls_path):
                print(f"警告：找不到类别文件夹 {cls_path}，跳过")
                continue

            # 遍历类别文件夹里的所有图片（核心修复：不再读文件夹本身）
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                if not os.path.isfile(img_path):
                    continue

                # 读取图片
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # 展平+归一化到0-1，提升模型效果
                img_flat = img.flatten().astype(np.float32) / 255.0
                X.append(img_flat)
                y.append(cls_idx)  # 十分类时标签0-9，二分类时0/1

    # 转numpy数组，强制标签一维，杜绝索引报错
    X = np.array(X)
    y = np.array(y).ravel()

    if len(X) == 0:
        raise RuntimeError("未加载到任何图片！请检查cifar10_images目录结构")

    print(f"共加载图片：{len(X)} 张，特征维度：{X.shape[1]}")

    # 切分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 数据集加载统一入口
def load_dataset(args):
    if args.data == 'titanic':
        return load_titanic()
    elif args.data == 'house':
        return load_house()
    elif args.data == 'mnist':
        return load_mnist(args.algo, args.mnist_class1, args.mnist_class2)
    elif args.data == 'cifar10':
        return load_cifar10(use_multi_class=args.cifar10_multi, class1=args.cifar10_class1, class2=args.cifar10_class2)
    else:
        raise ValueError(f"不支持的数据集：{args.data}")

# ===================== 算法与数据集适配校验 =====================
def check_algo_data_match(args):
    """校验算法与数据集的适配性，避免非法调用"""
    # 线性回归仅支持house回归数据集
    if args.algo == 'linear' and args.data != 'house':
        raise ValueError(f"线性回归(linear)仅支持house数据集，当前选择：{args.data}")
    
    # 逻辑回归/SVM仅支持二分类数据集，不支持house回归
    if args.algo in ['logistic', 'svm']:
        if args.data == 'house':
            raise ValueError(f"{args.algo}是二分类算法，不支持house回归数据集")
    
    # KNN仅支持分类任务，不支持house回归
    if args.algo == 'knn' and args.data == 'house':
        raise ValueError(f"当前KNN实现为分类算法，不支持house回归数据集")
    
    # ANN支持所有数据集
    if args.algo == 'ann':
        pass

# ===================== 模型初始化 =====================
def init_model(args):
    """根据参数初始化对应算法模型，与原算法接口完全对齐"""
    if args.algo == 'knn':
        return KNN(n_neighbors=args.n_neighbors)
    elif args.algo == 'linear':
        return LinearRegression(
            method=args.lr_method,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations
        )
    elif args.algo == 'logistic':
        return LogisticRegression(
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations
        )
    elif args.algo == 'svm':
        return SVM(C=args.C, kernel=args.kernel, max_iter=args.n_iterations)
    elif args.algo == 'ann':
    # 解析隐藏层结构
        hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
        return ANN(
            hidden_layers=hidden_layers,
            learning_rate=args.learning_rate,
            n_iterations=args.n_iterations,
            activation=args.activation,
            dropout_rate=args.dropout_rate)
    else:
        raise ValueError(f"不支持的算法：{args.algo}")

# ===================== 结果保存模块 =====================
def save_results(args, metrics, loss_history=None):
    """保存准确率/R²到log文件，损失曲线到png文件"""
    # 时间戳命名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.algo}_{args.data}_{current_time}"
    
    # 1. 保存指标log文件
    log_path = os.path.join(RESULTS_PATH, f"{base_name}.log")
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write(f"算法：{args.algo}\n")
        f.write(f"数据集：{args.data}\n")
        # 记录二分类所选类别
        if args.data in ['mnist', 'cifar10'] and args.algo in ['logistic', 'svm']:
            if args.data == 'mnist':
                f.write(f"二分类所选类别：{args.mnist_class1} vs {args.mnist_class2}\n")
            else:
                f.write(f"二分类所选类别：{args.cifar10_class1} vs {args.cifar10_class2}\n")
        f.write(f"执行时间：{current_time}\n")
        f.write(f"执行流程：{args.process}\n")
        f.write("-"*50 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}：{v}\n")
        f.write("="*50 + "\n")
    print(f"指标已保存至：{log_path}")
    
    # 2. 保存损失曲线png（线性回归梯度下降模式和ANN有损失记录）
    if loss_history is not None and len(loss_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Training Loss')
        plt.xlabel('Iteration Epoch')
        plt.ylabel('Loss')
        plt.title(f'{args.algo}_{args.data} Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        img_path = os.path.join(RESULTS_PATH, f"{base_name}_loss.png")
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"损失曲线已保存至：{img_path}")

# ===================== 训练主流程 =====================
def train(args):
    # 1. 加载数据集
    print(f"正在加载数据集：{args.data}")
    X_train, X_test, y_train, y_test = load_dataset(args)
    
    # 2. 特征标准化（提升模型效果，适配所有算法）
    print("正在进行特征标准化")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 初始化模型
    print(f"正在初始化算法：{args.algo}")
    model = init_model(args)
    
    # 4. 模型训练
    print("开始模型训练...")
    model.fit(X_train_scaled, y_train)
    print("模型训练完成")
    
    # 5. 模型评估
    print("正在计算模型指标...")
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    # 分类任务：准确率；回归任务：R²
    metric_name = "准确率" if args.algo != 'linear' else "R²决定系数"
    metrics = {
        f"训练集{metric_name}": train_score,
        f"测试集{metric_name}": test_score
    }
    print(f"训练集{metric_name}：{train_score}")
    print(f"测试集{metric_name}：{test_score}")
    
    # 6. 保存模型与标准化器
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    model_name = f"{args.algo}_{args.data}_{current_date}.pkl"
    model_path = os.path.join(MODELS_PATH, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    scaler_name = f"{args.algo}_{args.data}_scaler_{current_date}.pkl"
    scaler_path = os.path.join(MODELS_PATH, scaler_name)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"模型已保存至：{model_path}")
    print(f"标准化器已保存至：{scaler_path}")
    
    # 7. 保存结果
    loss_history = None
    if (args.algo == 'linear' and args.lr_method == 'gradient_descent') or args.algo == 'ann':
        loss_history = model.loss_history
    save_results(args, metrics, loss_history)

# ===================== 测试主流程 =====================
def test(args):
    # 1. 加载数据集（仅测试集）
    print(f"正在加载数据集：{args.data}")
    _, X_test, _, y_test = load_dataset(args)
    
    # 2. 加载模型与标准化器
    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
        # 自动匹配标准化器路径
        scaler_path = model_path.replace(f"{args.algo}_{args.data}", f"{args.algo}_{args.data}_scaler")
    else:
        # 自动查找最新的模型文件
        model_files = [f for f in os.listdir(MODELS_PATH) if f.startswith(f"{args.algo}_{args.data}") and f.endswith(".pkl") and "scaler" not in f]
        if not model_files:
            raise FileNotFoundError(f"未找到{args.algo}_{args.data}的预训练模型，请先执行训练流程")
        model_files.sort(reverse=True)
        model_path = os.path.join(MODELS_PATH, model_files[0])
        scaler_path = os.path.join(MODELS_PATH, model_files[0].replace(f"{args.algo}_{args.data}", f"{args.algo}_{args.data}_scaler"))
    
    print(f"正在加载模型：{model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"未找到对应的标准化器：{scaler_path}")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 3. 测试集标准化
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 模型预测与评估
    print("开始模型测试...")
    test_score = model.score(X_test_scaled, y_test)
    metric_name = "准确率" if args.algo != 'linear' else "R²决定系数"
    metrics = {f"测试集{metric_name}": test_score}
    print(f"测试集{metric_name}：{test_score}")
    
    # 5. 保存测试结果
    save_results(args, metrics)

# ===================== 主入口 =====================
if __name__ == "__main__":
    args = parse_args()
    # 校验算法与数据集适配性
    check_algo_data_match(args)
    # 执行对应流程
    if args.process == 'train':
        train(args)
    elif args.process == 'test':
        test(args)