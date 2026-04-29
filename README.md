# ML-project 机器学习算法框架
对外经济贸易大学AIT209-机器学习与项目实践 课程工程文件

本项目是一个基于NumPy实现的机器学习算法库，包含线性回归、逻辑回归、KNN、SVM 和人工神经网络（ANN），支持选择回归、二分类和多分类任务。
项目采用统一的命令行接口设计，用户可手动选择调用的机器学习算法和数据集，并且支持调整算法参数，一键完成数据加载、模型训练、评估和结果保存，内置有Titanic、波士顿房价、MNIST和CIFAR-10四种数据集。

## ✨ 项目特点
- **纯手写实现**：所有算法基于 NumPy 从零实现，包含完整的前向传播、反向传播和梯度下降逻辑
- **统一接口**：所有算法遵循相同的 `fit(X, y)` / `predict(X)` / `score(X, y)` 接口
- **多任务支持**：支持回归（房价预测）、二分类（Titanic 生存预测）和多分类（MNIST/CIFAR-10 图像分类）
- **自动工程**：自动创建文件夹、保存模型和标准化器、生成训练日志和损失曲线
- **灵活配置**：通过命令行参数自由调整算法超参数，无需修改代码
- **正则化支持**：ANN 实现了 Dropout 正则化，有效防止过拟合

📁 项目结构
ML-project/
├── data/                     # 数据集目录
│   ├── titanic/              # Titanic 数据集
│   │   ├── titanic_train.csv
│   │   └── titanic_test.csv
│   ├── house/                # 波士顿房价数据集
│   │   └── house_data.csv
│   ├── mnist/                # MNIST 手写数字数据集（按类别分文件夹）
│   │   ├── 0/
│   │   ├── 1/
│   └── cifar10_images/       # CIFAR-10 图像数据集
│       ├── train/
│       │   ├── airplane/
│       │   └── ...
│       └── test/
│           ├── airplane/
│           └── ...
├── src/                      # 算法实现目录
│   ├── linear_regression.py  # 线性回归（最小二乘法+梯度下降）
│   ├── logistic_regression.py# 逻辑回归
│   ├── KNN.py                # K近邻分类
│   ├── SVM.py                # 支持向量机（基于SMO算法）
│   └── ANN.py                # 人工神经网络（含Dropout）
├── models/                   # 训练好的模型和标准化器（自动生成）
├── results/                  # 训练结果（日志+损失曲线，自动生成）
└── main.py                   # 统一调度入口
└──

🛠️ 环境依赖
pip install numpy pandas opencv-python matplotlib scikit-learn

🚀 快速开始
1. 克隆仓库
git clone https://github.com/你的用户名/ML-project.git
cd ML-project

2. 准备数据集
将对应的数据集放入 `data/` 目录下，保持上述项目结构即可。

3. 运行第一个示例（线性回归预测房价）
python main.py --algo linear --data house --process train

运行完成后，模型会自动保存到 `models/` 目录，训练日志和损失曲线会保存到 `results/` 目录。

📖 详细使用指南
通用参数说明
| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--algo` | 选择算法 | `knn` / `linear` / `logistic` / `svm` / `ann` |
| `--data` | 选择数据集 | `titanic` / `house` / `mnist` / `cifar10` |
| `--process` | 执行流程 | `train`（训练）/ `test`（测试） |
| `--model_path` | 测试时指定模型路径 | 可选，不填则自动加载最新模型 |

1. 线性回归（房价预测）
# 最小二乘法
python main.py --algo linear --data house --process train

# 梯度下降法
python main.py --algo linear --data house --process train --lr_method gradient_descent --learning_rate 0.001

2. 逻辑回归
# Titanic 二分类
python main.py --algo logistic --data titanic --process train

# MNIST 0和1二分类
python main.py --algo logistic --data mnist --process train --mnist_class1 0 --mnist_class2 1

3. KNN
# Titanic 二分类
python main.py --algo knn --data titanic --process train --n_neighbors 5

# CIFAR-10 十分类
python main.py --algo knn --data cifar10 --process train --cifar10_multi

4. SVM

# Titanic 二分类（线性核）
python main.py --algo svm --data titanic --process train --kernel linear --C 1.0

# CIFAR-10 cat vs dog 二分类（RBF核）
python main.py --algo svm --data cifar10 --process train --cifar10_class1 cat --cifar10_class2 dog --kernel rbf

5. 人工神经网络（ANN）
# 房价回归预测
python main.py --algo ann --data house --process train

# Titanic 二分类（带Dropout）
python main.py --algo ann --data titanic --process train --dropout_rate 0.2 --learning_rate 0.001

# CIFAR-10 十分类（推荐配置）
python main.py --algo ann --data cifar10 --process train --cifar10_multi

6. 测试已训练模型
# 自动加载最新模型
python main.py --algo ann --data titanic --process test

# 指定模型路径
python main.py --algo linear --data house --process test --model_path models/linear_house_20260428.pkl
