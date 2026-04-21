# 标准SVM支持向量机二分类算法实现
# 基于SMO序列最小优化算法求解，输入为特征向量形式，接口与项目内其他算法完全统一
import numpy as np

class SVM:
    """
    标准SVM二分类器
    适配项目任务：titanic二分类、mnist/cifar10指定两类的二分类（含HOG特征输入）
    支持线性核（默认标准SVM）、高斯核，可自定义超参数，课堂演示可调参
    """
    def __init__(self, C=1.0, kernel='linear', max_iter=1000, tol=1e-3, sigma=1.0):
        """
        初始化SVM分类器
        :param C: 惩罚系数，控制对误分类样本的惩罚程度，默认1.0
        :param kernel: 核函数类型，可选 'linear'(线性核，标准SVM，默认) / 'rbf'(高斯核)
        :param max_iter: 最大迭代次数，默认1000
        :param tol: 收敛阈值，迭代变化小于该值则停止优化，默认1e-3
        :param sigma: 高斯核的带宽参数，仅rbf核生效，默认1.0
        """
        # 基础参数校验
        valid_kernels = ['linear', 'rbf']
        if kernel not in valid_kernels:
            raise ValueError(f"核函数仅支持{valid_kernels}，当前输入为{kernel}")
        if C <= 0:
            raise ValueError("惩罚系数C必须大于0")
        
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.sigma = sigma

        # 模型核心参数（训练后生成）
        self.alpha = None  # 拉格朗日乘子
        self.b = 0  # 偏置项
        self.support_vectors = None  # 支持向量特征矩阵
        self.support_labels = None  # 支持向量标签
        self.X_train = None  # 训练集特征（核函数计算用）
        self.y_train = None  # 训练集标签（核函数计算用）

    def _kernel(self, x1, x2):
        """
        内部方法：核函数计算
        :param x1: 特征向量1
        :param x2: 特征向量2
        :return: 核函数映射后的结果
        """
        if self.kernel == 'linear':
            # 线性核：标准SVM，<x1, x2>
            return np.dot(x1, x2.T)
        elif self.kernel == 'rbf':
            # 高斯核(RBF)：exp(-||x1-x2||² / (2σ²))
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)
            dist = np.sum((x1[:, np.newaxis] - x2) ** 2, axis=2)
            return np.exp(-dist / (2 * self.sigma ** 2))

    def fit(self, X_train, y_train):
        """
        模型训练方法，基于SMO序列最小优化算法求解SVM参数
        :param X_train: 训练集特征矩阵，shape=(样本数, 特征数)，每行对应一个样本的特征向量
        :param y_train: 训练集标签向量，shape=(样本数,)，仅支持0/1二分类标签
        :return: 模型自身实例，适配链式调用
        """
        # 数据格式转换与二分类校验（和逻辑回归接口统一，仅支持0/1二分类）
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)

        # 二分类合法性校验
        unique_labels = np.unique(y_train)
        if len(unique_labels) != 2:
            raise ValueError(f"SVM仅支持二分类任务，当前输入标签包含{len(unique_labels)}个类别，请仅保留2个类别")
        if not np.array_equal(unique_labels, [0, 1]):
            raise ValueError("标签必须为0和1的二值格式，请先完成标签映射")

        # SVM标签转换：将0/1转换为-1/1，符合SVM公式要求
        y = np.where(y_train == 0, -1, 1)
        n_samples, n_features = X_train.shape

        # 初始化参数
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X_train = X_train
        self.y_train = y

        # 预计算核矩阵，提升迭代效率
        kernel_matrix = self._kernel(X_train, X_train)

        # SMO序列最小优化算法核心迭代
        iter_count = 0
        while iter_count < self.max_iter:
            alpha_changed = 0
            # 遍历所有样本，优化每个alpha_i
            for i in range(n_samples):
                # 计算样本i的预测值与误差
                g_i = np.sum(self.alpha * y * kernel_matrix[i, :]) + self.b
                e_i = g_i - y[i]

                # 检验KKT条件，判断是否需要优化该alpha
                if (y[i] * e_i < -self.tol and self.alpha[i] < self.C) or (y[i] * e_i > self.tol and self.alpha[i] > 0):
                    # 随机选择第二个优化变量alpha_j，和alpha_i成对优化
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    g_j = np.sum(self.alpha * y * kernel_matrix[j, :]) + self.b
                    e_j = g_j - y[j]

                    # 保存优化前的alpha旧值
                    alpha_i_old = self.alpha[i].copy()
                    alpha_j_old = self.alpha[j].copy()

                    # 计算alpha_j的上下界L和H
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    if L == H:
                        continue

                    # 计算eta
                    eta = 2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                    if eta >= 0:
                        continue

                    # 优化alpha_j
                    self.alpha[j] -= y[j] * (e_i - e_j) / eta
                    # 裁剪alpha_j到[L, H]范围内
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # 检查alpha_j变化量，变化过小则跳过本次优化
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    # 同步优化alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 更新偏置项b
                    b1 = self.b - e_i - y[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, i] - y[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[i, j]
                    b2 = self.b - e_j - y[i] * (self.alpha[i] - alpha_i_old) * kernel_matrix[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * kernel_matrix[j, j]
                    # 按alpha范围更新b
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    alpha_changed += 1

            # 迭代终止判断：本轮无alpha更新，已收敛
            if alpha_changed == 0:
                break
            iter_count += 1

        # 提取支持向量（alpha>0的样本即为支持向量）
        support_idx = self.alpha > 1e-5
        self.support_vectors = X_train[support_idx]
        self.support_labels = y[support_idx]
        self.alpha = self.alpha[support_idx]

        return self

    def predict(self, X_test):
        """
        模型预测方法，输出测试集的预测分类标签（0/1），和项目内其他分类器接口完全统一
        :param X_test: 测试集特征矩阵，shape=(测试样本数, 特征数)，每行对应一个样本的特征向量
        :return: 预测标签数组，shape=(测试样本数,)，取值0/1
        """
        # 预检查：必须先训练再预测
        if self.support_vectors is None:
            raise RuntimeError("请先调用 fit() 方法完成模型训练后再执行预测")

        X_test = np.array(X_test, dtype=np.float32)
        # 计算核函数映射结果
        kernel_result = self._kernel(X_test, self.support_vectors)
        # 计算预测值
        y_pred = np.sum(self.alpha * self.support_labels * kernel_result, axis=1) + self.b
        # 符号函数转换为-1/1，再映射回0/1标签，和输入格式统一
        y_pred = np.where(np.sign(y_pred) == -1, 0, 1)
        return y_pred

    def score(self, X_test, y_test):
        """
        模型评估方法，计算测试集分类准确率，和项目内其他分类器接口完全统一
        :param X_test: 测试集特征矩阵
        :param y_test: 测试集真实标签（0/1）
        :return: 准确率 accuracy，取值范围 [0, 1]
        """
        y_pred = self.predict(X_test)
        y_true = np.array(y_test)
        # 计算预测正确的样本占比
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        return accuracy