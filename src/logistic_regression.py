# 标准逻辑回归二分类算法实现
# 仅支持二分类任务，输入为特征向量形式，接口与项目内其他算法完全统一
import numpy as np

class LogisticRegression:
    """
    标准逻辑回归二分类器
    适配项目任务：titanic二分类、mnist/cifar10指定两类的二分类
    采用批量梯度下降法优化对数似然损失，支持自定义超参数
    """
    def __init__(self, learning_rate=0.001, n_iterations=1000, fit_intercept=True):
        """
        初始化逻辑回归分类器
        :param learning_rate: 梯度下降学习率，默认0.001
        :param n_iterations: 梯度下降迭代轮数，默认1000
        :param fit_intercept: 是否拟合偏置项（截距），默认True
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        # 模型参数：权重+偏置
        self.weights = None
        self.intercept = None

    def _sigmoid(self, z):
        """
        内部方法：sigmoid激活函数，做数值稳定处理，避免溢出
        :param z: 线性变换输出
        :return: sigmoid映射结果，范围(0,1)
        """
        # 数值稳定优化：避免exp(z)过大/过小导致溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        """
        内部方法：给特征矩阵添加截距项（全1列）
        :param X: 原始特征矩阵
        :return: 新增截距项后的特征矩阵
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit(self, X_train, y_train):
        """
        模型训练方法，执行梯度下降优化参数
        :param X_train: 训练集特征矩阵，shape=(样本数, 特征数)，每行对应一个样本的特征向量
        :param y_train: 训练集标签向量，shape=(样本数,)，仅支持0/1二分类标签
        :return: 模型自身实例，适配链式调用
        """
        # 数据格式转换与校验
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train)

        # 二分类校验：严格限制仅支持二分类任务，符合项目要求
        unique_labels = np.unique(y_train)
        if len(unique_labels) != 2:
            raise ValueError(f"逻辑回归仅支持二分类任务，当前输入标签包含{len(unique_labels)}个类别，请仅保留2个类别")
        if not np.array_equal(unique_labels, [0, 1]):
            raise ValueError("标签必须为0和1的二值格式，请先完成标签映射")

        # 处理截距项
        if self.fit_intercept:
            X = self._add_intercept(X_train)
        else:
            X = X_train

        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # 批量梯度下降迭代优化
        for _ in range(self.n_iterations):
            # 线性预测 + sigmoid映射
            z = np.dot(X, self.weights)
            y_pred_proba = self._sigmoid(z)
            # 计算梯度（对数似然损失的梯度）
            gradient = np.dot(X.T, (y_pred_proba - y_train)) / n_samples
            # 更新参数
            self.weights -= self.learning_rate * gradient

        # 拆分权重与截距
        if self.fit_intercept:
            self.intercept = self.weights[0]
            self.weights = self.weights[1:]
        else:
            self.intercept = 0.0

        return self

    def predict_proba(self, X_test):
        """
        预测样本为正类(1)的概率
        :param X_test: 测试集特征矩阵，shape=(测试样本数, 特征数)
        :return: 正类概率数组，shape=(测试样本数,)
        """
        # 预检查：必须先训练再预测
        if self.weights is None:
            raise RuntimeError("请先调用 fit() 方法完成模型训练后再执行预测")

        X_test = np.array(X_test, dtype=np.float32)
        # 线性计算
        z = np.dot(X_test, self.weights) + self.intercept
        # 输出正类概率
        return self._sigmoid(z)

    def predict(self, X_test, threshold=0.5):
        """
        模型预测方法，输出测试集的预测分类标签（0/1）
        :param X_test: 测试集特征矩阵，shape=(测试样本数, 特征数)
        :param threshold: 分类阈值，默认0.5，大于阈值判为正类1
        :return: 预测标签数组，shape=(测试样本数,)，取值0/1
        """
        y_proba = self.predict_proba(X_test)
        # 按阈值二值化
        return np.where(y_proba >= threshold, 1, 0)

    def score(self, X_test, y_test):
        """
        模型评估方法，计算测试集分类准确率
        :param X_test: 测试集特征矩阵
        :param y_test: 测试集真实标签（0/1）
        :return: 准确率 accuracy，取值范围 [0, 1]
        """
        y_pred = self.predict(X_test)
        y_true = np.array(y_test)
        # 计算预测正确的样本占比
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        return accuracy