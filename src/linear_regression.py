# 标准线性回归算法实现
# 支持两种求解方式：最小二乘闭式解(正规方程)、梯度下降法(支持自定义batch_size)
# 适配项目任务：house房价数据集的训练与预测，输入为特征向量形式
import numpy as np

class LinearRegression:
    """
    标准线性回归器
    适配项目任务：house房价数据回归预测
    支持两种求解模式：最小二乘闭式解、梯度下降法(批量/小批量/随机梯度下降，由batch_size控制)
    """
    def __init__(self, 
                 method='normal_equation', 
                 learning_rate=0.01, 
                 n_iterations=1000, 
                 batch_size=32, 
                 fit_intercept=True):
        """
        初始化线性回归器
        :param method: 求解方式，可选 'normal_equation'(最小二乘闭式解，默认) / 'gradient_descent'(梯度下降法)
        :param learning_rate: 梯度下降学习率，仅梯度下降模式生效，默认0.01
        :param n_iterations: 梯度下降迭代轮数，仅梯度下降模式生效，默认1000
        :param batch_size: 梯度下降批次大小，可自定义输入，仅梯度下降模式生效，默认32
                         batch_size=样本总数 → 批量梯度下降(BGD)
                         batch_size=1 → 随机梯度下降(SGD)
                         1<batch_size<样本总数 → 小批量梯度下降(MBGD)
        :param fit_intercept: 是否拟合偏置项(截距)，默认True
        """
        # 基础参数校验
        valid_methods = ['normal_equation', 'gradient_descent']
        if method not in valid_methods:
            raise ValueError(f"求解方式仅支持{valid_methods}，当前输入为{method}")
        if batch_size < 1:
            raise ValueError("batch_size必须大于等于1")
        
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.fit_intercept = fit_intercept
        
        # 模型参数
        self.weights = None
        self.intercept = None
        # 梯度下降模式的损失记录，用于后续画图保存到results文件夹
        self.loss_history = []

    def _add_intercept(self, X):
        """内部方法：给特征矩阵添加截距项(全1列)"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _normal_equation_fit(self, X, y):
        """内部方法：最小二乘闭式解(正规方程)求解参数"""
        # 正规方程公式：theta = (X^T · X)^-1 · X^T · y
        # 使用伪逆pinv替代逆inv，避免矩阵奇异导致的报错，数值更稳定
        theta = np.linalg.pinv(X.T @ X) @ X.T @ y
        
        # 拆分权重与截距
        if self.fit_intercept:
            self.intercept = theta[0]
            self.weights = theta[1:]
        else:
            self.intercept = 0.0
            self.weights = theta

    def _gradient_descent_fit(self, X, y):
        """内部方法：梯度下降法求解参数，支持自定义batch_size"""
        n_samples, n_features = X.shape
        # 初始化参数
        theta = np.zeros(n_features)
        self.loss_history = []

        # 迭代优化
        for _ in range(self.n_iterations):
            # 每轮迭代打乱数据集，提升小批量梯度下降的稳定性
            shuffle_idx = np.random.permutation(n_samples)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]

            # 按batch_size分批次更新参数
            for i in range(0, n_samples, self.batch_size):
                # 截取当前批次数据
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                batch_size = X_batch.shape[0]

                # 批次预测值
                y_pred_batch = X_batch @ theta
                # 计算MSE损失的梯度
                gradient = (2 / batch_size) * (X_batch.T @ (y_pred_batch - y_batch))
                # 更新参数
                theta -= self.learning_rate * gradient

            # 记录每轮迭代的整体MSE损失，用于后续画图
            y_pred_full = X @ theta
            epoch_loss = np.mean((y_pred_full - y) ** 2)
            self.loss_history.append(epoch_loss)

        # 拆分权重与截距
        if self.fit_intercept:
            self.intercept = theta[0]
            self.weights = theta[1:]
        else:
            self.intercept = 0.0
            self.weights = theta

    def fit(self, X_train, y_train):
        """
        模型训练方法，根据初始化的method选择对应求解方式
        :param X_train: 训练集特征矩阵，shape=(样本数, 特征数)，每行对应一个样本的特征向量
        :param y_train: 训练集标签向量(房价真实值)，shape=(样本数,)
        :return: 模型自身实例，适配链式调用
        """
        # 数据格式转换
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        # 处理截距项
        if self.fit_intercept:
            X = self._add_intercept(X_train)
        else:
            X = X_train

        # 梯度下降模式下，校验batch_size合法性
        if self.method == 'gradient_descent' and self.batch_size > X.shape[0]:
            raise ValueError(f"batch_size({self.batch_size})不能超过训练样本总数({X.shape[0]})")

        # 选择对应求解方式
        if self.method == 'normal_equation':
            self._normal_equation_fit(X, y_train)
        else:
            self._gradient_descent_fit(X, y_train)

        return self

    def predict(self, X_test):
        """
        模型预测方法，输出测试集的预测值
        :param X_test: 测试集特征矩阵，shape=(测试样本数, 特征数)
        :return: 预测值数组，shape=(测试样本数,)
        """
        # 预检查：必须先训练再预测
        if self.weights is None:
            raise RuntimeError("请先调用 fit() 方法完成模型训练后再执行预测")

        X_test = np.array(X_test, dtype=np.float32)
        # 线性预测：y = X·w + b
        y_pred = X_test @ self.weights + self.intercept
        return y_pred

    def score(self, X_test, y_test):
        """
        模型评估方法，计算回归任务的R²决定系数
        :param X_test: 测试集特征矩阵
        :param y_test: 测试集真实标签(房价真实值)
        :return: R²分数，取值范围(-∞, 1]，越接近1表示模型拟合效果越好
        """
        y_pred = self.predict(X_test)
        y_true = np.array(y_test, dtype=np.float32)
        
        # 计算总平方和
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        # 计算残差平方和
        ss_residual = np.sum((y_true - y_pred) ** 2)
        # 计算R²
        r2_score = 1 - (ss_residual / ss_total) if ss_total != 0 else 0.0
        return r2_score