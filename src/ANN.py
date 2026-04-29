import numpy as np
import matplotlib.pyplot as plt

class ANN:
    """
    人工神经网络实现，支持回归和分类任务
    接口与项目中其他算法保持一致：fit(X, y) / score(X, y)
    """
    def __init__(self, hidden_layers=(64, 32), learning_rate=0.01, n_iterations=1000, 
                 activation='relu', task_type='auto', random_state=42, dropout_rate=0.0):
        """
        初始化神经网络
        :param hidden_layers: 隐藏层结构，元组形式，如(64,32)表示两个隐藏层，分别有64和32个神经元
        :param learning_rate: 学习率
        :param n_iterations: 训练迭代次数
        :param activation: 隐藏层激活函数：relu/sigmoid/tanh
        :param task_type: 任务类型：auto(自动识别)/regression(回归)/classification(分类)
        :param random_state: 随机种子
        """
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.activation = activation
        self.task_type = task_type
        self.random_state = random_state
        self.loss_history = []  # 记录训练损失，用于绘制曲线
        self.weights = []
        self.biases = []
        self.dropout_rate = dropout_rate
        self.training = True  # 新增：标记是否为训练模式
        
    def _activation(self, z):
        """隐藏层激活函数"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError(f"不支持的激活函数：{self.activation}")
    
    def _activation_derivative(self, a):
        """激活函数导数（输入为激活后的值a）"""
        if self.activation == 'relu':
            return np.where(a > 0, 1, 0)
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        elif self.activation == 'tanh':
            return 1 - np.square(a)
        else:
            raise ValueError(f"不支持的激活函数：{self.activation}")
    
    def _output_activation(self, z):
        """输出层激活函数"""
        if self.task_type == 'regression':
            return z  # 回归任务：线性输出
        elif self.task_type == 'classification':
            if self.n_classes == 2:
                return 1 / (1 + np.exp(-z))  # 二分类：sigmoid
            else:
                # 多分类：softmax（防止数值溢出）
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _loss(self, y_true, y_pred):
        """计算损失"""
        m = y_true.shape[0]
        if self.task_type == 'regression':
            # 回归：均方误差MSE
            return np.mean(np.square(y_true - y_pred))
        elif self.task_type == 'classification':
            if self.n_classes == 2:
                # 二分类：二元交叉熵
                epsilon = 1e-15
                y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            else:
                # 多分类：交叉熵
                epsilon = 1e-15
                y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    def _initialize_weights(self, input_dim, output_dim):
        """初始化权重和偏置"""
        np.random.seed(self.random_state)
        layers = [input_dim] + list(self.hidden_layers) + [output_dim]
        
        for i in range(len(layers) - 1):
            # He初始化（适合ReLU）
            std = np.sqrt(2 / layers[i])
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * std)
            self.biases.append(np.zeros((1, layers[i+1])))
    
    def _forward_propagation(self, X):
        """前向传播"""
        activations = [X]
        zs = []
        
        # 隐藏层
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self._activation(z)
            # 新增：训练时应用Dropout
            if self.training and self.dropout_rate > 0:
                mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                a *= mask
            activations.append(a)
        
        # 输出层
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self._output_activation(z)
        activations.append(a)
        
        return activations, zs
    
    def _backward_propagation(self, X, y, activations, zs):
        """反向传播"""
        m = X.shape[0]
        delta_weights = [np.zeros_like(w) for w in self.weights]
        delta_biases = [np.zeros_like(b) for b in self.biases]
        
        # 输出层误差
        if self.task_type == 'regression':
            delta = activations[-1] - y.reshape(-1, 1)
        elif self.task_type == 'classification':
            if self.n_classes == 2:
                delta = activations[-1] - y.reshape(-1, 1)
            else:
                delta = activations[-1] - y
        
        delta_weights[-1] = np.dot(activations[-2].T, delta) / m
        delta_biases[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        for l in range(2, len(self.weights) + 1):
            delta = np.dot(delta, self.weights[-l+1].T) * self._activation_derivative(activations[-l])
            delta_weights[-l] = np.dot(activations[-l-1].T, delta) / m
            delta_biases[-l] = np.sum(delta, axis=0, keepdims=True) / m
        
        # 更新权重和偏置
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * delta_weights[i]
            self.biases[i] -= self.learning_rate * delta_biases[i]
    
    def fit(self, X, y):
        """训练模型（新增目标值标准化，解决回归任务损失爆炸问题）"""
        # 自动识别任务类型
        if self.task_type == 'auto':
            if len(np.unique(y)) <= 10 and np.all(y == y.astype(int)):
                self.task_type = 'classification'
                self.n_classes = len(np.unique(y))
            else:
                self.task_type = 'regression'
                self.n_classes = 1

        # 处理分类任务的标签
        if self.task_type == 'classification':
            self.classes_ = np.unique(y)
            if self.n_classes > 2:
                # 多分类：转换为one-hot编码
                y_one_hot = np.zeros((y.shape[0], self.n_classes))
                for i, cls in enumerate(self.classes_):
                    y_one_hot[y == cls, i] = 1
                y = y_one_hot
        # 新增：回归任务目标值标准化
        else:
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
            y = (y - self.y_mean) / self.y_std

        # 初始化权重
        input_dim = X.shape[1]
        output_dim = 1 if (self.task_type == 'regression' or self.n_classes == 2) else self.n_classes
        self._initialize_weights(input_dim, output_dim)

        # 训练循环
        self.loss_history = []
        for i in range(self.n_iterations):
            activations, zs = self._forward_propagation(X)
            loss = self._loss(y, activations[-1])
            self.loss_history.append(loss)
            self._backward_propagation(X, y, activations, zs)
        
            # 每100次迭代打印一次进度
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.n_iterations}, 损失: {loss:.4f}")
    
        return self

    def predict(self, X):
        """预测（新增回归任务反标准化）"""
        activations, _ = self._forward_propagation(X)
        y_pred = activations[-1]
    
        if self.task_type == 'regression':
            # 反标准化，恢复原始房价尺度
            y_pred = y_pred.ravel() * self.y_std + self.y_mean
            return y_pred
        elif self.task_type == 'classification':
            if self.n_classes == 2:
                return (y_pred.ravel() > 0.5).astype(int)
            else:
                return self.classes_[np.argmax(y_pred, axis=1)]
    
    def score(self, X, y):
        """
        模型评估
        回归任务：返回R²决定系数
        分类任务：返回准确率
        """
        y_pred = self.predict(X)
        
        if self.task_type == 'regression':
            # 计算R²
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            return 1 - (ss_residual / ss_total)
        elif self.task_type == 'classification':
            # 计算准确率
            return np.mean(y_pred == y)