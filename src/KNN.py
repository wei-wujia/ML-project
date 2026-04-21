# 标准KNN算法实现
# 支持二分类、多分类任务，输入为特征向量形式，接口与scikit-learn对齐，适配工程化调用
import numpy as np

class KNN:
    """
    K近邻(K-Nearest Neighbors)标准分类算法
    适配项目任务：titanic二分类、mnist/cifar10 10分类
    """
    def __init__(self, n_neighbors=5, metric='euclidean'):
        """
        初始化KNN分类器
        :param n_neighbors: 近邻数K，默认值5，可自定义调参
        :param metric: 距离度量方式，可选 'euclidean'(欧氏距离，默认) / 'manhattan'(曼哈顿距离)
        """
        if n_neighbors < 1:
            raise ValueError("近邻数 n_neighbors 必须大于等于1")
        self.n_neighbors = n_neighbors
        self.metric = metric
        # 训练集数据（KNN无显式训练过程，仅存储训练数据用于预测阶段计算）
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        模型训练方法（KNN仅存储训练集，无参数迭代训练过程）
        :param X_train: 训练集特征矩阵，shape=(样本数, 特征数)，每行对应一个样本的特征向量
        :param y_train: 训练集标签向量，shape=(样本数,)
        :return: 模型自身实例，适配链式调用
        """
        # 转换为numpy浮点数组，保证数值计算兼容性
        self.X_train = np.array(X_train, dtype=np.float32)
        self.y_train = np.array(y_train)

        # 校验K值合法性
        if self.n_neighbors > len(self.X_train):
            raise ValueError(f"近邻数{self.n_neighbors}不能超过训练样本总数{len(self.X_train)}")
        return self

    def _calc_distance(self, x):
        """
        内部方法：计算单个测试样本与所有训练样本的距离
        :param x: 单个测试样本的特征向量，shape=(特征数,)
        :return: 距离数组，shape=(训练样本数,)
        """
        if self.metric == 'euclidean':
            # 欧氏距离：sqrt(Σ(x1-x2)²)，向量化计算，效率远高于循环
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            # 曼哈顿距离：Σ|x1-x2|
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"不支持的距离度量{self.metric}，仅支持'euclidean'和'manhattan'")
        return distances

    def predict(self, X_test):
        """
        模型预测方法，输出测试集的预测标签
        :param X_test: 测试集特征矩阵，shape=(测试样本数, 特征数)，每行对应一个样本的特征向量
        :return: 预测标签数组，shape=(测试样本数,)
        """
        # 预检查：必须先训练再预测
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("请先调用 fit() 方法完成模型训练后再执行预测")

        X_test = np.array(X_test, dtype=np.float32)
        y_predict = []

        # 遍历每个测试样本，完成KNN预测逻辑
        for x in X_test:
            # 1. 计算当前样本与所有训练样本的距离
            distances = self._calc_distance(x)
            # 2. 排序获取距离最小的前K个样本的索引
            nearest_idx = np.argsort(distances)[:self.n_neighbors]
            # 3. 获取K个近邻的标签
            nearest_labels = self.y_train[nearest_idx]
            # 4. 投票机制：取出现次数最多的标签作为预测结果（天然支持多分类）
            pred_label = np.bincount(nearest_labels).argmax()
            y_predict.append(pred_label)

        return np.array(y_predict)

    def score(self, X_test, y_test):
        """
        模型评估方法，计算测试集分类准确率
        :param X_test: 测试集特征矩阵
        :param y_test: 测试集真实标签
        :return: 准确率 accuracy，取值范围 [0, 1]
        """
        y_pred = self.predict(X_test)
        y_true = np.array(y_test)
        # 计算预测正确的样本占比
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        return accuracy