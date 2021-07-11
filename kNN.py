import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    def __init__(self,k):
        assert k >= 1,'k must be valid'
        self._k = k
        self._x_train = None
        self._y_train = None

    def fit(self,x_train,y_train):
        """
        根据训练数据x_trian和y_train训练knn模型
        :param x_train: 训练数据x
        :param y_train: 训练数据标签y
        :return:
        """
        self._x_train = x_train
        self._y_train = y_train
        return self
    def predict(self,x_predict):
        """对待预测数据集进行预测"""
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    def _predict(self,x):
        """计算x与_x_train的距离，然后排序通过投票的方式取最大值"""
        distances  = [sqrt(np.sum((_x-x)**2)) for _x in self._x_train]
        nearest = np.argsort(distances)
        topK_y = [self._y_train[i] for i in nearest[:self._k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]


    def __repr__(self):
        return "KNN(k={})".format(self._k)
