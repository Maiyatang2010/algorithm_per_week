# -*- coding:utf-8 -*-
"""
实现推荐系统召回模型之全能的FM模型
fm blog: https://zhuanlan.zhihu.com/p/58160982
Date:2019-03-10 
"""

import numpy as np
from random import normalvariate
from sklearn import preprocessing

class FM(object):
    def __init__(self, data_path, which_label, delimiter):
        """
        :param data_path: 数据路径
        :param which_label: 标签列
        :param delimiter: 列分隔符
        """

        self.data_path = data_path
        self.which_label = which_label 
        self.delimiter = delimiter

        self._w_0 = None    # 偏置项, shape=(1,)
        self._w = None      # 一阶系数, shape=(n, 1)
        self._v = None       # 特征隐向量矩阵, shape=(n, k)
        

    def normlize(self, features):
        """
        对features的每列进行归一化
        """
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(features)


    def sigmoid(self, value):
        return 1.0 / (1 + np.exp(-value))


    def load_data(self, data_path, which_label, delimiter):
        """
        :param data_path: 数据路径
        :param which_label: 标签列
        :param delimiter: 列分隔符
        :return: feature_mat, labels
        """
        
        data = np.loadtxt(data_path, delimiter=delimiter)
        labels = data[:, which_label]
        features = np.hstack([data[:, 0:which_label], data[:, which_label+1:]])

        return np.mat(self.normlize(features)), labels


    def train(self, k=8, alpha=0.01, iter=100):
        """
        :param data: 训练数据
        :k: 特征隐向量大小
        :alpha: 学习率
        :iter: 迭代次数
        """
        features, labels = self.load_data(self.data_path, self.which_label, self.delimiter)
        m, n = features.shape
        print("features's shape: (%d, %d)" % (m, n))
        print("labels's shape: %d" % len(labels))

        w_0 = 0.
        w = np.zeros((n, 1))
        v = normalvariate(0, 0.2) * np.ones((n, k))

        for it in range(iter):
            for idx in range(m):
                inter_1 = np.dot(features[idx], v)
                inter_2 = np.dot(np.multiply(features[idx], features[idx]), np.multiply(v, v))

                interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.

                # compute y_hat
                y_hat = w_0 + features[idx].dot(w) + interaction
                loss = self.sigmoid(labels[idx] * y_hat[0, 0]) - 1
                if loss >= -1:
                    loss_rec = 'pos'
                else:
                    loss_rec = 'neg'

                # update parameters
                w_0 = w_0 - alpha * loss * labels[idx]
                for i in range(n):
                    if features[idx, i] != 0:
                        w[i, 0] = w[i, 0] - alpha * loss * labels[idx] * features[idx, i]
                        for j in range(k):
                            v[i, j] = v[i, j] - alpha * loss * labels[idx] * (features[idx, i] * inter_1[0, j] - v[i, j] * features[idx, i]**2)
            print("finish the %dth iter" % it)

        self._w_0 = w_0
        self._w = w
        self._v = v

    
    def predict(self, X):
        """
        对给定的测试样本X:(m, n)输出预测概率
        """
        if (self._w_0 is None) or (self._w is None) or (self._v is None):
            print("Estimator not fitted, call `fit` first")
            return

        m, n = X.shape
        result = []
        for idx in range(m):
            inter_1 = X[idx].dot(self._v)
            inter_2 = np.multiply(X[idx], X[idx]).dot(np.multiply(self._v, self._v))
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.

            y_hat = self.sigmoid(self._w_0 + X[idx].dot(self._w) + interaction)
            result.append(y_hat)

        return result

    def test(self, data_path, which_label, delimiter):
        """
        测试例子
        """
        features, labels = self.load_data(data_path, which_label, delimiter)
        m, n = features.shape
        result = self.predict(features)
        
        error_cnt = 0.
        for idx in range(m):
            if result[idx] < 0.5 and labels[idx] == 1.0:
                error_cnt += 1
            elif result[idx] >= 0.5 and labels[idx] == -1.0:
                error_cnt += 1
            else:
                continue

        return 1 - error_cnt / m

if __name__ == "__main__":
    train_path = "./train_data.txt"
    test_path = "./test_data.txt"

    fm = FM(train_path, 8, " ")
    fm.train()
    print("test accuracy: %.4f" % fm.test(test_path, 8, " "))
