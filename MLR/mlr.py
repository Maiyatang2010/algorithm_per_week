#!/usr/bin/env python
"""
阿里妈妈MLR(mix logistic regression)算法实现
运行方式:
    python mlr.py --m=12 --p=108 --lr=0.1 --epoch=10000
"""

import getopt
import pandas as pd
import tensorflow as tf
from random import random
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def get_data():
    columns = [
            'age','workclass','fnlwgt','education','education-num',
            'marital-status','occupation','relationship','race','sex',
            'capital-gain','capital-loss','hours-per-week','native-country','label'
            ]
    # 连续型变量
    continus_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    # 离散型变量
    dummy_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

    # 读取全部样本
    train_data = pd.read_csv("adult.data", names=columns, sep=',')
    train_data['label'] = train_data['label'].map(lambda x: 1 if x.strip() == '>50K' else 0)
    train_data["type"] = "train"

    test_data = pd.read_csv("adult.test", names=columns, sep=',')
    test_data['label'] = test_data['label'].map(lambda x: 1 if x.strip() == '>50K' else 0)
    test_data["type"] = "test"

    all_data = pd.concat([train_data, test_data], axis=0)
    all_data = pd.get_dummies(all_data, columns=dummy_columns)

    train_data = all_data[all_data["type"]=="train"].drop("type", axis=1)
    test_data = all_data[all_data["type"]=="test"].drop("type", axis=1)

    for col in continus_columns:
        ss = StandardScaler()
        train_data[col] = ss.fit_transform(train_data[[col]])
        test_data[col] = ss.transform(test_data[[col]])

    train_y = train_data["label"]
    train_x = train_data.drop("label", axis=1)

    test_y = test_data["label"]
    test_x = test_data.drop("label", axis=1)

    return train_x, train_y, test_x, test_y



class MLR(object):
    def __init__(self, config):
        self.m = config["m"];                # 聚类个数
        self.p = config["p"];                # 特征维数
        self.lr = config["lr"];              # 学习率
        self.n_epoch = config["epoch"]       # 迭代次数

    def inference(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.p])
        self.y = tf.placeholder(tf.int64, shape=[None])

        self.u = tf.Variable(tf.random_normal([self.p, self.m], 0.0, 0.1), name="u")
        self.w = tf.Variable(tf.random_normal([self.p, self.m], 0.0, 0.1), name="w")

        U = tf.matmul(self.x, self.U)
        W = tf.matmul(self.x, self.W)

        pi = tf.nn.softmax(U)
        eta = tf.nn.softmax(W)

        self.pred = tf.reduce_sum(tf.multiply(pi, eta), 1)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.y))
        self.train_op = tf.train.FtrlOptimizer(self.lr).minimize(self.loss)

    def train(self):
        self.inference()
        train_x, train_y, test_x, test_y = get_data()
        
        with tf.Session() as sess:
            sess.run(tf.global_vairables_initializer())
            for epoch in range(self.n_epoch):
                pred_, cost_, _ = sess.run([self.pred, self.cost, self.train_op], feed_dict={self.x: train_x, self.y: train_y})
                train_auc = roc_auc_score(train_y, pred_)
                
                if epoch % print_every == 0:
                    pred, cost = sess.run([self.pred, self.cost], feed_dict={self.x: test_x, self.y: test_y})
                    test_auc = roc_auc_score(test_y, pred)
                    
                    print("epoch: %d\ttrain cost: %.6f\ttrain auc: %.6f\ttest cost: %.6f\ttest auc:%.6f" % (epoch, cost_, train_auc, cost, test_auc))


if __name__ = "__main__":
    opt = {
        "m": None,
        "p": None,
        "lr": None,
        "epoch": None,
    }

    options, _ getopt.getopt(sys.argv[1:], '', ["m=", "p=", "lr=", "epoch="])

    for key, value = in options:
        if key[2:] == "lr":
            opt[key[2:]] = float(value)
        else:
            opt[key[2:]] = int(value) 

    mlr = MLR(opt)
    mlr.train()
