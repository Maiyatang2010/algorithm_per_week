# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf

class FFM(object):
    def __init__(self, config):
        """
        initial params
        """
        self.k = config['k']
        self.f = config['f']
        self.n = config['n']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.feature2field = config['feature2field']

    def initial_variables(self):
        # placeholders
        self.X = tf.placeholer(tf.float32, [None, self.n])
        self.y = tf.placeholder(tf.int64, [None,])
        self.keep_prob = tf.placeholder(tf.float32)

        # variables
        self.b = tf.Variable(initial_value=tf.zeros([2], dtype=tf.float32), name='b')
        self.w1 = tf.Variable(initial_value=tf.truncated_normal(shape=[self.n, 2], mean=0.0, stddev=0.02), name='w1')
        self.v = tf.Variable(initial_value=tf.trucncated_normal(shape=[self.n, self.f, self.k], mean=0.0, stddev=0.02), name='v')

    def inference(self):
        linear_term = tf.add(tf.matmul(self.X, self.w1), self.b)
        interaction_term = tf.constant(0, dtype=tf.float32)
        for i in range(self.n):
            for j in range(i+1, self.n):
                interaction_term += tf.multiply(
                        tf.reduce_sum(tf.multiply(v[i, self.feature2field[i]], v[j, self.feature2field[j]])),
                        tf.multiply(self.X[:, i], self.X[:, j])
                        )

        y_pred = tf.nn.softmax(tf.add(linear_term, interaction_term))
        logits = tf.nn.sparse_softmax_cross_entropy_with_logits(lables=self.y, logits=y_pred)
        self.loss = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdagradDAOptimizer(self.lr)
        self.train_op = optimizer.minimize(loss)

    def train(self, n_epoch=10, data_path):
        """
        data_path: 数据路径
        n_epoch: 迭代次数
        """
        # build graph
        self.inference()

        # data training
        train_data = pd.read_csv(data_path, header=0, sep='\t')
        n_sample = train_data.shape[0]
        train_step = 0
        with tf.Session as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(n_epoch):
                idxs = np.random.permutation(n_sample)
                n_batch = int(len(idxs)/self.batch_size)
                for i in range(n_batch):
                    train_step += 1

                    batch_X = []
                    batch_y = []
                    for j in range(self.batch_size):
                        batch_X.append(train_data.iloc[idxs[i*self.batch_size+j], :-2].values)
                        batch_y.append(train_data.iloc[idxs[i*self.batch_size+j], -1])

                    batch_X = np.asarray(batch_X)
                    batch_y = np.asarray(batch_y)

                feed_dict = {self.X:batch_X, self.y:batch_y, self.keep_prob:1}
                loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
                
                if train_step % 200 == 0:
                    print("number of epochs: %d, train_step: %d, loss=%.4f" % (epoch, train_step, loss))
