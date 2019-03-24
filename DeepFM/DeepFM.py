# -*- coding:utf-8 -*-
"""
实现推荐系统CTR预估模型之一DeepFM
参考：https://www.jianshu.com/p/6f1c2643d31b中对DeepFM的说明
Date:2019-03-24
"""

import numpy as np
import tensorflow as tf

class DeepFM(object):
    def __init__(self, config):
        assert(config.use_fm or config.use_deep)
        assert config.loss_type in ['logloss', 'mse'], 'loss_type must be either "logloss" or "mse"

        self.feature_size = config.feature_size     # one-hot后的特征长度
        self.field_size = config.field_size         # 特征的field数
        self.embedding_size = config.embedding_size # 隐向量大小

        self.dropout_fm = config.dropout_fm         # dropout for FM
        self.dropout_deep = config.dropout_deep     # dropout for deep
        self.use_fm = config.use_fm                 # 是否使用FM
        self.use_deep = config.use_deep             # 是否使用deep

        self.deep_layers = config.deep_layers   # deep的隐藏层大小
        self.deep_layers_activation = config.deep_layers_activation # 隐藏层激活函数
        
        self.n_epoch = config.n_epoch               # 训练的epoch数目
        self.batch_size = config.batch_size         # 训练用的batch大小
        self.learning_rate = config.learning_rate   # 训练的学习率
        self.loss_type = config.loss_type           # 损失函数类型

    def add_placeholder(self):
        """
        添加用于输入placeholder
        """
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')

        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_fm')
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name='dropout_keep_deep')

    def add_variables(self):
        """
        添加用于训练的模型参数
        """
        weights = {}
        
        # embedding
        weights['feature_embeddings'] = tf.Variable(tf.truncated_normal(shape=[self.feature_size, self.embedding_size], mean=.0, stddev=.01), name='feature_embeddings')
        weights['feature_bias'] = tf.Variable(tf.truncated_normal(shape=[self.feature_size, 1], mean=.0, stddev=.01), name='feature_bias')

        # deep
        num_layers = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))

        weights['weight_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=tf.float32, name='weight_0')
        weights['bias_0'] = tf.Variable(tf.zeros(shape=[1, self.deep_layers[0]], dtype=tf.float32), name='bias_0')

        for i in range(1, num_layers):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights['weight_%d' % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])), dtype=tf.float32, name='weight_%d' % i)
            weights['bias_%d' % i] = tf.Variable(tf.zeros(shape=[1, self.deep_layers[i]], dtype=tf.float32), name='bias_%d' % i)

        # concat
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.field_size + self.embedding_size 
        else:
            input_size = self.deep_layers[-1]

        glorot = np.sqrt(2.0 / (input_size + 1))
        weights['weight_concat'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)), dtype=tf.float32, name='weight_concat')
        weights['bias_concat'] = tf.Variable(tf.zeros(shape=[1,], dtype=tf.float32), name='bias_concat')

        return weights


    def inference(self):
        """
        构造前向网络
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.add_placeholders()
            self.add_variables()

            self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.feat_index)
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings, feat_value)
           
            # first order
            self.y_first_order = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
            self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)
            self.y_first_order = tf.nn.dropout(self.y_first_order, keep_prob=self.dropout_keep_fm[0])

            # second order
            summed_features_emb = tf.reduce_sum(self.embeddings,1)
            summed_features_emb_square = tf.square(self.summed_features_emb)

            squared_features_emb = tf.square(self.embeddings)
            squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)

            self.y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)
            self.y_second_order = tf.nn.dropout(y_second_order,self.dropout_keep_fm[1])

            # deep
            self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
            self.y_deep = tf.nn.dropout(self.y_deep,self.dropout_keep_deep[0])
            for i in range(0,len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep,self.weights['weight_%d' % i]), self.weights['bias_%d' % i])
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, keep_prob=self.dropout_keep_deep[i+1])

            # concat
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            self:
                concat_input = self.y_deep

            self.y_out = tf.add(tf.matmul(concat_input,self.weights['weight_concat']),self.weights['bias_concat'])

            # loss
            if self.loss_type == 'logloss':
                self.y_out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.y_out)
            else:
                self.loss = tf.nn.l2_loss(tf.subtract(tf.label, tf.y_out))

            # optimizer
            self.train_op = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
