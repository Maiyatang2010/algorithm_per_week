#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
import tensorflow as tf

class CharRNN:
    def __init(self, configs):
        self.model = configs["model"]               # rnn单元类型
        self.vocab_size = configs["vocab_size"]     # 词汇大小
        self.rnn_size=  configs["rnn_size"]         # 状态维度
        self.num_layers = configs["num_layers"]     # 层数
        self.batch_size = configs["batch_size"]     # 批大小
        self.lr = configs["learning_ratge"]         # 学习率

    def build(self):
        self.inputs = tf.placeholder(tf.int32, [self.batch_size, None])
        self.outputs = tf.placeholder(tf.int32, [self.batch_size, None])

        if self.model == "rnn":
            cell_func = tf.contrib.rnn.BasicRNNCell
        elif self.model == "gru":
            cell_func = tf.contrib.rnn.GRUCell
        elif self.model == "lstm":
            cell_func = tf.contrib.rnn.BasicLSTMCell

        cell = cell_func(self.rnn_size, state_is_tuple=True)
        self.cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)
        initial_state = cell.zero_state(self.batch_size, tf.float32)

        self.embedding = tf.Variable(tf.random._uniform([self.vocab_size+1, self.rnn_size], -1.0, 1.0))
        input_data = tf.nn.embedding_lookup(self.embedding, self.inputs)

        output_data, last_state = tf.nn.dynamic_rnn(self.cell, input_data, initial_state=initial_state)
        output_data = tf.reshape(output_data, [-1, self.rnn_size])

        # logits
        weights = tf.Variable(tf.truncated_normal([self.rnn_size, self.vocab_size+1], mean=0.0, stddev=1.0))
        bias = tf.Variable(tf.zeros(shape=[self.vocab_size+1]))
        self.logits = tf.nn.bias_add(tf.matmul(output_data, weights), bias=bias)

        labels = tf.one_hot(tf.reshape(self.outputs, [-1]), depth=self.vocab_size+1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.logits)
        self.mean_loss = tf.reduce_mean(loss)
        self.prediction = tf.nn.softmax(self.logits)

        # train_op
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.mean_loss)









