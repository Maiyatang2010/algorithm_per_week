#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import tensorflow as tf

from model import CharRNN
from utils import build_dataset, gen_batch


def train(data_path, batch_size, rnn_size, lr, num_epochs, num_layers):
    """
    :param data_path: 训练文本路径
    :param batch_size: 批大小
    :param lr:         学习率
    :param num_epochs: epochs
    :param num_layers: 网络层数
    """

    poems_vector, word2idx, vocabs = build_dataset(data_path)
    x_batches, y_batches = gen_batch(batch_size, poems_vector, word2idx)

    configs = {
        "model": "lstm",
        "vocab_size": len(word2idx),
        "rnn_size": rnn_size,
        "num_layers": 2,
        "batch_size": batch_size,
        "lr": lr,
        "num_layers": num_layers
    }

    char_rnn = CharRNN(configs)
    char_rnn.build()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("start training ...")

        for epoch in range(num_epochs):
            n_batches = len(poems_vector) // batch_size
            for idx in range(n_batches):
                loss, _, _ = sess.run(
                    [char_rnn.mean_loss, char_rnn.train_op],
                    feed_dict={char_rnn.inputs: x_batches[idx], char_rnn.outputs: y_batches[idx]}
                )

                print("Epoch: %d, batch: %d, training mean loss: %.6f", epoch, idx, loss)

def main():
    data_path = "/home/mai/workspace/github/char_rnn/data/poems.txt"
    batch_size = 64
    rnn_size = 128
    lr = 0.01
    num_epochs = 50
    num_layers = 2

    train(data_path, batch_size, rnn_size, lr,  num_epochs, num_layers)


if __name__ == "__main__":
    main()
