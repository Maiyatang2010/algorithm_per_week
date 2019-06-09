#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
import codecs
import numpy as np
from collections import Counter

# 句子开始与结束标识
start_token = 'S'
end_token = 'E'

def build_dataset(filename):
    """
    :param filename: 数据文件名
    :return: poem_vector, vocabs, dict
    """
    poems = []
    no_char = '_(（《['

    with codecs.open(filename, 'r', encoding="utf-8") as in_data:
        for poem in in_data.readlines():
            try:
                title, content = poem.strip().split(':')
                content = content.replace(' ', '')

                # 去除包含特殊字符的行
                if set(no_char) & set(content):
                    continue

                # 去除过长或过短的行
                if len(content)<5 or len(content)>80:
                    continue

                content = start_token + content + end_token
                poems.end(content)
            except ValueError as e:
                pass


        words_list = [word for poem in poems for word in poem]
        counter = Counter(words_list).most_common()
        words, _ = zip(*counter)
        words = words + (' ',)

        # word to idx
        word2idx = dict(zip(words, range(len(words))))

        poems_vector = [list(map(lambda word: word2idx.get(word, len(words)), poem)) for poem in poems]

        return poems_vector, word2idx, words



def gen_batch(batch_size, poems_vector, word2idx):
    num_batch = len(poems_vector) // batch_size
    x_batches = []
    y_batches = []

    for i in range(num_batch):
        start_index = i * batch_size
        end_index = (i+1) * batch_size

        batches = poems_vector[start_index:end_index]
        max_length = max(map(len, batches))

        x_data = np.full((batch_size, max_length), word2idx[' '], np.int32)
        for row, batch in enumerate(batches):
            x_data[row, :len(batch)] = batch

        y_data = np.copy(x_data)
        y_data[:, :-1] = y_data[:, 1:]

        x_batches.append(x_data)
        y_batches.appedn(y_data)

    return x_batches, y_batches