# -*- coding-utf-8 -*-
import os
import random
random.seed(17)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
assert gensim.models.word2vec.FAST_VERSION > -1

from gensim.models import Word2Vec
import datetime
start = datetime.datetime.now()

def train(dir_path):
    """
    使用ml-20m数据训练item2vec，其中包含:
    1. movies:name->id
    2. ratings: uid, movie_id, rating, timestamp
    dir_path: 样本文件夹路径
    """
    print "reading traing data ..."
    df_movies = pd.read_csv(os.path.join(dir_path, 'movies.csv'))
    df_ratings = pd.read_csv(os.path.join(dir_path, 'ratings.csv'))

    # 构建movieId->name及name->movieId的映射
    movieId_to_name = pd.Series(df_movies.title.values, index = df_movies.movieId.values).to_dict()
    name_to_movieId = pd.Series(df_movies.movieId.values, index = df_movies.title).to_dict()

    # 划分训练测试集
    print "split train and test ..."
    df_ratings_train, df_ratings_test= train_test_split(
        df_ratings,
        stratify=df_ratings['userId'],
        random_state = 15688,
        test_size=0.30
    )

    print("Number of training data: "+str(len(df_ratings_train)))
    print("Number of test data: "+str(len(df_ratings_test)))

    # 对训练样本构造标签, 这里以4为阈值, >=4则说明用户喜欢否则用户不喜欢
    df_ratings_train['liked'] = np.where(df_ratings_train['rating']>=4, 1, 0)
    df_ratings_train['movieId'] = df_ratings_train['movieId'].astype('str')
    gp_user_like = df_ratings_train.groupby(['liked', 'userId'])

    # 以用户、标签维度将日志划分为不同句子
    splitted_movies = [gp_user_like.get_group(gp)['movieId'].tolist() for gp in gp_user_like.groups]
    for movie_list in splitted_movies:
        random.shuffle(movie_list)

    # word2vec模型训练
    print "training item2vec using word2vec ..."
    start = datetime.datetime.now()
    model = Word2Vec(
        sentences = splitted_movies,
        iter = 5,
        min_count = 10,
        size = 200,
        workers = 4,
        sg = 1,
        hs = 0,
        negative = 5,
        window = 9999999
    )
    print("trainng time passed: " + str(datetime.datetime.now()-start))
    model.save("item2vec.model")

def recommender(model, positive_list=None, negative_list=None, topn=20):
    recommend_movie_ls = []
    for movieId, prob in model.wv.most_similar_cosmul(positive=positive_list, negative=negative_list, topn=topn):
        recommend_movie_ls.append(movieId)
    return recommend_movie_ls

if __name__ == "__main__":
    print "model training phase:"
    train("/home/xu/workspace/data/ml-20m")

    print "recommendation samples:"
    recommender(["50872"])