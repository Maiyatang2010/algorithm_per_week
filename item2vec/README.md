## 基于Word2Vec的协同过滤：item2vec

* 一种采用embedding思想，计算item相似度的方法，原理解释可参考[这里](https://zhuanlan.zhihu.com/p/27067810)
* 训练所用数据为公共数据集ml-20m，下载链接[这里](http://files.grouplens.org/datasets/movielens/ml-20m.zip)
* 由于word2vec训练时是以句子的形式进行训练，在推荐领域中这里以每个用户的点击历史作为一个句子(ml-20m中以>=4看作用户的点击)
* 另外，nlp中句子中词是有上下文关系的，即词与词之间是有序的。但推荐领域中认为item之间并没有固定的顺序关系，所以训练过程中设定的窗口大小为无穷大

