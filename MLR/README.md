## MLR(mixed logistic regression, 混合逻辑斯特回归)
MLR是2012年阿里妈妈提出的一种学习非线性模式的算法。
它可以看成是LR的一个自然推广，采用分而治之的思路用分片线性的模式来拟合高维空间的非线性分类面。
公式表达如下：
![mlr](http://img.mp.itc.cn/upload/20170616/efd4ccc98a7d43478342c3214f5cbb8f.jpg)
其中超参数m表示分类面数，用以平衡模型的拟合和推广能力，m=1时模型退化成LR。

参考博文: [这里](https://blog.csdn.net/xmtblog/article/details/77968459)
