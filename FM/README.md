## FM(factor machine), 因子分解机
#### 1. 目录结构
```
.
├── README.md
├── fm.py
├── test_data.txt
└── train_data.txt

```

#### 2. 使用说明
```
# fm.py FM实现类
# test_data.txt 测试集(268, 9)
# train_data.txt 训练集(500, 9)
# 执行方式: python fm.py 
```

#### 3.参考博客
1. 介绍fm在工业界的使用(文风好，非常透彻): https://zhuanlan.zhihu.com/p/58160982
2. 具体实现参考(很赞，推导过程非常清晰): https://www.cnblogs.com/wkang/p/9588360.html

#### 4. 关于交叉项的实现理解

![avata](https://images2018.cnblogs.com/blog/1473228/201809/1473228-20180907110105340-1586099344.jpg)
$$
\frac{1}{2}\sum_{1}^{k}\Bigg(\bigg(\sum_{i=1}^{n}(v_{il}x_i)\bigg)^2-\sum_{i=1}^{n}v_{il}^2x_i^2\Bigg)
$$
转为矩阵描述时，假设shape(x)=(1, n), shape(V)=(n, k)，则等价如下：
$$
\frac{1}{2}sum(\bold{x}\bold{V}\odot(\bold{x}\bold{V}) - (\bold{x}\odot\bold{x})(\bold{V}\odot\bold{V}))
$$
