# 不讲5德的attention到底是啥?

attention由来已久，让它名声大噪的还是BERT，可以说NLP中，BERT之后，再无RNN和CNN。那么attention到底有哪些呢？hard attention、soft attention、global attention、local attention、self-attention, 啊，这些都是啥


# 起源

[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
文章是一个图像的描述生成，encoder利用CNN从图像中提取信息为L个D维的矩阵，作为decoder的输入。即:
$$
a_{i}, i=1,2, \ldots, L
$$


而decoder使用了lstm，认为L个D维向量可以构成一个序列（并非时间序列，而且空间上的序列），这里LSTM中输出为上一步的隐变量$h_{t-1}$, 上一步的输出$y_{t-1}$和$Z_t$, 隐变量和输出是LSMT已有的，那么$Z_t$是什么东西，怎么获得的呢？

这里作者提出了attention机制，机器自动学习获得attention权重后对向量加权求和，获得$Z_t$，很抽象，咱们直接上公式：
$$
e_{ti} = f_{att}(a_i, h_{i-1}) = act\_fuc( W^a \times a_i + W^h \times h_{i-1} + b ) \\
\alpha_{t i}=\frac{\exp \left(e_{t i}\right)}{\sum_{k=1}^{L} \exp \left(e_{t k}\right)}
$$
这里的权重获得使用了感知机结构，act_fuc是激活函数，可以是sigmoid、relu等，那么$Z_t$计算为attention权重的加权求和：
$$
Z_t = \sum_{i=1}^L \alpha_{ti} * a_i
$$

本文还提出来hard/soft attention, 那么什么是hard attention呢？

对于t时刻，其对于位置i的attention的权重记为$S_{ti}$，作者认为$S_{ti}$应该服从多元伯努利分布，即所有$s_{ti}, i=1,2,...L$中只有个为1，是一个one-hot向量，即：
$$
\begin{array}{l}p\left(s_{t, i}=1 \mid s_{j<t}, \mathbf{a}\right)=\alpha_{t, i} \\ \hat{\mathbf{z}}_{t}=\sum_{i} s_{t, i} \mathbf{a}_{i}\end{array}
$$

可以看到$\alpha_{t,i}$没有直接参与$\hat z_t$的计算，同时似乎没法求导，那怎么办呢？对，随机采样。





相对而言soft attention就容易理解，相比于one-hot, sotf即全部位置都会有加入，区别在于权重的大小，此时：
$$
Z_t = \beta_t * \sum_{i=1}^L \alpha_{ti} * a_i \\ \beta_{t}=\sigma\left(f_{\beta}\left(h_{t-1}\right)\right)
$$
其中

同时，模型损失中加入了$\alpha_{ti}$的正则项，这是为什么呢？
$$
L_{d}=-\log (P(\mathbf{y} \mid \mathbf{x}))+\lambda \sum_{i}^{L}\left(1-\sum_{t}^{C} \alpha_{t i}\right)^{2}
$$
首先attention权重进过sotfmax是保证$\sum_i^L \alpha_{ti} = 1$，同时此项损失中的正则保证$\sum_t^C \alpha_{ti} \approx 1$，其中$\sum_t^C \alpha_{ti}$表示同一个t的所有被关注(attention)的权重和，所以要求每个位置被重视的总和相等。

> 这个“被”字有点绕，举例i=1时候$\alpha_{11}$表示$a_1$对于$a_1$的权重，$\alpha_{21}$表示$a_1$对于$a_2$的权重，以此类推，$\alpha_{L1}$表示$a_1$对于$a_L$的权重，$\sum_t^C \alpha_{ti}$=1，即要求$a_1, a_2,...a_L$在被attention的总和相等，保证每个部位被同样关注。

# Scaled Dot-Product Attention

> Transformer中attention御用方式。

不讲5德，直接上公式，
$$
\text {Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V
$$
其中，self attention中，Q,K,V来源于同一个输入X：
$$
Q=X \times W^q  \\
K=X \times W^k  \\
V=X \times W^v
$$


# Reference

- https://www.zhihu.com/question/68482809/answer/1574319286
- https://zhuanlan.zhihu.com/p/47282410
- https://www.jiqizhixin.com/articles/2018-06-11-16

