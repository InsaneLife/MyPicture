# 不讲5德的attention到底是啥?

attention由来已久，让它名声大噪的还是BERT，可以说NLP中，BERT之后，再无RNN和CNN。那么attention到底有哪些呢？hard attention、soft attention、global attention、local attention、self-attention, 啊，这些都是啥


# 起源

[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
文章是一个图像的描述生成，encoder利用CNN从图像中提取信息为L个D维的矩阵，作为decoder的输入。即


而decoder使用了lstm，认为L个D维向量可以构成一个序列（并非时间序列，而且空间上的序列），这里LSTM中每一步就有上一步的输出向量$Z_t$, 上一步的隐变量$h_{t-1}$, 上一步的输出$y_{t-1}$, 隐变量和输出是可知的，那么$z_t$
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