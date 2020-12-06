# 不讲5德的attention到底是啥?

attention由来已久，让它名声大噪的还是BERT，可以说NLP中，BERT之后，再无RNN和CNN。那么attention到底有哪些呢？



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