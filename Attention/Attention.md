# 不讲5德的attention到底是啥?

attention由来已久，让它名声大噪的还是BERT，可以说NLP中，BERT之后，再无RNN和CNN。那么attention到底有哪些呢？hard attention、soft attention、global attention、local attention、self-attention, 啊，这些都是啥？相似度计算的dot、general、concat都是怎么计算的？


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

本文还提出来hard/soft attention.

## 那么什么是hard attention呢？

对于t时刻，其对于位置i的attention的权重记为$S_{ti}$，作者认为$S_{ti}$应该服从多元伯努利分布，即所有$s_{ti}, i=1,2,...L$中只有个为1，是一个one-hot向量，即：
$$
\begin{array}{l}p\left(s_{t, i}=1 \mid s_{j<t}, \mathbf{a}\right)=\alpha_{t, i} \\ \hat{\mathbf{z}}_{t}=\sum_{i} s_{t, i} \mathbf{a}_{i}\end{array}
$$

可以看到$\alpha_{t,i}$没有直接参与$\hat z_t$的计算，损失函数当然是在条件a的情况下最大化正确y的概率，即$\log p(y|a)$，作者通过Jensen 不等式将目标函数定义为$\log p(y|a)$的一个下界，巧妙的将s加入到损失函数中：
$$
\begin{aligned} L_{s} &=\sum_{s} p(s \mid \mathbf{a}) \log p(\mathbf{y} \mid s, \mathbf{a}) \\ & \leq \log \sum_{s} p(s \mid \mathbf{a}) p(\mathbf{y} \mid s, \mathbf{a}) \\ &=\log p(\mathbf{y} \mid \mathbf{a}) \end{aligned}
$$
计算梯度：
$$
\begin{aligned} \frac{\partial L_{s}}{\partial W}=\sum_{s} p(s \mid \mathbf{a}) &\left[\frac{\partial \log p(\mathbf{y} \mid s, \mathbf{a})}{\partial W}+\right.\left.\log p(\mathbf{y} \mid s, \mathbf{a}) \frac{\partial \log p(s \mid \mathbf{a})}{\partial W}\right] \end{aligned}
$$


损失函数似乎没法求导，那怎么办呢？对，随机采样。利用蒙特卡洛方法对 s 进行抽样，我们做 N 次这样的抽样实验，记每次取到的序列是是$\tilde{s}^{n}$，其概率就是1/N，那么梯度结果：
$$
\begin{aligned} \frac{\partial L_{s}}{\partial W} \approx \frac{1}{N} \sum_{n=1}^{N}\left[\frac{\partial \log p\left(\mathbf{y} \mid \tilde{s}^{n}, \mathbf{a}\right)}{\partial W}+\right. \left.\log p\left(\mathbf{y} \mid \tilde{s}^{n}, \mathbf{a}\right) \frac{\partial \log p\left(\tilde{s}^{n} \mid \mathbf{a}\right)}{\partial W}\right] \end{aligned}
$$


## soft attention

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

那么什么又是global attention 和 local attention呢？
# global attention 和 local attention

是否是说有些部分的attention并不用关注于全局的信息，只需要关注部分的信息就好了， 那么是否可以有attention只关注一部分位置上的输出呢？

> [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

[Effective](https://arxiv.org/abs/1508.04025)提出了global attention 和 local attention概念，具体可以看图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228003656741.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoaW5lMTk5MzA4MjA=,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20201228003705391.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NoaW5lMTk5MzA4MjA=,size_16,color_FFFFFF,t_70)

> 图中左边为全局attention，右边为local。蓝色块表示输入序列，红色块表示生成序列，可以看到，global在生成$c_t$时候回考虑全局的输入，和正常attention无异。
>
> local attention会有一个窗口，在窗口中的输入才会被计算权重，可以认为其余都是0。这让我想到了卷积🤣
>
> 最终的会将二者的context向量和$h_t$ concat作为最终的输出。

**global attention:** 对于global attention，其输入序列$\bar{h}_{s}, s=1,2, \ldots, n$, 对于输出序列$h_t$，和每个$\bar{h}_{s}$计算attention权重然后加权求和获得context向量, attention权重计算方式为：
$$
\alpha_t(s)=\frac{\exp \left(\operatorname{score}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s}\right)\right)}{\sum_{s^{\prime}} \exp \left(\operatorname{score}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s^{\prime}}\right)\right)} \tag{7}
$$
那么其中的score是怎么计算的呢，作者总结了一下历史的attention的权重3种计算方式：
$$
\operatorname{score}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s}\right)=\left\{\begin{array}{ll}\boldsymbol{h}_{t}^{\top} \overline{\boldsymbol{h}}_{s} & \text { dot } \\ \boldsymbol{h}_{t}^{\top} \boldsymbol{W}_{\boldsymbol{a}} \overline{\boldsymbol{h}}_{s} & \text { general } \\ \boldsymbol{v}_{a}^{\top} \tanh \left(\boldsymbol{W}_{\boldsymbol{a}}\left[\boldsymbol{h}_{t} ; \overline{\boldsymbol{h}}_{s}\right]\right) & \text { concat }\end{array}\right.
$$

> 其实包括后面的transformer、bert等，都是遵循此范式，不过是score计算方式在dot基础上除以向量维度的0.5次方，为了消除维度对score的影响。

**local attention:** 每次都计算全局的attention权重，计算开销会特别大，特别是输入序列很长的时候（例如一篇文档），所以提出了每次值关注一小部分position。那么怎么确定这一小部分呢？

文中设定了一个context向量$c_t$只关注其窗口$[p_t-D, p_t+D]$内的haidden states，而$p_t$怎么来的呢，文中又定义了这么几种方式：

- Monotonic alignment：$P_t=t$, 这显然不太合适翻译，除非是alignment的任务，例如序列标注任务，其标签和当前t强相关。
- Predictive alignment：$p_{t}=S \cdot \operatorname{sigmoid}\left(\boldsymbol{v}_{p}^{\top} \tanh \left(\boldsymbol{W}_{\boldsymbol{p}} \boldsymbol{h}_{t}\right)\right)$, 通过计算获得，取决于输入$h_t$，即$h_t$ favor alignment points（不知道咋翻译，G点吧），所以monotonic肯定是alignment任务才合适。

然后权重计算方式为：
$$
\boldsymbol{a}_{t}(s)=\operatorname{align}\left(\boldsymbol{h}_{t}, \overline{\boldsymbol{h}}_{s}\right) \exp \left(-\frac{\left(s-p_{t}\right)^{2}}{2 \sigma^{2}}\right)
$$

> 可能细心的观众要问，align是什么东西？好吧，自己看公式7.

可以看到，在普通的权重计算基础上，加入了一个距离的影响因子，距离越小，后面一项越大，说明此更倾向于中心位置到权重大，越远位置越边缘，甚至超过边缘就被裁掉（例如窗口外的就为0）

总结下来local attention关注部分position，而global attention关注全局的position。

# Scaled Dot-Product Attention

> Transformer中attention御用方式。用Transformer完全替代了RNN结构。
>
> [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
> [Weighted Transformer Network for Machine Translation](https://arxiv.org/abs/1711.02132)

不讲5德，直接上公式，
$$
\text {Attention}(Q, K, V)=\operatorname{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right) V \\ = \operatorname{softmax}\left(\left[\begin{array}{c}v_{1} \\ v_{2} \\ \cdots \\ v_{n}\end{array}\right] *\left[v_{1}^{T}, v_{2}^{T}, \ldots, v_{n}^{T}\right]\right) *\left[\begin{array}{c}v_{1} \\ v_{2} \\ \ldots \\ v_{n}\end{array}\right]
$$
其中，$v_i$表示每一步的token的向量，在self attention中，Q,K,V来源于同一个输入X：
$$
Q_i=X_i \times W^q  \\
K_i=X_i \times W^k  \\
V_i=X_i \times W^v
$$

可以看到，和之前attention计算方式差异并不大，分母多了一项$\sqrt{d_{k}}$是为了消除维度对于attention的影响。

同时还提出了多头机制（multi-head attention），有点类似于CNN中的卷积核数目。

![image-20210107233327919](/Users/zhiyang.zzy/project/py3project/MyPicture/Attention/image-20210107233327919.png)

**multi-head attention**：由多个scaled dot-product attention组成，输出结果concat，每一个attention都都有一套不同的权重矩阵$W_{i}^{Q}, W_{i}^{K}, W_{i}^{V}$, 会有不同的初始化值。
$$
\begin{aligned} \operatorname{MultiHead}(Q, K, V) &=\operatorname{Concat}\left(\operatorname{head}_{1}, \ldots, \mathrm{head}_{\mathrm{h}}\right) W^{O} \\ \text { where head }_{\mathrm{i}} &=\operatorname{Attention}\left(Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V}\right) \end{aligned}
$$

> 同时由于Transformer中设置了残差网络，设置隐层单元数目和头数时候要注意是否满足：num_attention_heads * attention_head_size = hidden_size

同时还是用position-wise feed-forward networks、position encoding、layer normalization、residual connection等，继续填坑，后续也有一些对transformer的改造，会继续更新。



## Position-wise Feed-Forward Networks



## position encoding

从attention的计算中可以看出，不同时序的序列计算attention的结果是一样的，导致Transformer会变成一个词袋模型，那么怎么引入序列的信息呢？所以这里就需要对position进行表示，加到原有的token向量上，让每个token中包含位置信息，不同的token之间包含相对位置信息，那么怎么表示这种绝对和相对的位置信息呢？

论文中position encoding使用了公式:
$$
\begin{aligned} P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\ P E_{(p o s, 2 i+1)} &=\cos \left(\text { pos } / 10000^{2 i / d_{\text {model }}}\right) \end{aligned}
$$

> 并且论文试验了使用基于训练的position embedding方式，发现效果差别不大，而上面方式优势在于不需要训练，减少了计算量。

但是看后续bert源码中仍然使用position embedding的方式，即每个position随机初始化一个向量，通过和模型一起训练来拟合最终的position向量。

代码见：[attention.py](https://github.com/InsaneLife/MyPicture/blob/master/Attention/attention.py)

# Reference

- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Weighted Transformer Network for Machine Translation](https://arxiv.org/abs/1711.02132)
- https://www.zhihu.com/question/68482809/answer/1574319286
- https://zhuanlan.zhihu.com/p/47282410
- https://www.jiqizhixin.com/articles/2018-06-11-16
- https://github.com/JayParks/transformer

