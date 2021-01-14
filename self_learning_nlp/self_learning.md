# Context-Aware Query Suggestion by Mining Click-Through and Session Data

1. query聚类：将query通过点击二部图计算距离，然后进行聚类，解决稀疏性问题。基于假设：如果两个query很相似，那么他们的点击URL分布也会很相似，所以通过query的点击url的分布来表示query。

![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2020/png/51008/1600822550869-269e5d2c-cfb0-40de-acb1-19dc354b11f4.png)
$$
\overrightarrow{q_{i}}[j]=\left\{\begin{array}{ll}\text { norm }\left(w_{i j}\right) & \text { if edge } e_{i j} \text { exists; } \\ 0 & \text { otherwise }\end{array}\right.
$$


> 其中边的权重$w_{ij}$为$query_i$到$\text{url}_i$的点击次数。如果一个query经常呗搜索，一个不经常被搜索，即使他们分布相似，但他们之间的点击次数会相差很大，为了消除这一影响，对于w会做一个平滑：
> $$
> \operatorname{norm}\left(w_{i j}\right)=\frac{w_{i j}}{\sqrt{\sum_{\forall e_{i k}} w_{i k}^{2}}}
> $$
> 

聚类距离计算：
$$
\operatorname{distance}\left(q_{i}, q_{j}\right)=\sqrt{\sum_{u_{k} \in U}\left(\overrightarrow{q_{i}}[k]-\overrightarrow{q_{j}}[k]\right)^{2}} \tag{2}
$$

> 同时，聚类需要query两两计算距离，真实情况下query会十分多，这样计算效率低下，十分耗时。另外，聚类的个数也是未知的。而且不同的url作为query的表示向量，会十分巨大且稀疏。最后日志也是增量变化，类簇也需要逐步更新。

聚类算法：

![image-20210112152905131](self_learning.assets/image-20210112152905131.png)

对于每个类簇C也会用向量表示：
$$
\vec{c}=\operatorname{norm}\left(\frac{\sum_{q_{i} \in C} \overrightarrow{q_{i}}}{|C|}\right)
$$
然后query和C计算距离使用公式2. 同时用直径D度量来评价簇的紧密性:
$$
D=\sqrt{\frac{\sum_{i=1}^{|C|} \sum_{j=1}^{|C|}\left(\overrightarrow{q_{i}}-\overrightarrow{q_{j}}\right)^{2}}{|C|(|C|-1)}}
$$

> 会设置一个$D_{max}$，每个C的直径D都不能超过$D_{max}$。

> 同时，由于URL非常多，会造成表达矩阵维度过大，并且十分稀疏（query的平均度只有8.2，url只有1.8）。因此在寻找和q最相近的C的时候，找寻的类簇C必须要和q有过共享的url，即有过一条边（这里感觉还可以优化）

因此提出了dimension array data structure。

![image-20210114110607609](self_learning.assets/image-20210114110607609.png)

同时边的连接仍然会很多，作者认为其中一些权重较小的边可能会是潜在的噪声（用户随机点击），可以对其进行修剪来减少计算量。设定$w_{ij}$为$q_i$对于$utl_j$的权重（点击了多少次），那么$w_i = \sum_j w_{ij}$，会设置一个绝对权重和相对权重的阈值来做过滤，文中设定的绝对阈值5相对阈值0.1。

**query推荐**：通过用户的历史查询构建session，并将query用其类簇c表示，构建后缀树来做推荐。

在对话领域，对于未知类别功能的query，可以使用此方法来对query做粗粒度聚类，然后类簇的内部在做一个细粒度的拆分，例如训练一个相似度模型来做区分。



# Learning to Aend, Copy, and Generate for Session-Basedery Suggestion

> 假设：用户说的话未被理解正确，通常会换个说法再说一遍。
>
> 那么如何来挖掘出这种改写呢？

![image-20210114165611635](self_learning.assets/image-20210114165611635.png)

将用户历史query构建为session，通过seq2seq模型来对语义建模，生成推荐的query。

训练目标：1. 生成的损失：每个token预测和标签的交叉熵。2. copy的损失，和生成的损失类似

> 有用户错误、asr错误，通过这种seq2seq的模型，学习session中的语言模型，能够对其中一些错误进行纠正，获得正确的query改写。







和上面方法不同，我们拥有tts播报，怎么融入TTS来做分析？

