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

> 同时，由于URL非常多，会造成表达矩阵维度过大，并且十分稀疏（query的平均度只有8.2，url只有1.8）。因此C和q在









# 问题

怎么融入TTS来做分析。