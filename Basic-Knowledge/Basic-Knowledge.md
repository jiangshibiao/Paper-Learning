## Classical Terms

+ **Sigmoid**: $S(x)=\frac{1}{1+e^{-x}}$
    + 是一个从 $[-\infty,+\infty]$ 到 $[0,1]$ 的映射，可以将任意值转化成概率值。
    + 曾用于神经网络的激活函数，但是很容易梯度消失，因为 $f'(x)=f(x)(1-f(x))$
+ **Softmax**: $S(x_j)=\frac{e^{x_j}}{\sum \limits_{i=1}^n e^{x_i}}$
    + 将任意向量压缩成权值和为 $1$ 的向量，可模拟概率，常用于多分类模型。
+ **CrossEntropy**: $H(p,q)=-\sum \limits_{i=1}^n p(x_i)\log q(x_i)$
    + 交叉熵能够衡量同一个随机变量中的两个不同概率分布的差异程度。
    + 交叉熵前通常先进行一步 Softmax，模拟出近似的概率分布。

## Tricks

+ Clipping Gradient 梯度裁剪
    + 为了解决梯度爆炸带来的 loss 无法下降的问题。设置了一个梯度阈值`clip_gradient`。
    + 在后向传播中求出各参数的梯度后不急着更新，先求出所有梯度的 L2 范数并与阈值相比较。
    + 如果 $||g|| > \mathrm{clip\_gradient}$， 对所有的梯度乘上 $\mathrm{clip\_gradient/||g||}$ 来修正。
