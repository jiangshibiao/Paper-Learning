## Classical Functions

+ **Sigmoid**: $S(x)=\frac{1}{1+e^{-x}}$
    + 是一个从 $[-\infty,+\infty]$ 到 $[0,1]$ 的映射，可以将任意值转化成概率值。
    + 曾用于神经网络的激活函数，但是很容易梯度消失，因为 $f'(x)=f(x)(1-f(x))$
+ **Softmax**: $S(x_j)=\frac{e^{x_j}}{\sum \limits_{i=1}^n e^{x_i}}$
    + 将任意向量压缩成权值和为 $1$ 的向量，可模拟概率，常用于多分类模型。
+ **CrossEntropy**: $H(p,q)=-\sum \limits_{i=1}^n p(x_i)\log q(x_i)$
    + 交叉熵能够衡量同一个随机变量中的两个不同概率分布的差异程度。
    + 交叉熵前通常先进行一步 Softmax，模拟出近似的概率分布。

## Optimizer

+ [相关的综述论文](https://arxiv.org/pdf/1609.04747.pdf)

+ 梯度下降算法 $\theta'=\theta - \eta \nabla J(\theta)$
    + **BGD**（Batch Gradient Descent）：取所有数据的平均梯度更新。
    + **SGD**（Stochastic Gradient Descent）：每次取单组数据的梯度更新。
    + **MBGD**（Mini-Batch Gradient Descent）：取一个 batch 数据的梯度更新。常见做法。
    
+ 动量相关 $\theta'=\theta-v_t$

    + **Momentum**：加快收敛并减少震荡。$v_t=\gamma v_{t-1}+\eta \nabla J(\theta) $
    + **NAG**（Nesterov Accelerated Gradient）：计算梯度时站在未来角度。$v_{t}=\gamma v_{t-1}+\eta \nabla J(\theta-\gamma v_{t-1})$ 

+ 自适应学习率
    + **Adagrad**（Adaptive gradient algorithm）：各参数的学习率随之前的值自适应地减少。
        + 对于每一个参数 $i$，设 $G_i=\sum_t \nabla J_{t,i}(\theta)^2$
        + 则第 $t$ 轮迭代时，参数 $i$ 的变化 $\theta'_i=\theta_i-\frac{\eta}{\sqrt{G_i +\epsilon}} \cdot \nabla J_{t,i}(\theta)$
    + **RMSprop**：为了解决 Adagrad 学习率急剧下降的问题。
        + 把第 $t$ 轮各参数的梯度向量简记为 $g_t$，则  $\theta'=\theta-\frac{\eta}{\sqrt{E[g^{2}]_{t}+\epsilon}} g_{t}$
        + 其中  $E[g^{2}]_{t}=\gamma E[g^{2}]_{t-1}+(1-\gamma) g_{t}^{2}$
        + Hinton 建议设定 $\gamma=0.9, \eta=0.001$
    + **Adam**：Adaptive Moment Estimation
        + 同时借鉴了 RMSprop 和动量方法： $m_{t}=\beta_{1} m_{t-1}+(1-\beta_{1}) g_{t}$，$v_{t}=\beta_{2} v_{t-1}+(1-\beta_{2}) g_{t}^{2}$
        + 多次迭代后会偏向 $0$，要修正这种偏差：$\hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}$，$\hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}$
        + 最后 $\theta’=\theta-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}$。**实践表明，Adam 效果比较好。**


## Some Tricks

+ Clipping Gradient 梯度裁剪
    + 为了解决梯度爆炸带来的 loss 无法下降的问题。设置了一个梯度阈值`clip_gradient`。
    + 在后向传播中求出各参数的梯度后不急着更新，先求出所有梯度的 L2 范数并与阈值相比较。
    + 如果 $||g|| > \mathrm{clip\_gradient}$， 对所有的梯度乘上 $\mathrm{clip\_gradient/||g||}$ 来修正。
