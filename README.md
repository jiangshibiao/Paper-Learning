# Paper Learning
 > try to learn some classical papers

## 目标检测：YOLOv1

+ [论文地址](http://arxiv.org/abs/1506.02640)
+  注：[参考此文章进行学习和总结](https://blog.csdn.net/guleileo/article/details/80581858)
+ 核心思想：利用整张图作为网络的输入，直接在输出层回归 bounding box 的位置及其所属的类别。
+ 实现方法
	1. 将整个图片分成 $S \times S$ 的网格（每个网格是一个预测单元。如果有一个 object 中心落入其中，就由它进行预测）。
	2. 每个网格都来预测 $B$ 个
