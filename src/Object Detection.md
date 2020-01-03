---
marp: true
---
# 目标检测
**李英平**

---
# 目标检测的基本任务

- 目标检测（Object Detection）：分类（Classification）+ 识别（Localization）
![目标检测](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9hZTAxLmFsaWNkbi5jb20va2YvVVRCOFhaTXF2cFBKWEtKa1NhaFZxNnh5ekZYYTAuanBn)

---
# 目标检测的代表技术

- 传统方法
    - VJ检测器
    - HOG检测器
    - DPM
- 基于深度学习的方法
    - One-Stage：YOLO系列、SSD
    - Two-Stage：RCNN系列

---
![目标检测发展历程](https://img-blog.csdnimg.cn/20190522161958407.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTY2NTM2MA==,size_16,color_FFFFFF,t_70)

---
# 目标检测的评价指标
- 对于一张图片上的BBoX和True Box，一般设定一个阈值，超过阈值的定位真正类TP。进而有精确率$P=\frac{Number(TP)}{Number(TotalObjects)}$。
- 对于一类物体，在所有包含此类物体的图像上的精度值就是该类的平均精度$AP_C=\frac{\Sigma P_C}{TotalImages_C}$。
- 对于所有的类别，每一个类别可以计算出一个AP，定义$mAP=\frac{\Sigma AP}{Number(Classes)}$。
- mAP是建立在某一特定的IoU阈值下的，当给出一组IoU阈值时，每一个阈值下有一个mAP，定义$mmAP=\frac{\Sigma mAP}{Number(IoU Threshold)}$。

---
> 2012年，Alex等人提出的AlexNet网络在ImageNet大赛上以远超第二名的成绩夺冠，卷积神经网络乃至深度学习重新引起了广泛的关注。 ——佚名

# 基于深度学习的方法面临的两个关键问题
- 如何利用深度的神经网络去做目标的定位？
- 如何在一个小规模的数据集上训练能力强劲的网络模型？

---
# R-CNN（2014）
- [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](https://ieeexplore.ieee.org/document/6909475)
- Girshick, Ross Donahue, Jeff Darrell, Trevor Malik, Jitendra  --  CVPR

--- 
# 定位问题的解决方法
- R-CNN：Regions with CNN features
![R-CNN](https://img-blog.csdnimg.cn/20181210155342586.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JyaWJsdWU=,size_16,color_FFFFFF,t_70)
- 选择性搜索（Selective Search）、AlexNet、SVM

---
# 选择性搜索
- [Selective search for object recognition](https://link.springer.com/article/10.1007%2Fs11263-013-0620-5)
- 既然不知道目标的尺寸是什么大小，所以就遍历所有尺寸，先将图像分割成若干小尺度的区域，再组建合并成更大尺寸的图片区域。
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/20200101143629.png)
- 选择性搜索要考虑的三个问题：适应不同尺度、策略多样化、算法速度快。

---
# 数据集规模小的解决方案
- 迁移学习（Supervised pre-training + Domian-specific fine-tuning）
    - 使用大规模数据集ImageNet预训练模型使具有良好的泛化能力；
    - 在特定的数据集上继续训练网络中负责分类的卷积层来适应特定问题。

---
# R-CNN的模型效果
![](https://img-blog.csdnimg.cn/20181210155409619.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JyaWJsdWU=,size_16,color_FFFFFF,t_70)
- 边框回归（Bounding Box Regressing， BB）认为候选区域和Ground Truth之间是线性关系，因此边框回归将所有与Ground Truth的IoU（Intersection over Union，交并比）超过设定阈值的边框的位置和尺寸与Ground Truth的宽高和尺寸一并作为输入，训练线性回归器。

---
# R-CNN的不足
1. 耗时过大：R-CNN处理一张图片要数十秒，选择性搜索耗费了一定的时间，选择性搜索生成的2000个候选框串行地经过AlexNet提取特征更是耗时巨大。
2. AlexNet特征提取、SVM分类、BB回归三个模块独立训练，训练复杂，所需空间大。

---
# SPPNet(2014)
- [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  --  PAMI

---
# SPPNet的主要改进
- CNN需要固定尺寸的输入（以满足全连接层的输入要求），在缩放裁切输入图像的过程中会导致图像的信息损失和几何失真。SPPNet使用卷积层提全局特征图，引入了空间金字塔池化层（SPP Layer）在特征图上生成每个候选区的特征向量，保留了更多的信息，也加快了网络的速度。
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/20200101145510.png)

---
# Spatial Pyramid Pooling Layer
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/20200101150115.png)
- 将特征图上每一个候选区使用$1\times 1,2\times 2,4\times 4$三个结构做Max Pooling，将特征连接到一起生成一个固定长度的全连接层输入。

---
# Fast R-CNN（2015）
- [Fast R-CNN](https://ieeexplore.ieee.org/document/7410526/)
- Girshick, Ross  --  CVPR

---
# Fast R-CNN的改进
![](https://upload-images.jianshu.io/upload_images/3940902-7569280b566d0e58.png?imageMogr2/auto-orient/strip|imageView2/2/w/888/format/webp)
- Fast R-CNN避免了对每一个候选框重复提取特征，而是将整张图片归一化之后通过一个VGG-16网络获取整张图片的特征，通过一个**RoI池化层**在整张特征图中选取候选区的部分，池化生成这一候选区的特征向量。
- RoI Pooling：对于每一个候选框，将其分为$H\times W$个区域，在每个区域中做Max Pooling，将结果组合起来就构成了该区域的特征图。

---
# Faster R-CNN（2016）
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://ieeexplore.ieee.org/document/7485869)
- Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun -- PAMI

---
# Faster R-CNN的网络结构
![](https://img-blog.csdnimg.cn/20191105195816143.png)
- Faster-RCNN：VGG-16共享卷积网络提取特征，RPN网络生成候选区域，使用NMS选择其中的一部分，RoI Pooling生成特征向量，而后进行分类和边框回归。
- RPN网络的引入，加快了候选区域的生成，进一步提升了RCNN网络的推断速度。

---
# RPN网络
![](https://img-blog.csdnimg.cn/20191105195832294.png)
- 通过$3\times 3$卷积，为每一个像素点生成9个先验框（Anchor），对于每一个Anchor进行分类，输出其作为前景和背景的分数，并回归出一组定位向量$(t_x, t_y, t_w, t_h)$。

---
# Anchor与坐标变换关系
- Anchor是预先设定的一组矩形区域,在Faster-RCNN中，他被设定为9个矩形框，有三组面积$(128^2, 256^2, 512^2)$和三组长宽比$(1:1, 1:2, 2:1)$组成。设某一Anchor的形状大小为$(x_a, y_a, w_a, h_a)$。
- RPN网络回归的结果就是RPN认为的目标的边框相对于预设Anchor的变换关系。
$$
\left\{ 
\begin{matrix}
b_x = t_x\cdot w_a + x_a \\
b_y = t_y\cdot w_a + y_a \\
b_w = w_a\cdot \exp (t_w) \\
b_h = h_a\cdot \exp (t_h) \\
\end{matrix}
\right.
$$
- 特征图上的坐标和原图上的关系：$B_i=b_i\cdot \Pi_0^N s_n$,其中$s_n$表示第$n$层卷积的步长。

---
# 非极大值抑制（Non-Maximum Suppression，NMS）
```python
def py_cpu_nms(dets, thresh):
    #x1、y1、x2、y2、以及score赋值
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    #每一个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #按照score置信度降序排序
    order = scores.argsort()[::-1]
    keep = [] #保留的结果框集合
    while order.size > 0:
        i = order[0]
        keep.append(i) #保留该类剩余box中得分最高的一个
        #得到相交区域,左上及右下
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        #计算相交的面积,不重叠时面积为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #计算IoU：重叠面积 /（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #保留IoU小于阈值的box
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1] #因为ovr数组的长度比order数组少一个,所以这里要将所有下标后移一位
    return keep
```

---
# Mask R-CNN（2017）
- [Mask R-CNN](https://ieeexplore.ieee.org/document/8237584)
- He, Kaiming Gkioxari, Georgia Dollar, Piotr Girshick, Ross  --  ICCV

---
# Mask R-CNN的改进
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/20200101161125.png)
- 使用RoIAlign替换了RoI Pooling，在分类和回归之外又增加了一个语义分割分支。
- 结合FCN网络设计新增的Mask分支可以用于人的姿态估计等其他任务。

---
# RoIAlign
- Faster R-CNN中采用RoI Pooling的方式，使用RoI Pooling计算的过程中，若特征图尺寸和池化的尺寸不是倍数关系，对于小数坐标，采用最近邻插值，这就导致了采样后的结果和原图有一些偏差。
- RoIAlign采用了双线性插值取代最近邻插值，而后再做Max Pooling或Average Pooling。
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/20200101162209.png)
- Mask R-CNN的消融实验和Faster R-CNN与Faster R-CNN with RoIAlign的对比试验也证明了RoIAlign有效地提高了精度。

---
# Mask分支
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/20200101164106.png)
- 从Faster R-CNN with RoIAlign和Mask R-CNN的精度对比中可以看出，Mask R-CNN的精度提升不止是RoIAlign带来的，作者认为，Mask分支带来的loss的改变间接影响了主干网络的效果。

---
# Two-Stage网络总结
- Two-Stage网络的发展历程：
    - R-CNN：提出了Region Proposal $\rightarrow$ Classification & Regressing的框架结构，提出迁移学习的解决方案。
    - SPPNet：使用整张特征图和SPP层改良了R-CNN的串行特征提取。
    - Fast R-CNN：使用整张特征图和RoI Pooling改良了R-CNN的串行特征提取。
    - Faster R-CNN：使用RPN网络替换了R-CNN的选择性搜索。
    - Mask R-CNN：使用整张特征图和RoIAlign改良了R-CNN的串行特征提取，增加Mask分支提升精度，拓展网络的适用任务领域。

---
# Two-Stage网络总结
![](https://pic2.zhimg.com/80/v2-494f9b18d6bb57156ad0a7f9b2751125_hd.jpg)
- Two-Stage网络具有较高的精度，但是其速度较慢，无法做到实时检测。

---
# YOLO（2016）
- [You only look once: Unified, real-time object detection](https://ieeexplore.ieee.org/document/7780460)
- Joseph Redmon ; Santosh Divvala ; Ross Girshick ; Ali Farhadi  --  CVPR

---
# YOLO的整体思路
- 将候选区的生成和识别两个阶段合二为一，将输入图片分为$7\times 7$个网格（Grid），对于每一个网格，预测出两个边框及其分类信息。
>RCNN：我们先来研究一下图片，嗯，这些位置很可能存在一些对象，你们对这些位置再检测一下看到底是哪些对象在里面。
YOLO：我们把图片大致分成98个区域，每个区域看下有没有对象存在，以及具体位置在哪里。
RCNN：你这么简单粗暴真的没问题吗？
YOLO：当然没有......咳，其实是有一点点问题的，准确率要低一点，但是我非常快！快！快！而且，你的预选区最后不也要回归调整嘛。
RCNN：为什么你用那么粗略的候选区，最后也能得到还不错的bounding box呢？
YOLO：你不是用过边框回归吗？我拿来用用怎么不行了。

---
# YOLO的网络结构
![](https://pic1.zhimg.com/v2-fb3ca1a334bf15697b0c75a1b2accf30_r.jpg)
- Resize $448\times 448(7\times 64) \rightarrow$ GoogleNet-Like Network $\rightarrow$ NMS

---
# YOLO中卷积网络的输出
- YOLO希望每一个Grid负责预测一个目标，换言之，如若Ground Truth中的一个True Box的中心落在了某一Grid里，那么YOLO希望这个目标就由这个Grid来预测。
- YOLO希望每一个Grid都能预测出两个Bounding Box，对于每一个Bounding Box，其特征向量表示为$(x_center, y_center, w, h, confidence, classes)$。
    - 其中$confidence = Pr(Object)*IoU_{prediction}^{ground truth}$，前一项代表目标识别的置信度，若该Grid中存在一个Ground Truth中的Box，该值为$1$，反之为$0$，后一项IoU代表了BB回归的置信度。
    - classes中包含了$C$个值，$C$为待分类的类别数，每一个位置表示若此处存在一个目标，其类别为$C_i$的条件概率$P(C_i|Object)$。
- 综上所述，YOLO的卷积层输出应是一个$Grids\times Grids\times (5\times 2 + Classes)$维的张量，每一个Grid里IoU较大的一个Bounding Box负责预测该目标。

---
# YOLO中卷积网络的输出
![](https://pic2.zhimg.com/v2-1ad557fda288473b0335fe64e03bc049_r.jpg)

---
# YOLO的损失函数
![](https://img-blog.csdnimg.cn/20181113213058140.png)
- 损失函数的五项依次是：负责预测该对象的边界框的定位误差；负责预测该对象的边界框的尺寸误差；负责预测该对象的边界框的置信度误差；不负责预测该对象的边界框的置信度误差；各个边界框的分类误差。

---
# YOLO的损失函数
- 对于BBox的预测，尺寸较小的BBox产生的偏移相对于尺寸较大的BBox的偏移产生的影响更大，为了缓和一下这个情况，YOLO的损失函数中对$w$和$h$求了平方根。
![](https://pic2.zhimg.com/80/v2-bfac676d0f0db4a1d9f4f9aa782341dd_hd.png)
- 一般给$\lambda_{coord}$较大的值，给$\lambda_{noobj}$较小的值。

---
# YOLO中的非极大值抑制
- 对于每一个Grid，类别为$C_i$的对象在第$j$个Bounding Box中的得分$Score_{ij} = P(C_i|Object)*Confidence_j$。因此，每一个Grid会有$2C$个得分，整个网络的输出会有$2CS^2$个得分，对于每一种类别，应该有$2S^2$个的得分。
- 对于所有的得分，设置一个阈值，使用该阈值对所有的得分进行一次过滤，将高于该阈值的BBox进行NMS处理，得到最终的结果。

---
# YOLO模型的优缺点
- YOLO以速度见长，YOLO的处理速度可以达到45 FPS，YOLO的出现，真正使实时目标检测变成了现实，且YOLO作为One-Stage网络，可以进行端到端的训练，也很方便。
- YOLO的网格设置很稀疏，且每个网格只预测两个BBox，所以模型的总体精度不高，YOLO不擅长处理小目标，尤其是堆叠在一起的小目标，会判断成一类，YOLO使用的Pooling层也会使一些信息丢失，造成定位的不准确。

---
# SSD（2016）
- [SSD: Single shot multibox detector](https://link.springer.com/chapter/10.1007%2F978-3-319-46448-0_2)
- Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg  --  LNCS

---
# SSD的改进
- SSD的特征提取网络在VGG-16上进行了修改，替换掉了池化层，删掉了Dropout层和最后一层全连接层，使其变成了一个全卷积网络。
- SSD使用了多尺度特征，在不同尺度下的6张特征图上做出检测，将各层的检测结果一并进行非极大值抑制。
- SSD使用了Default Box来预测BBox。
![](https://img2018.cnblogs.com/blog/785033/201810/785033-20181026193802505-1803963450.png)

---
# Default Box
- SSD中的Default Box在其尺度遵循公式$S_k=S_{min}+\frac{S_{max}-S_{min}}{m-1}(k-1)$,其中$m$表示所使用的特征图的数量，$S_k$表示由浅而深的第$k$张特征图上的Default Box的尺度，$S_{min}$是最浅层的尺度，为$0.2$，$S_{max}$是最深层的尺度，为$0.9$。
- 对于每一层特征图，SSD设4~6个Default Box，其横纵比$\alpha_r$分别为$\{1, 2, 3, \frac{1}{2}, \frac{1}{3}\}$，所以对于每一个Default Box，其宽为$w_k^\alpha=S_k\sqrt{\alpha_r}$，其高为$h_k^\alpha=\frac{S_k}{\sqrt{\alpha_r}}$，当横纵比为1时，增加一个尺度的Default Box，其尺度为$S_k^{'}=\sqrt{S_kS_{k+1}}$。
- 在每一层特征图中，$w_k^\alpha$和$h_k^\alpha$的值与图像的输入大小的乘积代表了Default Box的实际大小，这样的小特征图可以用来检验大目标，大特征图可以用来检验小目标。
- 每一个Default Box应预测出一个BBox，包括其位置尺寸和类别置信度，对于SSD300，其输出是一个$38\times38\times4+19\times19\times6+10\times10\times6+5\times5\times6+3\times3\times4+1\times1\times4=8732$个长度为$(4+classes)$的向量。

---
# SSD的训练机制
![](https://img-blog.csdnimg.cn/20190119191723385.png?)
- 对于Ground Truth中的每一个True Box，在所有的Default Box中选取IoU最大的为正样本，尔后，取所有与任意True Box的IoU大于0.5的做正样本，其余做负样本。训练中，为了平衡正负样本，采用了Hard Negative Mining的策略，每一次对Confidence Loss进行排序，使得正负样本比例保持在1：3。

---
# YOLO v2/YOLO 9000（2017）
- [YOLO9000: Better, faster, stronger](https://ieeexplore.ieee.org/document/8100173)
- Redmon, Joseph Farhadi, Ali  --  CVPR

---
# YOLO v2的改进
![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaTa8ut6HiawDLo6ygc7C4ia5bBpHItUDrPYiccLk3huAw93svM4Pdq9N2tsNEsdoquDp6INtDbmtgGnuhVjuUXl0A/640?wx_fmt=png)
- 批归一化、高分辨率、Anchor、新网络结构Darknet-19、先验框聚类、直接定位预测、Passthrough、多尺度训练

---
# Batch Normalization
- 一般，数据的归一化都只在数据的输入层进行，在训练的过程中，网络参数的变化会导致其后面各层输入的数据分布发生变化，称为内部协变量漂移现象。BN层对每一个批次的训练数据做规范化使其重新具有正态分布，而为了让输出具有更好的表达能力，又增加了一个尺度变换和偏移使其重新分布。
![](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180219084749642-1647361064.png)

---
# Darknet-19
- Darknet-19网络浮点运算量少，运算速度更快。
![](https://raw.githubusercontent.com/Bye-lemon/Pictures/master/20200102213640.png)

---
# 高分辨率 & 多尺度训练
- YOLO使用ImageNet进行预训练，其分类网络的输入尺寸是$224\times224$但是在做检测时，其输入尺寸常常是$448\times448$,预训练模型对其适应并不是很好，YOLO v2重新在$448\times448$的分辨率上对网络参数进行了预训练。
- YOLO v2去掉了全连接层，因此可以适应各种尺寸的输入，由于下采样倍率是32，YOLO v2使用了320~608等10个尺度的输入大小进行训练。
- 由于YOLO v2适应了任意尺度的输入，如果使用更高分辨率的输入做检测，会有更高的精度。

---
# Anchor
- 使用Faster R-CNN中的Anchor机制使得YOLO的精度降低了0.3个百分点，但是使得召回率提升了7个百分点。作者认为，这说明模型有很大的提升空间，但是其有以下两个问题：
    - Anchor的大小是手工设定的，其设定的优劣直接影响到了训练的困难程度。
    - Anchor的使用使得早期训练变得很不稳定，经常会使得BBox偏移到了别的Grid。

---
# 先验框聚类
- YOLO v2希望通过聚类生成一系列先验框来代替手工设定先验框，YOLO v2使用K-means算法进行聚类，其距离度量通过$d(box, centroid)=1-IoU(box,centroid)$进行计算。综合考虑模型复杂度和召回率，YOLO v2使用了5个先验框。
![](https://ss.csdn.net/p?https://mmbiz.qpic.cn/mmbiz_png/iaTa8ut6HiawDLo6ygc7C4ia5bBpHItUDrPibliaEibamYUyLpr1uxypJwPpyuE9fddqd8ChkxBt3JBciaFicmI1fYzLYg/640?wx_fmt=png)
- YOLO v2的输出张量是$W*H*A$个$(5+C)$维向量,其中，$W\times H$为网格数、$B$为采用的先验框的个数、$C$为类别数。

---
# 直接定位预测
- 修改Faster R-CNN预测BBox相对于Anchor的偏移为BBox相对Grid的偏移，通过Sigmoid函数将结果约束在0和1之间，使得模型的训练更加稳定。
![](https://img-blog.csdnimg.cn/20190123113924612.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2trMTIzaw==,size_16,color_FFFFFF,t_70)

---
# Passthrough Layer
- 为了减少Pooling过程中带来了信息损失，提升对小目标的识别效果，YOLO v2通过Passthrough层将一些浅层信息传递到特征图中。
![](https://segmentfault.com/img/remote/1460000016842642)

---
# YOLO9000
- YOLO9000通过WordTree的设计使得网络能检验更多种类的对象。
![](https://segmentfault.com/img/remote/1460000016842648)

---
# YOLO v3（2018）
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- Joseph Redmon, Ali Farhadi  --  arXiv

---
# YOLO v3的改进
- Darknet-53、多尺度特征检测、使用Logistic替代Softmax进行分类。
![](https://img-blog.csdnimg.cn/20190329210004674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2xpdHQxZQ==,size_16,color_FFFFFF,t_70)

---
# YOLO v3的网络结构
- YOLO v3舍弃了所有的池化层和全连接层，整个网络是全卷积网络，张量在其中的尺度变换通过改变卷积的步长实现。
- YOLO v3的最小基本单元是一层卷积后接一层批归一化再接一层Leaky ReLU。
- YOLO v3使用的Darknet-53借鉴了ResNet的残差结构。
- YOLO v3在对BBox进行预测前先将Anchor进行了一个逻辑回归，计算Anchor包含目标的概率的大小，来滤掉不必要的Anchor减小运算量。

---
# 多尺度目标检测
- YOLO v3取了三个不同尺度的特征图进行目标检测，这三个特征图的边长比例为$13:26:52$，这三张特征图由浅层特征图和深层特征图上采样后融合而来，这个操作取代了YOLO v2的Passthrough层用来结合细粒度特征。每张特征图使用3个Anchor进行检测，YOLO v3认为大尺度的特征图感受野小，更利于检测小目标，因此在大尺度特征图上使用小尺寸的Anchor，在小尺度的特征图上感受野大，使用大尺寸的Anchor，这里的Anchor依然使用聚类获得。
- YOLO v3的输出张量由$(52\times 52+26\times 26+13\times 13)\times3=10647$个$(5+classes)$维张量组成。

---
![](https://upload-images.jianshu.io/upload_images/2709767-bc5def91d05e4d3a.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)