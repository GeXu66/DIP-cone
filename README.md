# Use deep learning object detection to cross a track surrounded by red and blue cone barrels

## 1.1第一圈技术实现路线

第一圈整体上采取视觉估计导航与同步gmapping建图的方式。

视觉方面采用基于Darknet框架的YoloV4模型进行实时推理来检测锥桶获取位置，并融合传统图像处理,进行视觉导航，同时，用扩展卡尔曼滤波融合传感器计算odom结合gmapping进行同步建图。

![img](https://github.com/GeXu66/DIP-cone/blob/main/png/final.gif)

Figure 1 YoloV4在COCO数据集的性能表现

## 1.2第二圈技术实现路线

第二圈采用Triangular Delaunay三角剖分路径规划，基于已经建好的图进行导航。首先根据建好的图识别出锥桶的位置，然后基于Triangular Delaunay算法连接锥桶取终点计算局部路径，进而生成全局路径。


生成路径后需要优化全局路径并进行轨迹跟踪。

优化全局路径具有四个指标可供选择，可以根据不同需求选取不同指标：

 i．最短路径

ii．最小曲率（有或没有迭代调用）

iii．最短时间

iv．考虑动力系统行为的最短时间

最小曲率线非常接近弯道中的最小时间线，但一旦不利用小车的加速度限制就会有所不同。然而，最小时间优化需要更多的参数并且需要更多的计算时间，可能会影响控制频率。

 

## 1.3 创新点

### 1.摄像头雷达传感器融合

使用单目摄像头无法获取锥桶的相对位置，对于第一圈的运动决策来说很不利，所以我们融合图像与点云数据来获得锥桶的深度信息与位置信息。首先通过摄像头获取图像，加载模型进行推理，获得锥桶在图像中的坐标，经过和雷达同步标定后，即可通过矩阵变换利用雷达的数据来估计图像中锥桶的深度信息，实现双目摄像头的效果。

### 2.采用神经网络YoloV4模型，利用OpenCV的DNN模块加速CPU推理

DNN模块是OpenCV最新推出的一个用于读取各种模型文件并进行推理的模块，API清晰明了易于调用。我们首先准备好数据集并利用Darknet框架进行训练。Darknet是底层基于C++的深度学习框架，可以用其训练诸如RCNN系列、YOLO系列、SSD系列的网络模型，这里我们选择了一个高精度轻量级网络模型YOLOV4-Tiny。DNN模块加载cfg文件、读取模型权重并进行前向推理，其速度在CPU可达20FPS，效果显著，因此在保证了一定控制频率的情况下可以应用一些别的算法来提升比赛效果。

### 3.DBSCAN聚类算法判定锥桶

对单线激光雷达而言，每一圈扫到的是一些离散的点，对于程序来说并不知道锥桶在哪里，因此我们需要采用聚类算法来确定锥桶的位置。DBSCAN是基于密度的聚类算法，该方法不需要像K-Means一样指定簇的数目，而是根据邻域和密度阈值来聚类，具有很高的稳定性。

### 4.三角剖分路径规划

由于赛道是由锥桶围成的，所以如果调用move_base中的teb或者dwa算法可能生成的路径会从锥桶中间穿过，如果将膨胀系数调整的过大可能使得导航报错无法运行，因此设计一种新的路径规划算法很有必要。三角剖分路径规划的思想就是根据离散的点生成三角形集合，我们定义红色锥桶和蓝色锥桶连线的三角边为有效边，有效边中点连成的折线即为有效路径，我们在有效路径的基础上进行平滑优化，使得路径效果最好。

逐点插入的Lawson算法是Lawson在1977年提出的，该算法思路简单，易于编程实现。基本原理为：首先建立一个大的三角形或多边形，把所有数据点包围起来，向其中插入一点，该点与包含它的三角形三个顶点相连，形成三个新的三角形，然后逐个对它们进行空外接圆检测，同时用Lawson设计的局部优化过程LOP进行优化，即通过交换对角线的方法来保证所形成的三角网为Delaunay三角网。
 上述基于散点的构网算法理论严密、唯一性好，网格满足空圆特性，较为理想。由其逐点插入的构网过程可知，遇到非Delaunay边时，通过删除调整，可以构造形成新的Delaunay边。在完成构网后，增加新点时，无需对所有的点进行重新构网，只需对新点的影响三角形范围进行局部联网，且局部联网的方法简单易行。同样，点的删除、移动也可快速动态地进行。但在实际应用当中，这种构网算法当点集较大时构网速度也较慢，如果点集范围是非凸区域或者存在内环，则会产生非法三角形。
