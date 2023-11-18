# PCL介绍
PCL是一个c++库，用于处理3D point <br>
PCL提供eigen库用于矩阵计算 <br>
PCL提供OpenMP库和TBB（threading building blocks线程块）库用于多核并行 <br>
PCL提供FLANN库来做快速的k近邻计算 <br>
<br>
PCL中的boost共享指针：用于传输各个模块和算法数据 <br>
PCL中对3D点云的处理包括：滤波，特征估计，表面重构，模型拟合，点云分割，三维重构等 <br>
<br>
<br>
**PCL的使用流程：** <br>
1.创建processing对象（滤波，特征估计，点云分割）<br>
2.使用setInputCloud将输入的点云数据传到module中 <br>
3.调参 <br>
4.计算得到输出 <br>
5.将原始输入和计算的结果一起传输到FPFH估计对象中 <br>
<img src='image/1.jpg' width="60%" height="60%">
<br>
**PCL的各个库以及作用：** <br>
1. libpcl_filters: 对数据进行过滤，例如下采样，去除离群点，标记提取 <br>
2. libpcl_features:  计算三维特征例如计算表面的法线，曲率，边界点估计，moment不变，主曲率，PPFH和FPFH描述，自旋图片，积分图片，NARF描述，RIFT，RSD，VFH，SIFT等 <br>
3. libpcl_io: 实现输入输出操作 <br>
4. lib_segmentation: 实现聚类提取，通过简单的通用方法来拟合模型，多边形棱镜提取 <br>
5. libpcl_surface: 包含实现三维表面重构所需要的各种算法，例如网格划分，凸体壳，移动最小二乘等 <br>
6. libpcl_registration: 包含实现三维点云重构所需要的各种算法，例如ICP等 <br>
7. libpcl_keypoints: 包含实现各种关键点提取方法，关键点是用于提取特征前的预处理部分 <br>
8. libpcl_range_image: 实现了对从点云数据集创建的范围图像的支持  <br>
<br>
**PCL和ROS**  <br>
perception processing graph（PPG）：感知图像处理 <br>
1. ROS：是最近的三维视觉处理库 <br>
2. PCL：基于ROS库中获得的经验，PCL中的每个算法都以独立构建的模块提供，每个模块都创建一个procssing graphs，这就想ROS系统中一个一个连接起来的节点。 <br>
3. nodelets：是PCL中动态可加载的插件，操作起来类似ROS中的节点，作用是避免不必要数据的复制或序列化/反序列化。 <br>
<br> 
**PCL的可视化：** <br>
PCL的点云可视化是基于VTK。VTK为渲染三维点云和表面数据提供了强大的多平台支持，包括对张量、纹理和体积方法的可视化支持。 <br>
<br>
**PCL的可视化库可以提供以下功能：** <br>
1. 为n维数据提供渲染方法 <br>
2. 提供绘制基本三维形状的方法 <br>
3. 为2维图提供histogram visualization module <br>
4. 提供大量的几何图形和颜色处理程序。 <br>
5. 提供rangeimage 可视化模块。 <br>
 <br>
**PCL的使用实例：** <br>
1. 导航和建图 <br>
2. 目标物识别 <br>
3. 操作和抓取 <br>








