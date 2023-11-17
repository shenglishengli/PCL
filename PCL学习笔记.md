# PCL介绍
PCL是一个c++库，用于处理3D point
PCL提供eigen库用于矩阵计算
PCL提供OpenMP库和TBB（threading building blocks线程块）库用于多核并行
PCL提供FLANN库来做快速的k近邻计算

PCL中的boost共享指针：用于传输各个模块和算法数据
PCL中对3D点云的处理包括：滤波，特征估计，表面重构，模型拟合，点云分割，三维重构等

**PCL的使用流程：**
1.创建processing对象（滤波，特征估计，点云分割）<br>
2.使用setInputCloud将输入的点云数据传到module中<br>
3.调参<br>
4.计算得到输出<br>
5.将原始输入和计算的结果一起传输到FPFH估计对象中<br>







