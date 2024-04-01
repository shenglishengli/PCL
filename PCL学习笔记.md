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

**PCL和ROS：**  <br>
perception processing graph（PPG）：感知图像处理 <br>
1. ROS：是最近的三维视觉处理库 <br>
2. PCL：基于ROS库中获得的经验，PCL中的每个算法都以独立构建的模块提供，每个模块都创建一个procssing graphs，这就想ROS系统中一个一个连接起来的节点。 <br>
3. nodelets：是PCL中动态可加载的插件，操作起来类似ROS中的节点，作用是避免不必要数据的复制或序列化/反序列化。 <br>
<br>

**PCL的可视化**  
PCL的点云可视化是基于VTK。VTK为渲染三维点云和表面数据提供了强大的多平台支持，包括对张量、纹理和体积方法的可视化支持。 <br>
<br>

**PCL的可视化库可以提供以下功能**  
1. 为n维数据提供渲染方法 <br>
2. 提供绘制基本三维形状的方法 <br>
3. 为2维图提供histogram visualization module <br>
4. 提供大量的几何图形和颜色处理程序。 <br>
5. 提供rangeimage 可视化模块。 <br>
 <br>
 
**PCL的使用实例**  
1. 导航和建图 <br>
2. 目标物识别 <br>
3. 操作和抓取 <br>

# Linux安装PCL  
**环境**Oracle VM VirtualBox下安装ubuntu
**PCL官网下载地址**
https://pointclouds.org/downloads/  
**PCL官网Linux下安装教程**
https://pcl.readthedocs.io/projects/tutorials/en/master/compiling_pcl_posix.html#compiling-pcl-posix  
```git
git clone https://github.com/PointCloudLibrary/pcl.git
cd pcl
mkdir build
cd build
cmake ..
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j2
make -j2 install
```
如果cmake ..失败，是因为还需要下载依赖库
```git
sudo apt install git cmake libboost-all-dev libeigen3-dev libflann-dev libvtk7-dev libqhull-dev
```
如果安装libvtk7-dev出现下列错误时，安装libvtk7-jni即可
```git
$ sudo apt install libvtk7-dev
Reading package lists... Done
......
The following packages have unmet dependencies:
 libvtk7-dev : Depends: libvtk7-java (= 7.1.1+dfsg1-2) but it is not going to be installed
E: Unable to correct problems, you have held broken packages.

$ sudo apt install libvtk7-java
Reading package lists... Done
Building dependency tree
Reading state information... Done
... ...
The following packages have unmet dependencies:
 libvtk7-java : Depends: libvtk7-jni (= 7.1.1+dfsg1-2) but it is not going to be installed
E: Unable to correct problems, you have held broken packages.

$ sudo apt install libvtk7-jni
Reading package lists... Done
Building dependency tree
Reading state information... Done
The following additional packages will be installed:
  libqt5x11extras5 libvtk7.1 libvtk7.1-qt
Suggested packages:
  vtk7-doc vtk7-examples
The following packages will be REMOVED:
  libpcl-dev libvtk6-dev libvtk6-java libvtk6-jni libvtk6-qt-dev
The following NEW packages will be installed:
  libqt5x11extras5 libvtk7-jni libvtk7.1 libvtk7.1-qt
0 upgraded, 4 newly installed, 5 to remove and 0 not upgraded.
Need to get 38.6 MB of archives.
After this operation, 127 MB of additional disk space will be used.
Do you want to continue? [Y/n] y
......
Setting up libvtk7-jni (7.1.1+dfsg1-2) ...
Processing triggers for libc-bin (2.27-3ubuntu1.4) ...
```

# Linux下使用PCL
**教程地址**  
https://pcl.readthedocs.io/projects/tutorials/en/latest/  
**使用Binaries**  
下载官网数据https://github.com/PointCloudLibrary/data  
1. 使用pcl_viewer查看.pcd文件
```git
pcl_viewer xxxx.pcd
```
2. pcl_pcd_convert_NaN_nan：将pcd中的NaN值转为nan
```git
pcl_pcd_convert_NaN_nan input.pcd output.pcd
```
3. pcl_convert_pcd_ascii_binary:将pcd文件从ASCII码转为二进制文件或者二进制压缩文件
```git
pcl_convert_pcd_ascii_binary <file_in.pcd> <file_out.pcd> 0/1/2 (ascii/binary/binary_compressed) [precision (ASCII)]
```
4. pcl_concatenate_points_pcd ： 将两个或多个PCD文件的点放到到单个PCD文件中
```git
pcl_concatenate_points_pcd <filename 1..N.pcd>
```   
5. pcl_pcd2vtk： 将pcd文件转为vtk文件
```git
pcl_pcd2vtk input.pcd output.vtk
```
6. pcl_pcd2ply： 将pcd文件转为ply文件
```git
pcl_pcd2ply input.pcd output.ply
```
7. pcl_mesh2pcd ：将CAD文件转为pcd文件
8. pcl_octree_viewer：可视化octree
```git
octree_viewer <file_name.pcd> <octree resolution>
```
**PCL中的基础数据结构**   
1. width : width*height=point的个数
2. height ：height=1表示point cloud为无组织的
3. points ：pcl::PointCloud<pcl::PointXYZ>数据类型等价于std::vector<pcl::PointXYZ>
```git
pcl::PointCloud<pcl::PointXYZ> cloud;
std::vector<pcl::PointXYZ> data = cloud.points;
``` 
4. is_dense : 当point的值为非nan时is_dense=true，当point的值为nan时is_dense=false
5. sensor_origin_ : sensor的平移信息
6. sensor_orientation : sensor的旋转信息

**在c++项目中使用pcl**
1. 在某一文件夹例如project文件夹下新建一个pcd_write_test.cpp文件，并将下列内容复制到pcd_write_test.cpp文件中
```git
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

int
  main ()
{
  pcl::PointCloud<pcl::PointXYZ> cloud;

  // Fill in the cloud data
  cloud.width    = 5;
  cloud.height   = 1;
  cloud.is_dense = false;
  cloud.resize (cloud.width * cloud.height);

  for (auto& point: cloud)
  {
    point.x = 1024 * rand () / (RAND_MAX + 1.0f);
    point.y = 1024 * rand () / (RAND_MAX + 1.0f);
    point.z = 1024 * rand () / (RAND_MAX + 1.0f);
  }

  pcl::io::savePCDFileASCII ("test_pcd.pcd", cloud);
  std::cerr << "Saved " << cloud.size () << " data points to test_pcd.pcd." << std::endl;

  for (const auto& point: cloud)
    std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;

  return (0);
}
```
2. 在和pcd_write_test.cpp同级目录下创建CMakelists.txt，并将下列内容复制到CMakelists.txt中
```git
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pcd_write)
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable(pcd_write pcd_write.cpp)
target_link_libraries(pcd_write ${PCL_LIBRARIES})
```
cmake_minimum_required(VERSION 3.5 FATAL_ERROR) ：cmake的版本不能低于3.5 否则报错  
project(pcd_write) ：最终生成的可执行文件叫pcd_write  
find_package(PCL 1.2 REQUIRED) ：PCL的版本不能低于1.2，REQUIRED表示需求PCL中的所有内容  
include_directories(${PCL_INCLUDE_DIRS}) ： 
link_directories(${PCL_LIBRARY_DIRS}) ：  
add_definitions(${PCL_DEFINITIONS}) ：  
add_executable(pcd_write pcd_write.cpp) ：告知cmake从pcd_write.cpp中创建可执行文件，cmake将负责后缀（在Windows平台上为.exe，在UNIX上为空白） 
target_link_libraries(pcd_write_test ${PCL_LIBRARIES}) ：让链接器知道我们链接的库  
3. 还是在和pcd_write_test.cpp同级目录下打开命令行，按下列命令输入命令
```git
$ cd /PATH/TO/MY/GRAND/PROJECT
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./pcd_write_test
```   
**利用pcl做矩阵变换**  
1. 在某一文件夹例如project文件夹下新建一个pcd_write_test.cpp文件，并将下列内容复制到pcd_write_test.cpp文件中
```git
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

// This function displays the help
void
showHelp(char * program_name)
{
  std::cout << std::endl;
  std::cout << "Usage: " << program_name << " cloud_filename.[pcd|ply]" << std::endl;
  std::cout << "-h:  Show this help." << std::endl;
}

// This is the main function
int
main (int argc, char** argv)
{

  // Show help
  if (pcl::console::find_switch (argc, argv, "-h") || pcl::console::find_switch (argc, argv, "--help")) {
    showHelp (argv[0]);
    return 0;
  }

  // Fetch point cloud filename in arguments | Works with PCD and PLY files
  std::vector<int> filenames;
  bool file_is_pcd = false;

  filenames = pcl::console::parse_file_extension_argument (argc, argv, ".ply");

  if (filenames.size () != 1)  {
    filenames = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

    if (filenames.size () != 1) {
      showHelp (argv[0]);
      return -1;
    } else {
      file_is_pcd = true;
    }
  }

  // Load file | Works with PCD and PLY files
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());

  if (file_is_pcd) {
    if (pcl::io::loadPCDFile (argv[filenames[0]], *source_cloud) < 0)  {
      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
      showHelp (argv[0]);
      return -1;
    }
  } else {
    if (pcl::io::loadPLYFile (argv[filenames[0]], *source_cloud) < 0)  {
      std::cout << "Error loading point cloud " << argv[filenames[0]] << std::endl << std::endl;
      showHelp (argv[0]);
      return -1;
    }
  }

  /* Reminder: how transformation matrices work :

           |-------> This column is the translation
    | 1 0 0 x |  \
    | 0 1 0 y |   }-> The identity 3x3 matrix (no rotation) on the left
    | 0 0 1 z |  /
    | 0 0 0 1 |    -> We do not use this line (and it has to stay 0,0,0,1)

    METHOD #1: Using a Matrix4f
    This is the "manual" method, perfect to understand but error prone !
  */
  Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();

  // Define a rotation matrix (see https://en.wikipedia.org/wiki/Rotation_matrix)
  float theta = M_PI/4; // The angle of rotation in radians
  transform_1 (0,0) = std::cos (theta);
  transform_1 (0,1) = -sin(theta);
  transform_1 (1,0) = sin (theta);
  transform_1 (1,1) = std::cos (theta);
  //    (row, column)

  // Define a translation of 2.5 meters on the x axis.
  transform_1 (0,3) = 2.5;

  // Print the transformation
  printf ("Method #1: using a Matrix4f\n");
  std::cout << transform_1 << std::endl;

  /*  METHOD #2: Using a Affine3f
    This method is easier and less error prone
  */
  Eigen::Affine3f transform_2 = Eigen::Affine3f::Identity();

  // Define a translation of 2.5 meters on the x axis.
  transform_2.translation() << 2.5, 0.0, 0.0;

  // The same rotation matrix as before; theta radians around Z axis
  transform_2.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitZ()));

  // Print the transformation
  printf ("\nMethod #2: using an Affine3f\n");
  std::cout << transform_2.matrix() << std::endl;

  // Executing the transformation
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  // You can either apply transform_1 or transform_2; they are the same
  pcl::transformPointCloud (*source_cloud, *transformed_cloud, transform_2);

  // Visualization
  printf(  "\nPoint cloud colors :  white  = original point cloud\n"
      "                        red  = transformed point cloud\n");
  pcl::visualization::PCLVisualizer viewer ("Matrix transformation example");

   // Define R,G,B colors for the point cloud
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (source_cloud, 255, 255, 255);
  // We add the point cloud to the viewer and pass the color handler
  viewer.addPointCloud (source_cloud, source_cloud_color_handler, "original_cloud");

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20); // Red
  viewer.addPointCloud (transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

  viewer.addCoordinateSystem (1.0, "cloud", 0);
  viewer.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
  //viewer.setPosition(800, 400); // Setting visualiser window position

  while (!viewer.wasStopped ()) { // Display the visualiser until 'q' key is pressed
    viewer.spinOnce ();
  }

  return 0;
}
```
2. 在和pcd_write_test.cpp同级目录下创建CMakelists.txt，并将下列内容复制到CMakelists.txt中
```git
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pcl-matrix_transform)
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable (matrix_transform matrix_transform.cpp)
target_link_libraries (matrix_transform ${PCL_LIBRARIES})
```
3. 还是在和pcd_write_test.cpp同级目录下打开命令行，按下列命令输入命令
```git
$ cd /PATH/TO/MY/GRAND/PROJECT
$ mkdir build
$ cd build
$ cmake ..
$ make
$ ./matrix_transform cube.ply
```
**增加自定义的pointT type**  
1.PCL自带的point type：XYZ数据，点特征直方图等等  
```git
PointXYZI  成员：X Y Z intensity
PointXYZRGBA  成员：X Y Z r g b a
PointXY  成员：X Y
interestPoint  成员：X Y Z strength
Normal  成员：normal[] curvature
PointNormal  成员：X Y Z normal[3] curvature
PointXYZRGBNormal  成员：X Y Z R G B A normal[3] curvature
PointXYZINormal  成员：X Y Z intensity normal[3] curvature
PointWithRange  成员：X Y Z range
PointWithViewpoint 成员：X Y Z vp_x vp_y vp_z
MomentInvariants  成员：j1 j2 j3
PrincipalRadiiRSD  成员：r_min r_max
Boundary  成员：boundary_point
PrincipalCurvatures  成员：principal_curvature[3] pc1 pc2
PFHSignature125  成员：pfh[125](点特征直方图)
FPFHSignature33  成员：fpfh[33](快速点特征直方图)
VFHSignature308  成员：vfh[308]
Narf36  成员：X Y Z roll pitch yaw descriptor[36]
BorderDescription 成员：X Y BorderTraits
IntensityGradient 成员：gradient
Histogram  成员：histogram[N]
PointWithScale  成员：X Y Z scale
PointSurfel 成员：X Y Z normal[3] R G B A radius confidence curvature
```
2.PCL不自带，需要自己定义的point type  
**写一个新的PCL类**  
xxx.h  头文件  
```git
 #include <pcl/filters/filter.h>
 namespace pcl
 {
   template<typename PointT>
   class BilateralFilter : public Filter<PointT>
   {
   };
 }
```
xxx.hpp  声明方法  
```git
#include <pcl/filters/bilateral.h>
```
xxx.cpp  实现方法  
```git
 #include <pcl/filters/bilateral.h>
 #include <pcl/filters/impl/bilateral.hpp>
```
CMakeList.txt   build项目
```git
 # Find "set (srcs", and add a new entry there, e.g.,
 set (srcs
      src/conditional_removal.cpp
      # ...
      src/bilateral.cpp
      )

 # Find "set (incs", and add a new entry there, e.g.,
 set (incs
      include pcl/${SUBSYS_NAME}/conditional_removal.h
      # ...
      include pcl/${SUBSYS_NAME}/bilateral.h
      )

 # Find "set (impl_incs", and add a new entry there, e.g.,
 set (impl_incs
      include/pcl/${SUBSYS_NAME}/impl/conditional_removal.hpp
      # ...
      include/pcl/${SUBSYS_NAME}/impl/bilateral.hpp
      )
```
**3D特征是如何在PCL中运作的**  
setInputCloud() ：输入点云数据  
setSearchSurface()  ：遍历点云表面查找邻接点  
setIndices()  ：为点云加上索引  
**估计PointCloud表面的法线**    
1.如何计算点云表面的法线？  
利用协方差矩阵的特征值和特征向量计算法线。  
2.PCL中计算法线的步骤  
```git
pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
```
输入点云数据
```git
ne.setRadiusSearch (0.03);
```
使用0.03m范围内的点来计算法线
```git
compute3DCentroid (cloud, xyz_centroid);
```
估计邻接点群的质心
```git
computeCovarianceMatrix (cloud, xyz_centroid, covariance_matrix);
```
计算协方差矩阵
```git
computePointNormal (const pcl::PointCloud<PointInT> &cloud, const std::vector<int> &indices, Eigen::Vector4f &plane_parameters, float &curvature);
```
计算单个点的法线  
cloud表示输入的点  
indices表示k个最近邻居的集合  
plane_parameters表示xyz三个方向的法线  
curvature表示协方差矩阵的特征值  
```git
flipNormalTowardsViewpoint (const PointT &point, float vp_x, float vp_y, float vp_z, Eigen::Vector4f &normal);
```
令所有的法向量都朝向veiwpoint方向  
**利用积分图像估计法线**   
计算积分图像  
```git
pcl::IntegralImageNormalEstimation<pcl::PointXYZ,pcl::Normal> ne;
```
基于积分图形计算法线  
```git
ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
ne.setMaxDepthChangeFactor(0.02f);
ne.setNormalSmoothingSize(10.0f);
ne.setInputCloud(cloud);
ne.compute(*normals);
```
**计算单个点的直方图（PFH）算子**   
PFH算子是通过点周边K个邻居的几何属性得到的4维特征。  
假设点周边有K个邻居，两两组合得到（假设P1 P2是K个邻居中的两个）  
```git
u=n1(n1是P1的法线)
v=u*(P1-P2)/||P1-P2||
W=U*V
```
```git
u=n2(n2是P2的法线)
v=u*(P1-P2)/||P1-P2||
W=U*V
```
上面的uvw进一步计算得到  
```git
angle1=v.n1
angle2=u.(P1-P2)/d(d是P1和P2的欧几里得距离)
angle3=arctan(w.n1,u.n1)
```
```git
angle1=v.n2
angle2=u.(P1-P2)/d(d是P1和P2的欧几里得距离)
angle3=arctan(w.n2,u.n2)
```
最后会得到4个特征（angle1,angle2,angle3,d） 
```git
computePairFeatures (const Eigen::Vector4f &p1, const Eigen::Vector4f &n1,
                     const Eigen::Vector4f &p2, const Eigen::Vector4f &n2,
                     float &f1, float &f2, float &f3, float &f4);
```
PCL中使用PFH算子的步骤如下：  
1.创建PFH类，并计算点云法线  
```git
pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
pfh.setInputCloud (cloud);
pfh.setInputNormals (normals);
```
2.计算PFH特征  
```git
computePointPFHSignature (const pcl::PointCloud<PointInT> &cloud,
                          const pcl::PointCloud<PointNT> &normals,
                          const std。。了。。。 ::vector<int> &indices,
                          int nr_split,
                          Eigen::VectorXf &pfh_histogram);
```
**计算viewpoint特征直方图（VFH）的算子**   
1.什么是viewpoint特征：视点特征就是计算视点与法线之间的特征数据（角度距离等）  
2.使用方法：
```git
pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
vfh.setInputCloud (cloud);
vfh.setInputNormals (normals);
pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
vfh.setSearchMethod (tree);

// Output datasets
pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

// Compute the features 计算出VFH特征
vfh.compute (*vfhs);
```
**计算惯性矩和偏心距的算子**  
1. 点云特征提取的步骤：点云的协方差矩阵-->特征值，特征向量-->绕当前轴旋转-->用当前轴计算偏心率  
2.使用方法：  
使用pcl::MomentOfInertiaEstimation
```git
//载入点云
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
feature_extractor.setInputCloud (cloud);
feature_extractor.compute ();
//获取MomentOfInertiaEstimation计算的结果
feature_extractor.getMomentOfInertia (moment_of_inertia);
feature_extractor.getEccentricity (eccentricity);
feature_extractor.getAABB (min_point_AABB, max_point_AABB);
feature_extractor.getOBB (min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
feature_extractor.getEigenValues (major_value, middle_value, minor_value);
feature_extractor.getEigenVectors (major_vector, middle_vector, minor_vector);
feature_extractor.getMassCenter (mass_center);
```
**计算旋转投影特征的算子**   
1.使用pcl::ROPSEstimation来提取特征  
2.计算特征的步骤：截取局部表面-->计算局部参考系-->将局部参考系和OX,OY,OZ对齐-->旋转过程中计算点在XY,XZ,YZ的投影-->得到分布矩阵-->计算每个分布矩阵的中心距  
```git
pcl::ROPSEstimation <pcl::PointXYZ, pcl::Histogram <135> > feature_estimator;
  feature_estimator.setSearchMethod (search_method);
  feature_estimator.setSearchSurface (cloud);
  feature_estimator.setInputCloud (cloud);
  feature_estimator.setIndices (indices);
  feature_estimator.setTriangles (triangles);
  feature_estimator.setRadiusSearch (support_radius);
  feature_estimator.setNumberOfPartitionBins (number_of_partition_bins);
  feature_estimator.setNumberOfRotations (number_of_rotations);
  feature_estimator.setSupportRadius (support_radius);
```
**计算全局对齐空间分布特征的算子**   
1.该算子主要用于物体识别和姿态估计  
2.计算步骤：计算点云的参考系-->将点云的参考系和标准坐标系对齐-->根据点云的空间分布计算点云的特征-->根据特征识别物体  
3.具体计算公式步骤：计算点云的协方差矩阵-->计算特征值特征向量-->将特征值从大到小排列对应得到X轴Y轴Z轴-->j将X轴Y轴Z轴与标准坐标系对齐-->得到直方图
4.使用pcl::GASDSignature984/pcl::GASDSignature512来计算全局空间分布特征  
```git
//创建GASD类，并输入点云数据
pcl::GASDColorEstimation<pcl::PointXYZRGBA, pcl::GASDSignature984> gasd;
pcl::GASDEstimation<pcl::PointXYZ, pcl::GASDSignature512> gasd;
  gasd.setInputCloud (cloud);
//计算特征
pcl::PointCloud<pcl::GASDSignature984> descriptor;
gasd.compute (descriptor);
//获得对齐变换
Eigen::Matrix4f trans = gasd.getTransform ();
//跟新直方图
for (std::size_t i = 0; i < std::size_t( descriptor[0].descriptorSize ()); ++i)
  {
    descriptor[0].histogram[i];
  }
```
**使用passthrough过滤器过滤点云**   
1.使用的方法pcl::PassThrough<pcl::PointXYZ>  
2.
```git
pcl::PassThrough<pcl::PointXYZ> pass;
pass.setInputCloud(cloud)
pass.setFilterFieldName("z");
pass.setFilterLimits(0,1);
pass.filter(*cloud_filtered);
```
**使用voxelGriid过滤器对点云进行下采样**  
1.计算原理：计算体素中所有点的质心，用这个点来表示该体素  
2.代码
```git
pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered);
```
**使用staticalOutlinerRemoval移除离群点**   
1.计算原理：计算每个点到周围邻居的距离，查看距离分布是否是告诉分布，如果距离超过了均值和标准差，则该邻居是离群点  
2.代码
```git
// Create the filtering object
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud (cloud);
  sor.setMeanK (50);
  sor.setStddevMulThresh (1.0);
  sor.filter (*cloud_filtered);
```
**使用parameter模型对点进行投影**   
1.计算原理：设置点将被投影到的平面上，ProjectInliers将点投影到平面上  
2.代码
```git
// Create a set of planar coefficients with X=Y=0,Z=1
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  coefficients->values.resize (4);
  coefficients->values[0] = coefficients->values[1] = 0;
  coefficients->values[2] = 1.0;
  coefficients->values[3] = 0;

// Create the filtering object
  pcl::ProjectInliers<pcl::PointXYZ> proj;
  proj.setModelType (pcl::SACMODEL_PLANE);
  proj.setInputCloud (cloud);
  proj.setModelCoefficients (coefficients);
  proj.filter (*cloud_projected);
```
**从点云中提取索引**   
1.计算原理：使用extractIndices过滤器对点云进行分割，从而得到索引  
2.代码  
```git
//使用voxelGrid过滤器进行下采样
 pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud (cloud_blob);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered_blob);
//使用SACSegmentation创建分割对象
 pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (1000);
  seg.setDistanceThreshold (0.01);
//使用ExtractIndices提取索引
  pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers);
  extract.setNegative (false);
  extract.filter (*cloud_p);
  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);
```

**使用conditional或者radiusOutlier removal来移除离群点**  
1.计算原理：  
conditionalRemoval过滤器的计算原理：对点云中不满足一个或多个给定条件的索引进行删除  
radiusOutlierRemoval过滤器的计算原理：计算点云中每个点周围满足距离条件的邻居个数，当邻居个数少于阈值时，就是离群点  
2.代码  
```git
//使用conditionalRemoval过滤器
    pcl::RadiusOutlierRemoval<pcl::PointXYZ> outrem;
    outrem.setInputCloud(cloud);
    outrem.setRadiusSearch(0.8);
    outrem.setMinNeighborsInRadius (2);
    outrem.setKeepOrganized(true);
    outrem.filter (*cloud_filtered);
//使用radiusOutlierRemoval过滤器
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond (new
    pcl::ConditionAnd<pcl::PointXYZ> ());
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::GT, 0.0)));
    range_cond->addComparison (pcl::FieldComparison<pcl::PointXYZ>::ConstPtr (new
    pcl::FieldComparison<pcl::PointXYZ> ("z", pcl::ComparisonOps::LT, 0.8)));
    pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
    condrem.setCondition (range_cond);
    condrem.setInputCloud (cloud);

    condrem.setKeepOrganized(true);
    condrem.filter (*cloud_filtered);
```
**如何从范围图像中提取NARF关键点**  
使用NarfKeypoint narf_keypoint_detector提取关键点
1.计算原理：
2.代码：
```git
pcl::RangeImageBorderExtractor range_image_border_extractor;
pcl::NarfKeypoint narf_keypoint_detector (&range_image_border_extractor);
narf_keypoint_detector.setRangeImage (&range_image);
narf_keypoint_detector.getParameters ().support_size = support_size;
pcl::PointCloud<int> keypoint_indices;
narf_keypoint_detector.compute (keypoint_indices);
```
**如何使用KdTree搜索**  
1.什么是KdTree：是一种数据结构，在点云处理中k-d树都是三维的  
2.计算原理：有点类似深度优先遍历  
3.代码
```git
//创建kd树类
pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud (cloud);
  pcl::PointXYZ searchPoint;
  searchPoint.x = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.y = 1024.0f * rand () / (RAND_MAX + 1.0f);
  searchPoint.z = 1024.0f * rand () / (RAND_MAX + 1.0f);
```
**点云压缩**  
1.计算原理：因为点云一般包含距离，颜色，法线等附加信息，因此点云慧占用大量内存，因此需要对点云进行压缩  
2.使用openNIGrabber对点云进行压缩  
3.代码：
```git
//创建OpenNIGrabber抓取器
pcl::Grabber* interface = new pcl::OpenNIGrabber ();
std::function<void(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
[this] (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud) { cloud_cb_ (cloud); };
connect callback function for desired signal. In this case its a point cloud with color values
boost::signals2::connection c = interface->registerCallback (f);
//开始接收点云
interface->start ();

// 存储压缩点云的字符流
std::stringstream compressedData;
// 压缩点云
PointCloudEncoder->encodePointCloud (cloud, compressedData);
// 解压缩点云
PointCloudDecoder->decodePointCloud (compressedData, cloudOut);
```
**八叉树的搜索操作和空间划分操作**  
1.什么是八叉树：八叉树是一种基于树结构的用来管理三维数据的一种数据结构，该结构内部每个节点都正好有八个子节点     
2.代码
```git
//创建八叉树
  pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree (resolution);
  octree.setInputCloud (cloud);
  octree.addPointsFromInputCloud ();
//使用体素的邻居搜索
if (octree.voxelSearch (searchPoint, pointIdxVec))
  {
    std::cout << "Neighbors within voxel search at (" << searchPoint.x 
     << " " << searchPoint.y 
     << " " << searchPoint.z << ")" 
     << std::endl;
              
    for (std::size_t i = 0; i < pointIdxVec.size (); ++i)
   std::cout << "    " << (*cloud)[pointIdxVec[i]].x 
       << " " << (*cloud)[pointIdxVec[i]].y 
       << " " << (*cloud)[pointIdxVec[i]].z << std::endl;
  }
//K近邻搜索
int K = 10;
  std::vector<int> pointIdxNKNSearch;
  std::vector<float> pointNKNSquaredDistance;
  std::cout << "K nearest neighbor search at (" << searchPoint.x 
            << " " << searchPoint.y 
            << " " << searchPoint.z
            << ") with K=" << K << std::endl;

  if (octree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
  {
    for (std::size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
      std::cout << "    "  <<   (*cloud)[ pointIdxNKNSearch[i] ].x 
                << " " << (*cloud)[ pointIdxNKNSearch[i] ].y 
                << " " << (*cloud)[ pointIdxNKNSearch[i] ].z 
                << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
  }
//半径内邻居搜索
std::vector<int> pointIdxRadiusSearch;
  std::vector<float> pointRadiusSquaredDistance;
  float radius = 256.0f * rand () / (RAND_MAX + 1.0f);
  std::cout << "Neighbors within radius search at (" << searchPoint.x 
      << " " << searchPoint.y 
      << " " << searchPoint.z
      << ") with radius=" << radius << std::endl;
  if (octree.radiusSearch (searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
  {
    for (std::size_t i = 0; i < pointIdxRadiusSearch.size (); ++i)
      std::cout << "    "  <<   (*cloud)[ pointIdxRadiusSearch[i] ].x 
                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].y 
                << " " << (*cloud)[ pointIdxRadiusSearch[i] ].z 
                << " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
  }
```
**无组织点云数据的空间变化检测**   
1.计算原理：通过递归比较八叉树的树结构来判断空间的变化  
2.代码
```git
//实例化第一个点云
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZ> );
//创建基于八叉树的点云变化探测类，OctreePointCloudChangeDetector可以同时保存和管理八叉树
pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree (resolution);
//充实八叉树，即减少八叉树对内存的消耗又保留先前的八叉树结构
octree.switchBuffers ();
//实例化第二个点云
pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZ> );
//使用getPointIndicesFromNewVoxels判断cloudB中的体素在cloudA中是否存在
std::vector<int> newPointIdxVector;
octree.getPointIndicesFromNewVoxels (newPointIdxVector);
```
**如何从点云中创建范围图像**  
1.计算原理：  
2.代码
```git
//定义范围图像的参数
float angularResolution = (float) (  1.0f * (M_PI/180.0f));  //   1.0 degree in radians
float maxAngleWidth     = (float) (360.0f * (M_PI/180.0f));  // 360.0 degree in radians
float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
float noiseLevel=0.00;
float minRange = 0.0f;
int borderSize = 1;

pcl::RangeImage rangeImage;
  rangeImage.createFromPointCloud(pointCloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                  sensorPose, coordinate_frame, noiseLevel, minRange, borderSize);
  
```
**如何从范围图中提取边界**  
1.边界分类，一共分三类，对象边界，阴影边界。面纱点，阴影边界就是激光雷达中常见的边界  
2.代码
```git
//创建用于标记边界和不标记边界的PCD文件
std::string far_ranges_filename = pcl::getFilenameWithoutExtension (filename)+"_far_ranges.pcd";
if (pcl::io::loadPCDFile(far_ranges_filename.c_str(), far_ranges) == -1)
  std::cout << "Far ranges file \""<<far_ranges_filename<<"\" does not exists.\n";
//创建RangeImageBorderExtractor对象，该对象提供范围图像并计算边界信息，计算结果存放在border_descriptions中
pcl::RangeImageBorderExtractor border_extractor (&range_image);
pcl::PointCloud<pcl::BorderDescription> border_descriptions;
border_extractor.compute (border_descriptions);
```
**基于对应分组算法的3D物体识别**  
1.计算原理：使用对应分组算法来聚类点云，并估计当前帧聚类模型的6DOF  
2.代码  
```git
//聚类任务
//加载PCD文件
parseCommandLine (argc, argv);
  if (pcl::io::loadPCDFile (model_filename_, *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }
  if (pcl::io::loadPCDFile (scene_filename_, *scene) < 0)
  {
    std::cout << "Error loading scene cloud." << std::endl;
    showHelp (argv[0]);
    return (-1);
  }
//调整分辨率
 float resolution = static_cast<float> (computeCloudResolution (model));
//计算法线
pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  norm_est.setKSearch (10);
  norm_est.setInputCloud (model);
  norm_est.compute (*model_normals);

  norm_est.setInputCloud (scene);
  norm_est.compute (*scene_normals);
//下采样找到关键点
pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  uniform_sampling.filter (*model_keypoints);
  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;
  uniform_sampling.setInputCloud (scene);
  uniform_sampling.setRadiusSearch (scene_ss_);
  uniform_sampling.filter (*scene_keypoints);
  std::cout << "Scene total points: " << scene->size () << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;
//将3D特诊和关键点关联起来
descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);

  descr_est.setInputCloud (scene_keypoints);
  descr_est.setInputNormals (scene_normals);
  descr_est.setSearchSurface (scene);
  descr_est.compute (*scene_descriptors);
//使用KdTreeFLANN计算模型特征和场景特征之间的关系，使用欧几里得距离找到最相似的模型特征
  for (std::size_t i = 0; i < scene_descriptors->size (); ++i)
  {
    std::vector<int> neigh_indices (1);
    std::vector<float> neigh_sqr_dists (1);
    if (!std::isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
    {
      continue;
    }
    int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
    if(found_neighs == 1 && neigh_sqr_dists[0] < 0.25f) 
    {
      pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
      model_scene_corrs->push_back (corr);
    }
  }
//对找到的对应关系进行聚类，使用的聚类算法是hough3Dgrouping
pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
    clusterer.setHoughBinSize (cg_size_);
    clusterer.setHoughThreshold (cg_thresh_);
    clusterer.setUseInterpolation (true);
    clusterer.setUseDistanceWeight (false);
    clusterer.setInputCloud (model_keypoints);
    clusterer.setInputRf (model_rf);
    clusterer.setSceneCloud (scene_keypoints);
    clusterer.setSceneRf (scene_rf);
    clusterer.setModelSceneCorrespondences (model_scene_corrs);
    clusterer.recognize (rototranslations, clustered_corrs);

```
**隐式形状模型**  
1.什么是隐式形状模型算法：训练步骤：第一步检测关键点，第二部估计关键点，第三步对关键点进行k-means聚类，第四步计算给定云的质心，第五步统计每个visual word的权重。训练步骤结束后就得到训练模型，然后就可以进行搜索过程：第一步检测关键点，第二步在每个特征附近找到最近的聚类（visual word），第三步对训练模型中的每一个visual word进行投票
2.使用ImplicitShapeModel实现隐式形状模型算法
```git

```
**PCL配准API**  
1.什么是pcl配准API：将三位点云对齐到完整模型中的过程称为配准  
2.什么是成对配准：输出两个点云之间的旋转矩阵和平移矩阵  
3.两个点云的配准步骤：找到关键点-->计算每个关键点的特征-->根据特征将两个点云的关键点一一对应起来-->计算旋转矩阵和平移矩阵  
4.关键点：NARF,SIFT,FAST  
特征：NARF,FPFH,BRIEF,SIFT  
一一对应关键点：暴力匹配，kd树最近邻搜索（FLANN），图像空间搜索，索引空间搜索  
计算旋转矩阵和平移矩阵：使用SVD，ICP循环  
**如何通过迭代找到最近点**  
1.为什么要寻找最近点：用于匹配关键点  
2.代码： 
使用IterativeClosestPoint
```git
pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
//最开始的点云
  icp.setInputSource(cloud_in);
//想要的点云
  icp.setInputTarget(cloud_out);
//如果两个点云对齐了，IterativeClosestPoint会输出true
pcl::PointCloud<pcl::PointXYZ> Final;
  icp.align(Final);
  std::cout << "has converged:" << icp.hasConverged() << " score: " <<
  icp.getFitnessScore() << std::endl;
```
**如何增量的对齐点云**   
1.步骤：
2.代码：
```git
//加载数据
 std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
  loadData (argc, argv, data);
//创建ICP对象将两个点云联系起来，进行下采样，计算曲率
void pairAlign (const PointCloud::Ptr cloud_src, const PointCloud::Ptr cloud_tgt, PointCloud::Ptr output, Eigen::Matrix4f &final_transform, bool downsample = false)
//设置迭代次数
reg.setMaximumIterations (2);
//手动迭代30次
for (int i = 0; i < 30; ++i)
{
        [...]
        //计算旋转平移矩阵
        points_with_normals_src = reg_result;
        reg.setInputCloud (points_with_normals_src);
        reg.align (*reg_result);
        //跟踪并累积ICP的返回值
        Ti = reg.getFinalTransformation () * Ti;
        //当第N次和第N+1次的变化矩阵的差值小于阈值时，就更新阈值
        if (std::abs ((reg.getLastIncrementalTransformation () - prev).sum ()) < reg.getTransformationEpsilon ())
   reg.setMaxCorrespondenceDistance (reg.getMaxCorrespondenceDistance () - 0.001);
 prev = reg.getLastIncrementalTransformation ();
        [...]
}
//迭代结束就找到了最佳的转换矩阵

```
**如何交互迭代找到最近云**   
1.作用：主要是一个可视化工具用于查看点云的转换过程  
**如何使用正态分布变换**  
1.作用：使用正态分布（NDT）来计算两个点云之间的转换矩阵，DNT算法也是一种配准算法  
2.代码：
```git
//加载数据两个点云，并将其中一个点云作为参考系
pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud (new pcl::PointCloud<pcl::PointXYZ>);
//对输入点云做预处理，做滤波
pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
  approximate_voxel_filter.setLeafSize (0.2, 0.2, 0.2);
  approximate_voxel_filter.setInputCloud (input_cloud);
  approximate_voxel_filter.filter (*filtered_cloud);
  std::cout << "Filtered cloud contains " << filtered_cloud->size ()
            << " data points from room_scan2.pcd" << std::endl;
//使用NDT算法
 pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
//设置NDT内部的参数
ndt.setMaximumIterations (35);//迭代次数
//ndt开始配准，target_cloud是参考点云。filtered_cloud是需要被转换的点云
ndt.setInputSource (filtered_cloud);
  ndt.setInputTarget (target_cloud);
//初始化变换矩阵
Eigen::AngleAxisf init_rotation (0.6931, Eigen::Vector3f::UnitZ ());
  Eigen::Translation3f init_translation (1.79387, 0.720047, 0);
  Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix ();
//计算对齐结果
pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZ>);
  ndt.align (*output_cloud, init_guess);
```
**手持扫描仪**  
**估计刚性物体的位姿**  
代码：
```git
//加载数据
pcl::io::loadPCDFile<PointNT> (argv[2], *scene_before_downsampling) < 0)
//下采样
pcl::console::print_highlight ("Downsampling...\n");
  pcl::VoxelGrid<PointNT> grid;
  const float leaf = 0.005f;
  grid.setLeafSize (leaf, leaf, leaf);
  grid.setInputCloud (object);
  grid.filter (*object);
  grid.setInputCloud (scene_before_downsampling);
  grid.filter (*scene);
//使用NormalEstimationOMP计算点云法线
 pcl::console::print_highlight ("Estimating scene normals...\n");
  pcl::NormalEstimationOMP<PointNT,PointNT> nest;
  nest.setRadiusSearch (0.005);
  nest.setInputCloud (scene);
  nest.setSearchSurface (scene_before_downsampling);
  nest.compute (*scene);
//使用FPFHEstimationOMP 类计算点云的特征直方图
 pcl::console::print_highlight ("Estimating features...\n");
  FeatureEstimationT fest;
  fest.setRadiusSearch (0.025);
  fest.setInputCloud (object);
  fest.setInputNormals (object);
  fest.compute (*object_features);
  fest.setInputCloud (scene);
  fest.setInputNormals (scene);
  fest.compute (*scene_features);
//SampleConsensusPrerejective 类做ransacx循环，消除杂点对位姿估计的影响
pcl::console::print_highlight ("Starting alignment...\n");
  pcl::SampleConsensusPrerejective<PointNT,PointNT,FeatureT> align;
  align.setInputSource (object);
  align.setSourceFeatures (object_features);
  align.setInputTarget (scene);
  align.setTargetFeatures (scene_features);
  align.setMaximumIterations (50000); // Number of RANSAC iterations
  align.setNumberOfSamples (3); // Number of points to sample for generating/prerejecting a pose
  align.setCorrespondenceRandomness (5); // Number of nearest features to use
  align.setSimilarityThreshold (0.95f); // Polygonal edge length similarity threshold
  align.setMaxCorrespondenceDistance (2.5f * leaf); // Inlier threshold
  align.setInlierFraction (0.25f); // Required inlier fraction for accepting a pose hypothesis
//开始配准/对齐
{
    pcl::ScopeTime t("Alignment");
    align.align (*object_aligned);
  }
//对齐的对象存储在点云object_aligned中。
printf ("\n");
    Eigen::Matrix4f transformation = align.getFinalTransformation ();
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (0,0), transformation (0,1), transformation (0,2));
    pcl::console::print_info ("R = | %6.3f %6.3f %6.3f | \n", transformation (1,0), transformation (1,1), transformation (1,2));
    pcl::console::print_info ("    | %6.3f %6.3f %6.3f | \n", transformation (2,0), transformation (2,1), transformation (2,2));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("t = < %0.3f, %0.3f, %0.3f >\n", transformation (0,3), transformation (1,3), transformation (2,3));
    pcl::console::print_info ("\n");
    pcl::console::print_info ("Inliers: %i/%i\n", align.getInliers ().size (), object->size ());
    pcl::visualization::PCLVisualizer visu("Alignment");
    visu.addPointCloud (scene, ColorHandlerT (scene, 0.0, 255.0, 0.0), "scene");
    visu.addPointCloud (object_aligned, ColorHandlerT (object_aligned, 0.0, 0.0, 255.0), "object_aligned");
    visu.spin ();
```
**如何使用随机样本构建模型**  
**平面模型分割**  
1.代码：
```git
//创建SACSegmentation对象，使用ransac寻找内点
pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setDistanceThreshold (0.01);
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);
//使用内点拟合平面
std::cerr << "Model coefficients: " << coefficients->values[0] << " " 
                                      << coefficients->values[1] << " "
                                      << coefficients->values[2] << " " 
                                      << coefficients->values[3] << std::endl;
```

**圆柱体模型分割**  
1.计算步骤：设置距离阈值范围-->计算每个点的法线-->保存平面模型，保存圆柱形模型  
2.代码
```git
//创建SACSegmentation对象，并设置圆柱体分割参数
pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::SACSegmentation<pcl::PointXYZ> seg;
 seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (0.05);
  seg.setRadiusLimits (0, 0.1);
```
**基于欧几里得提取簇**  
1.计算原理：本质是一种聚类方法，为点云创建kd树-->创建簇列表Q-->将点p加入到Q中，在pi的周围按半径r将点纳入簇中-->当Q中所有点都处理完毕，就将Q中的点全部添加簇C中，并将Q重置为空列表  
2.代码  
使用pcl::EuclideanClusterExtraction提取簇  
```git
//创建KD树对象
pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);
//创建PointIndices 向量，用于保存每一个PointIndices簇的索引
std::vector<pcl::PointIndices> cluster_indices;
//创建EuclideanClusterExtraction 对象，设置半径，用于计算簇
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.02); // 2cm
  ec.setMinClusterSize (100);
  ec.setMaxClusterSize (25000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);
//迭代cluster_indices用于为每一个条目创建新的点云，将当前簇的所有点写入点云中
int j = 0;
  for (const auto& cluster : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto& idx : cluster.indices) {
      cloud_cluster->push_back((*cloud_filtered)[idx]);
    } //*
    cloud_cluster->width = cloud_cluster->size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

```
**针对区域增长进行分割**  
1.计算原理：使用的方法是使用点法线之间的角度进行聚类  
计算每个点的曲率-->对曲率进行排序-->选取曲率值最小的点添加到种子集合中-->计算每个种子点找到邻近点-->计算每个邻近点法线和种子点法线之间的角度，如果小于阈值，则添加到簇-->计算每个邻近点曲率和种子点曲率之间的差值，如果曲率小于阈值，则添加到簇  
2.代码：
使用pcl::RegionGrowing 类实现区域增长过程中进行簇的计算  
```git
//加载数据
pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("region_growing_tutorial.pcd", *cloud) == -1)
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }
//创建pcl::NormalEstimation 类计算法线
  pcl::search::Search<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);
//使用pcl:：RegionGrowing用于分割点云
 pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);
  reg.setMaxClusterSize (1000000);
 reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  reg.setIndices (indices);
  reg.setInputNormals (normals);
 reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);
```
**基于颜色的对增长的区域进行分割**  
1.计算原理：第一步将颜色相近的两个相邻簇合并在一起，第二步使用合并算法进行过分割和欠分割控制  
2.代码：  
使用pcl::RegionGrowingRGB 类进行计算
```git
//加载数据
 pcl::PointCloud <pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZRGB>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZRGB> ("region_growing_rgb_tutorial.pcd", *cloud) == -1 )
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }
//创建pcl::RegionGrowingRGB类，并设置索引，搜索方法
 pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
  reg.setInputCloud (cloud);
  reg.setIndices (indices);
  reg.setSearchMethod (tree);
//设置距离阈值，用于判定邻近点
  reg.setDistanceThreshold (10);
//设置颜色阈值，用于判定点是否属于同一簇
reg.setPointColorThreshold (6);
//设置簇颜色阈值
reg.setRegionColorThreshold (5);
//启动算法
std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);
```
**基于最小切割的分割**  
1.计算原理：计算出点云的中心和半径，从而将点云分为前景点和背景点  
构建的点云图包含一组顶点，一组源点，一组汇点  
构建的图中每一个顶点都通过边将源点和汇点连接起来  
算法为每条边分配权重，权重分为3种类型：距离，背景惩罚，中心距离
2.代码：
使用pcl::MinCutSegmentation类  
```git
//加载数据
  pcl::PointCloud <pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud <pcl::PointXYZ>);
  if ( pcl::io::loadPCDFile <pcl::PointXYZ> ("min_cut_segmentation_tutorial.pcd", *cloud) == -1 )
  {
    std::cout << "Cloud reading failed." << std::endl;
    return (-1);
  }
//创建pcl::MinCutSegmentation类
pcl::MinCutSegmentation<pcl::PointXYZ> seg;
//
 pcl::PointCloud<pcl::PointXYZ>::Ptr foreground_points(new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointXYZ point;
  point.x = 68.97;
  point.y = -18.55;
  point.z = 0.57;
  foreground_points->points.push_back(point);
  seg.setForegroundPoints (foreground_points);
 seg.setSigma (0.25);
  seg.setRadius (3.0433856);
 seg.setNumberOfNeighbours (14);
//启动程序
std::vector <pcl::PointIndices> clusters;
  seg.extract (clusters);
```
**有条件的欧几里得分割算法**  
1.计算原理：该方法和之前的欧几里得聚类方法，区域生长分割方法，基于颜色的分割方法一样，使用了贪婪类/区域生长/洪水填充原理。与之前的方法相比该方法的优点是用户可以自定义聚类约束，如纯欧几里得，平滑度，RGB等。缺点是，没有初始种子系统，没有过度分段和欠分段控制，一次时间效率比较低。  
2.代码
```git
//创建 pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
std::cerr << "Segmenting to clusters...\n", tt.tic ();
  pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
  cec.setInputCloud (cloud_with_normals);
  cec.setConditionFunction (&customRegionGrowing);
  cec.setClusterTolerance (500.0);
  cec.setMinClusterSize (cloud_with_normals->size () / 1000);
  cec.setMaxClusterSize (cloud_with_normals->size () / 5);
  cec.segment (*clusters);
  cec.getRemovedClusters (small_clusters, large_clusters);
  std::cerr << ">> Done: " << tt.toc () << " ms\n";
```
**基于法线差异的分割算法**  
1.计算原理：法线差（DON）是一种高效的计算3D点云的算法  
给定半径，截取点云表面-->计算截取的表面法线-->对所有点的法线做归一化-->过滤掉杂点  
2.代码：  
```git
//创建pcl::NormalEstimationOMP 类，该类利用多线程计算法线
pcl::NormalEstimationOMP<PointXYZRGB, PointNormal> ne;
  ne.setInputCloud (cloud);
  ne.setSearchMethod (tree);
//使用NormalEstimation.setRadiusSearch()计算大半径和小半径法线
std::cout << "Calculating normals for scale..." << scale1 << std::endl;
  pcl::PointCloud<PointNormal>::Ptr normals_small_scale (new pcl::PointCloud<PointNormal>);
  ne.setRadiusSearch (scale1);
  ne.compute (*normals_small_scale);
  // calculate normals with the large scale
  std::cout << "Calculating normals for scale..." << scale2 << std::endl;
  pcl::PointCloud<PointNormal>::Ptr normals_large_scale (new pcl::PointCloud<PointNormal>);
  ne.setRadiusSearch (scale2);
  ne.compute (*normals_large_scale);
//初始化点云便于后面输出
PointCloud<PointNormal>::Ptr doncloud (new pcl::PointCloud<PointNormal>);
  copyPointCloud (*cloud, *doncloud);
//创建pcl::DifferenceOfNormalsEstimation来计算法线差
pcl::DifferenceOfNormalsEstimation<PointXYZRGB, PointNormal, PointNormal> don;
  don.setInputCloud (cloud);
  don.setNormalScaleLarge (normals_large_scale);
  don.setNormalScaleSmall (normals_small_scale);
  if (!don.initCompute ())
  {
    std::cerr << "Error: Could not initialize DoN feature operator" << std::endl;
    exit (EXIT_FAILURE);
  }
  // Compute DoN
  don.computeFeature (*doncloud);
```
**将点云聚类为超素体**  
1.什么是超体素：VCCS是一种3D点云数据种的像素，他们均匀的分布在3D空间中，使用八叉树结构来存储    
2.代码  
使用pcl::SupervoxelClustering，将点云聚类为超素体  
VCCS是一种区域生长算法，以种子中心开始，以R为半径搜索相邻体素-->计算范围内体素的距离，颜色，法线特征-->做归一化处理
```git

```
**使用 ProgressiveMorphologicalFilter 分割识别地面回波**  
1.计算原理：使用的是渐进形态过滤器来分割地面点  
2.代码：
```git
//创建pcl::ProgressiveMorphologicalFilter过滤器
pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;
  pmf.setInputCloud (cloud);
  pmf.setMaxWindowSize (20);
  pmf.setSlope (1.0f);
  pmf.setInitialDistance (0.5f);
  pmf.setMaxDistance (3.0f);
  pmf.extract (ground->indices);
//将索引传递给pcl::ExtractIndices 过滤器
pcl::ExtractIndices<pcl::PointXYZ> extract;
  extract.setInputCloud (cloud);
  extract.setIndices (ground);
  extract.filter (*cloud_filtered);
//地面返回的内容被写入磁盘以供以后检查
  pcl::PCDWriter writer;
  writer.write<pcl::PointXYZ> ("samp11-utm_ground.pcd", *cloud_filtered, false);
//使用相同参数的过滤器，获取非地面对象的返回
extract.setNegative (true);
  extract.filter (*cloud_filtered);
```
**使用 ModelOutlierRemoval 过滤点云**  
1.计算原理：利用已知系数的 SAC_Model从点云中提取参数模型，例如已知是平面或者球体
2.代码：
```git
// position.x: 0, position.y: 0, position.z:0, radius: 1
  pcl::ModelCoefficients sphere_coeff;
  sphere_coeff.values.resize (4);
  sphere_coeff.values[0] = 0;
  sphere_coeff.values[1] = 0;
  sphere_coeff.values[2] = 0;
  sphere_coeff.values[3] = 1;

  pcl::ModelOutlierRemoval<pcl::PointXYZ> sphere_filter;
  sphere_filter.setModelCoefficients (sphere_coeff);
  sphere_filter.setThreshold (0.05);
  sphere_filter.setModelType (pcl::SACMODEL_SPHERE);
```
