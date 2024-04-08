# PCL代码学习
```git
//载入pcd文件
std::string infile=argv[1];
pcl::PCLPointCloud2 blob;
pcl::io::loadPCDFile(infile, blob);

//对点云进行下采样
sor.setDownsampleAllData(false);
sor.setInputCloud(cloud);
small_cloud_downsampled=PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
float smalldownsample=static_cast<float> (scale1/decimation);
sor.setLeafSize(smalldownsample,smalldownsample,smalldownsample);
sor.filter(*small_cloud_downsampled);

//计算法线
pcl::NormalEstimationOMP<PointT,PointNT> ne;
ne.setInputCloud(cloud);
ne.setSearchMethod(tree);
ne.setViewPoint(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max());
pcl::PointCloud<PointNT>::Ptr normals_small_scale(new pcl::PointCloud<PointNT>);
ne.setRadiusSearch(scale1);
ne.compute(*normals_small_scale);

//计算 Difference of Normals（DoN）特征
pcl::DifferenceOfNormalsEstimation<PointT,PointNT,PointOutT> don;
don.setInputCloud(cloud);
don.setNormalScaleLarge(normals_large_scale);
don.setNormalScaleSmall(normals_small_scale);
don.computeFeature(*doncloud);

//设置条件，按照条件滤除点云
pcl::ConditionOr<PointOutT>::Ptr range_cond (new pcl::ConditionOr<PointOutT> ());
range_cond->addComparison (pcl::FieldComparison<PointOutT>::ConstPtr (new pcl::FieldComparison<PointOutT> ("curvature", pcl::ComparisonOps::GT, threshold)));
pcl::ConditionalRemoval<PointOutT> condrem;
condrem.setCondition (range_cond);
condrem.setInputCloud (doncloud);

//创建一个 KD 树搜索对象，用于在点云中进行近邻搜索
pcl::search::KdTree<PointOutT>::Ptr segtree (new pcl::search::KdTree<PointOutT>);
segtree->setInputCloud (doncloud);

//使用欧几里得聚类算法进行聚类
std::vector<pcl::PointIndices> cluster_indices;
pcl::EuclideanClusterExtraction<PointOutT> ec;
ec.setClusterTolerance (segradius);
ec.setMinClusterSize (50);
ec.setMaxClusterSize (100000);
ec.setSearchMethod (segtree);
ec.setInputCloud (doncloud);
ec.extract (cluster_indices);







```
