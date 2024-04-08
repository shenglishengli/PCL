#include <QCoreApplication>
#include <iostream>
#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>

#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/don.h>


using namespace std;
using namespace pcl;
using PointT=pcl::PointXYZRGB;
using PointNT=pcl::PointNormal;
using PointOutT=pcl::PointNormal;
using SearchPtr=pcl::search::Search<PointT>::Ptr;

static void sameType();
static void differenceType();

int main(int argc, char *argv[])
{
//    using PointType=pcl::PointXYZ;
//    using CloudType=pcl::PointCloud<PointType>;
//    CloudType::Ptr cloud(new CloudType);
//    cloud->height=10;
//    cloud->width=10;
//    cloud->is_dense=true;
//    cloud->resize(cloud->height*cloud->width);
//    cout<<(*cloud)(0,0)<<endl;
//    PointType p;
//    p.x=1;
//    p.y=2;
//    p.z=3;
//    cout<<(*cloud)(0,0)<<endl;


//    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
//    cloud=pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointCloud>);
//    pcl::io::loadPCDFile<pcl::PointXYZ> ("your_pcd_file.pcd",*cloud);
//    pcl::PointXYZ minPt,maxPt;
//    pcl::getMinMax3D(*cloud,minPt,maxPt);
//    cout<<"Max x:"<<maxPt.x<<endl;
//    cout<<"Max y:"<<maxPt.y<<endl;
//    cout<<"Max z:"<<maxPt.z<<endl;
//    cout<<"Min x:"<<minPt.x<<endl;
//    cout<<"Min y:"<<minPt.y<<endl;
//    cout<<"Min z:"<<minPt.z<<endl;

//    sameType();
//    differenceType();

//    pcl::PointXYZ p_valid;
//    p_valid.x=0;
//    p_valid.y=0;
//    p_valid.z=0;
//    cout<<"is p_valid valid"<<pcl::isFinite(p_valid)<<endl;
//    pcl::PointXYZ p_invalid;
//    p_invalid.x=numeric_limits<float>::quiet_NaN();
//    p_invalid.y=0;
//    p_invalid.z=0;
//    cout<<"is p_invalid valid"<<pcl::isFinite(p_invalid)<<endl;

//    // 创建一个点云对象
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
//        // 向点云对象中添加一些点
//        for (float x = -1.0; x <= 1.0; x += 0.5) {
//            for (float y = -1.0; y <= 1.0; y += 0.5) {
//                for (float z = -1.0; z <= 1.0; z += 0.5) {
//                    pcl::PointXYZ point;
//                    point.x = x;
//                    point.y = y;
//                    point.z = z;
//                    cloud->points.push_back(point);
//                }
//            }
//        }
//        // 设置点云的宽度和高度（这两个参数必须设置，用于表示点云的结构）
//        cloud->width = cloud->points.size();
//        cloud->height = 1;  // 单行点云
//        // 将点云保存为 PCD 文件
//        pcl::PCDWriter writer;
//        writer.write<pcl::PointXYZ>("/home/ren/Documents/result/small_cloud.pcd", *cloud, false);  // 第三个参数表示是否保存为二进制格式




    constexpr double scale1=0.2;
    constexpr double scale2=2;
    constexpr double threshold=0.25;
    double segradius=0.2;
    bool approx=false;
    constexpr double decimation=100;
    if(argc<2)
    {
        std::cerr<<"expected 2 arguments:inputfile outputfile"<<std::endl;
    }
    std::string infile=argv[1];
    std::string outfile=argv[2];
    pcl::PCLPointCloud2 blob;
    pcl::io::loadPCDFile(infile, blob);
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    std::cout<<"loading point cloud...";
    pcl::fromPCLPointCloud2 (blob,*cloud);
    std::cout<<"done."<<std::endl;
    SearchPtr tree;
    if(cloud->isOrganized())
    {
        tree.reset(new pcl::search::OrganizedNeighbor<PointT>());
    }else
    {
        tree.reset(new pcl::search::KdTree<PointT>(false));
    }
    tree->setInputCloud((cloud));
    PointCloud<PointT>::Ptr small_cloud_downsampled;
    PointCloud<PointT>::Ptr large_cloud_downsampled;
    if(approx)
    {
        std::cout<<"downsampling point cloud for approximation"<<std::endl;
        pcl::VoxelGrid<PointT> sor;
        sor.setDownsampleAllData(false);
        sor.setInputCloud(cloud);
        small_cloud_downsampled=PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        float smalldownsample=static_cast<float> (scale1/decimation);
        sor.setLeafSize(smalldownsample,smalldownsample,smalldownsample);
        sor.filter(*small_cloud_downsampled);
        std::cout<<"using leaf size of"<<smalldownsample<<"for small scale"<<small_cloud_downsampled->size()<<"points"<<std::endl;
        large_cloud_downsampled=PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);
        constexpr float largedownsample=static_cast<float>(scale2/decimation);
        sor.setLeafSize(largedownsample,largedownsample,largedownsample);
        sor.filter(*large_cloud_downsampled);
        std::cout<<"using leaf size of"<<largedownsample<<"for small scale"<<large_cloud_downsampled->size()<<"points"<<std::endl;
    }
    pcl::NormalEstimationOMP<PointT,PointNT> ne;
    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree);
    ne.setViewPoint(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max());
    if(scale1>=scale2)
    {
        std::cerr<<"error:large scale must be >small scale!"<<std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout<<"calculateing normals for scale.."<<scale1<<std::endl;
    pcl::PointCloud<PointNT>::Ptr normals_small_scale(new pcl::PointCloud<PointNT>);
    if(approx)
    {
        ne.setSearchSurface(small_cloud_downsampled);
    }
    ne.setRadiusSearch(scale1);
    ne.compute(*normals_small_scale);
    std::cout<<"calculateing normals for scale.."<<scale2<<std::endl;
    pcl::PointCloud<PointNT>::Ptr normals_large_scale(new pcl::PointCloud<PointNT>);
    if(approx)
    {
        ne.setSearchSurface(large_cloud_downsampled);
    }
    ne.setRadiusSearch(scale2);
    ne.compute(*normals_large_scale);
    PointCloud<PointOutT>::Ptr doncloud(new pcl::PointCloud<PointOutT>);
    copyPointCloud(*cloud,*doncloud);
    std::cout<<"calculating doN.."<<std::endl;
    pcl::DifferenceOfNormalsEstimation<PointT,PointNT,PointOutT> don;
    don.setInputCloud(cloud);
    don.setNormalScaleLarge(normals_large_scale);
    don.setNormalScaleSmall(normals_small_scale);
    if(!don.initCompute())
    {
        std::cerr<<"error:could not initialize don feature operator"<<std::endl;
        exit(EXIT_FAILURE);
    }
    don.computeFeature(*doncloud);
    pcl::PCDWriter writer;
    writer.write<PointOutT>(outfile,*doncloud,false);
    std::cout<<"filtering out don mag"<<threshold<<"..."<<std::endl;
    pcl::ConditionOr<PointOutT>::Ptr  range_cond(new pcl::ConditionOr<PointOutT>);
    range_cond->addComparison(pcl::FieldComparison<PointOutT>::ConstPtr(new pcl::FieldComparison<PointOutT>("curvature",pcl::ComparisonOps::GT,threshold)));
    pcl::ConditionalRemoval<PointOutT> condrem;
    condrem.setCondition(range_cond);
    condrem.setInputCloud(doncloud);
    pcl::PointCloud<PointOutT>::Ptr doncloud_filtered(new pcl::PointCloud<PointOutT>);
    condrem.filter(*doncloud_filtered);
    doncloud=doncloud_filtered;
    std::cout<<"filtered Pointcloud:"<<doncloud->size()<<"data point"<<std::endl;
    std::stringstream ss;
    ss<<outfile.substr(0,outfile.length()-4)<<"_threshold"<<threshold<<".pcd";
    writer.write<PointOutT>(ss.str(),*doncloud,false);
    std::cout<<"clustering using euclideanclusterextraction with tolerance<="<<segradius<<"..."<<std::endl;
    pcl::search::KdTree<PointOutT>::Ptr segtree(new pcl::search::KdTree<PointOutT>);
    segtree->setInputCloud(doncloud);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointOutT> ec;
    ec.setClusterTolerance(segradius);
    ec.setMinClusterSize(50);
    ec.setMaxClusterSize(100000);
    ec.setSearchMethod(segtree);
    ec.setInputCloud(doncloud);
    ec.extract(cluster_indices);
    int j=0;
    pcl::PointCloud<PointOutT>::Ptr cloud_cluster_don(new pcl::PointCloud<PointOutT>);
    for(const auto& cluster:cluster_indices)
    {
        for(const auto &index:cluster.indices)
        {
            cloud_cluster_don->points.push_back((*doncloud)[index]);
        }
    }
    cloud_cluster_don->width=cloud_cluster_don->size();
    cloud_cluster_don->height=1;
    cloud_cluster_don->is_dense=true;
    std::cout<<"pointcloud representing the cluster"<<cloud_cluster_don->size()<<"data points"<<std::endl;
    //std::stringstream ss;
    ss<<outfile.substr(0,outfile.length()-4)<<"threshold"<<threshold<<"_cluster_"<<j<<".pcd";
    writer.write<PointOutT>(ss.str(),*cloud_cluster_don,false);
    ++j;




    return 0;
}

void sameType()
{
    using CloudType=pcl::PointCloud<pcl::PointXYZ>;
    CloudType::Ptr cloud(new CloudType);
    CloudType::PointType p;
    p.x=1;
    p.y=2;
    p.z=3;
    cloud->push_back(p);
    cout<<p.x<<" "<<p.y<<" "<<p.z<<endl;
    CloudType::Ptr cloud2(new CloudType);
    copyPointCloud(*cloud,*cloud2);
    CloudType::PointType p_retrieved=(*cloud2)[0];
    cout<<p_retrieved.x<<" "<<p_retrieved.y<<" "<<p_retrieved.z<<endl;
}

void differenceType()
{
    using CloudType=pcl::PointCloud<pcl::PointXYZ>;
    CloudType::Ptr cloud(new CloudType);
    CloudType::PointType p;
    p.x=1;
    p.y=2;
    p.z=3;
    cloud->push_back(p);
    cout<<p.x<<" "<<p.y<<" "<<p.z<<endl;
    using CloudType2=pcl::PointCloud<pcl::PointNormal>;
    CloudType2::Ptr cloud2(new CloudType2);
    copyPointCloud(*cloud,*cloud2);
    CloudType2::PointType p_retrieved=(*cloud2)[0];
    cout<<p_retrieved.x<<" "<<p_retrieved.y<<" "<<p_retrieved.z<<endl;
}
