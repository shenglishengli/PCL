#include <QCoreApplication>
#include <iostream>
#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
using namespace std;

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
