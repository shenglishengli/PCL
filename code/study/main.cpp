#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/io.h>

int main()
{
    pcl::visualization::CloudViewer viewer("Cloud Viewer");
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

    //正方体点云
    for(int i=0; i<20; i++)
    {
        for(int j=0; j<20; j++)
        {
            for(int k = 0; k<20; k++)
            {
                cloud->push_back(pcl::PointXYZ((i-10)/1.0f, (j-10)/1.0f, (k-10)/1.0f));
            }
        }
    }

    viewer.showCloud(cloud);

    while(!viewer.wasStopped());

    return 0;
}
