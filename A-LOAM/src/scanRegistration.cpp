// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//Notes：   Liu Guangzu            LGZ9763@163.com

#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv2/imgproc.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

//帧率
const double scanPeriod = 0.1;

//定义启动延迟时间
const int systemDelay = 0;
//记录启动延迟时间 
int systemInitCount = 0;
//根据启动延迟时间systemInitCount判断是否可以启动进行工作
bool systemInited = false;
//定义激光雷达的线数，16线/32线/64线
int N_SCANS = 0;
//存储每个点的曲率
float cloudCurvature[400000];
//定义数组记录每个曲率在点集cloudSize中的索引
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

//曲率排序比较函数，常规操作
bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

//定义话题发布者
//发布所有激光点
ros::Publisher pubLaserCloud;
//发布曲率最大的点
ros::Publisher pubCornerPointsSharp;
//发布曲率稍大的点
ros::Publisher pubCornerPointsLessSharp;
//发布曲率最小的点
ros::Publisher pubSurfPointsFlat;
//发布曲率稍小的点
ros::Publisher pubSurfPointsLessFlat;
//发布移除的点
ros::Publisher pubRemovePoints;
//发布每个线束激光的点
std::vector<ros::Publisher> pubEachScan;

//是否要发布每一条线束上的激光点
bool PUB_EACH_LINE = false;
//定义激光雷达的剔除范围，将范围内的激光雷达数据进行剔除
double MINIMUM_RANGE = 0.1; 

template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    //整合cloud_in和cloud_out，使得cloud_out的头信息和容量与cloud_in相同
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        //如果距离距离小于设定值，则不往cloud_out中添加数据
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        //重新设定输出点集的大小
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}
//一个回调函数，用于接受处理点云数据
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    //ROS中不同节点启动通常需要时间，如果订阅房先启动，那么无法在ros master中获取到对应的发布消息。其次刚启动的前几帧数据通常会不稳定进行舍弃
    //这里做得目的是设置一个启动时间，等待所有事件依次均启动后，再进行工作，这样做的优点是防止出现上述问题
    //但是这么做可能导致启动前的几帧数据不会参与计算，直接弃用。这部分由systemDelay进行控制，由于kitti是录制的数据较为稳定，所以这里systemDelay设置为0
    if (!systemInited)
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    //记录整个获取特征点一次所花的时间
    TicToc t_whole;
    //记录准备过程中所花的准备时间
    TicToc t_prepare;
    
    //记录每个线束第五个点的索引
    std::vector<int> scanStartInd(N_SCANS, 0);
    //记录每个线束倒数第五个点的索引
    std::vector<int> scanEndInd(N_SCANS, 0);

    //将订阅到的点云数据转化为pcl格式的点云数据
    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    //定义剔除空值后的点云的索引
    std::vector<int> indices;
    //剔除点云中没有回波信息的NULL值点，例如打向天空的点
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    //移除设定距离内的点
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    //获取目前点集中的点数
    int cloudSize = laserCloudIn.points.size();
    //计算每一帧起始点角度和结束点角度，激光雷达数据获取过程是顺时针，不符合常用的坐标系，常用坐标系是逆时针角度增加，所以下面代码是将顺时针增加变为逆时针增加
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;
    //由于上面数据进行了更改，所以在变化过程中应当将数据放在合理的范围内，以下是对数据可能出现的变化进行了合理的调整
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);
    //定义变量用于确定是否在一圈的前半部分或者后半部分
    bool halfPassed = false;
    //定义变量用于记录最终有多少满足处于扫描线上的激光点
    int count = cloudSize;
    PointType point;
    //定义变量用于存储所有线束中的点，每个线束中的点是一个PointCloud
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    //下面的for循环是将输入的点进行线束划分，确定点所在的线束
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        //计算激光点的俯仰角，用于将激光点分入不同的线束
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        //定义变量用于存储激光点所在的线束
        int scanID = 0;

        //当激光雷达为velodyne16线的激光雷达时，观测角为-15度～+15度
        //所以在计算scanID时+15度将所有度数设置正值，线束计算是由下往上依次是0-16；
        //垂直角度分辨率为2度(30度的角内有16根线)，0.5是由于int是向下取整，不满足四舍五入，所以这里加0.5，让其满足四舍五入
        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        //当激光雷达为velodyne32线的激光雷达时，观测角为-30.67度～+10.67度
        //所以在计算scanID时+30.67(92/3)度将所有度数设置正值，线束计算是由下往上依次是0-32；
        //垂直角度分辨率为1.33(4/3)度
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        //当激光雷达为velodyne64线的激光雷达时，观测角为-24.8度～+2度
        //所以在计算scanID时现将所有度数设置正值，线束计算是由下往上依次是0-16；
        //垂直角度分辨率为0.40度(26.8度的角内有64根线)，0.5是由于int是向下取整，不满足四舍五入，所以这里加0.5，让其满足四舍五入
        else if (N_SCANS == 64)
        { 
            //以32线处的角度为判断条件分割上半部分和下半部分的线束
            /*************可能是由于激光雷达线束不均匀的原因，后期在此处加注***********/ 
            if (angle >= -8.83)
                //位于上半部分激光线束的求解方法 
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                //下半部分线束的求解方法
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);
            //仅使用0-50条的线束，超过该线束段和超出限定角度的点将被舍弃
            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);
        //计算点进行逆时针旋转后的角度所在的范围，分为前半圈和后半圈计算
        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        { 
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        //此处认为激光扫描仪进行均匀扫描，所以采用线性求解的方法计算激光点瞬时的时间戳
        //现在的激光雷达每个点都有时间戳，不需要这么进行计算了
        float relTime = (ori - startOri) / (endOri - startOri);
        //代码没有用到强度信息，所以此处将强度信息存储了其他内容，整数部分是该店所在的线束，小数部分是线束从起始位置到该点的实时时刻，scanPeriod取0.1表示扫描一次需要0.1s
        point.intensity = scanID + scanPeriod * relTime;
        //将激光电存入对应的线束当中
        laserCloudScans[scanID].push_back(point); 
    }
    
    cloudSize = count;
    printf("points size %d \n", cloudSize);
    //由于每个激光电计算曲率需要前五个点和后五个点，所以每个激光线束的前五个点和后五个点无法计算，所以记录每个线束的第五个点和倒数第五个点，计算区间内其他激光点的曲率
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    { 
        //记录每个线束的第五个点在所有点cloudSize中的索引
        scanStartInd[i] = laserCloud->size() + 5;
        //将矩阵形式的数组变为向量形式的数组，方便后续使用
        *laserCloud += laserCloudScans[i];
        //记录每个线束的倒数第五个点在所有点cloudSize中的索引
        scanEndInd[i] = laserCloud->size() - 6;
    }

    printf("prepare time %f \n", t_prepare.toc());

    //计算每个点的曲率(平面光滑度)，计算公式是根据LOAM论文中公式(1)进行计算
    for (int i = 5; i < cloudSize - 5; i++)
    { 
        //解释一下：中间有一个减10倍的当前点，分摊给前后各五个点以后就是公式(1)的后半部分
        //这里只是把所有激光点中的前5个点和后5个点剔除了，没有对每个线束都进行处理，应该是为了代码的整洁，后面数据处理的时候会进行处理，不会受到此处的影响。
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;

        //计算每个点的曲率
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        //记录每个曲率在点集cloudSize中的索引，后面会对曲率进行排序，所以此处记录索引方便后续提取特征激光点，属于常规操作
        cloudSortInd[i] = i;
        //为了均匀的采集每个线束上的点，对确定为特征点后，其附近的前五点和后五点被打上标签cloudNeighborPicked[i] = 1，这些点将不会被设置为特征点
        cloudNeighborPicked[i] = 0;
        //给几个特征不同的点加上标签，cornerPointsSharp的标签为2，cornerPointsLessSharp的标签为1，surfPointsLessFlat的标签为0，surfPointsFlat的标签为-1.
        cloudLabel[i] = 0;
    }

    //定义时间记录变量，记录特征点提取所花费的时间
    TicToc t_pts;

    //记录曲率最大的特征点的集合
    pcl::PointCloud<PointType> cornerPointsSharp;
    //记录曲率稍微大的特征点的集合
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    //记录平面特征点(曲率最小)的集合
    pcl::PointCloud<PointType> surfPointsFlat;
    //记录曲率稍小的特征点的集合
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {
        //如果一个线束中的点少于6个那么这个线束就没必要进行计算
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {
            //将线束分为六等分，从每等分里挑选两个曲率比较大的焦点点，和四个平面点
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            //将线束中的点按照曲率大小从小到大排序
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            //计算排序时间
            t_q_sort += t_tmp.toc();
            //记录每条线束曲率较大的点
            int largestPickedNum = 0;
            //挑选曲率较大的特征点
            for (int k = ep; k >= sp; k--)
            {
                //cloudSortInd现在中的id是根据曲率进行排序后的ID
                int ind = cloudSortInd[k]; 
                //如果目的激光点没有被打上附近已经处理的标签并曲率大于0.1
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {

                    largestPickedNum++;
                    if (largestPickedNum <= 2)
                    {        
                        //将曲率最大的两个点加入到cornerPointsSharp和cornerPointsLessSharp；
                        //给曲率最大的点加上标签2                
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else if (largestPickedNum <= 20)
                    {
                        //将曲率稍微大的点加入到cornerPointsLessSharp。加入18个曲率稍微大的点
                        //给曲率稍微大的点加上标签1                          
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else
                    {
                        break;
                    }
                    //将该点打上附近已经处理的标签
                    cloudNeighborPicked[ind] = 1; 
                    
                    //在满足距离条件后，对同一线束后五个点打上附近已经处理的标签
                    for (int l = 1; l <= 5; l++)
                    {
                        //如果相邻的点之间的距离小于0.05时，则给相邻的点加上标签，点之间的距离大于0.05时，就停止标记
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    //在满足距离条件后，对同一线束前五个点打上附近已经处理的标签
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            //挑选曲率较小的平面点
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
             //如果目的激光点没有被打上附近已经处理的标签并曲率小于0.1
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    //将曲率最小的4个点加入到surfPointsFlat；
                    //给曲率最小的4个点加上标签-1   
                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }
                    //将该点打上附近已经处理的标签
                    cloudNeighborPicked[ind] = 1;
                    //与上面曲率点相同打标签
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            //对于剩下没有打标签的点统一设为稍微平滑的点
            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }
        //将稍微平滑的点进行将采样，减少需要处理的点
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);
        //将将采样过后的点写入集合当中
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    //将处理的所有激光电发布
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    //将处理的曲率最大的点发布
    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    //将处理的曲率稍大的点发布
    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    //将处理的曲率最小的点发布
    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    //将处理的曲率稍大的点发布
    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    //将每一条线束的激光电进行发布
    // pub each scam
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

int main(int argc, char **argv)
{
    //初始化配准节点，节点名称为scanRegistration
    ros::init(argc, argv, "scanRegistration");
    //初始化句柄
    ros::NodeHandle nh;
    //获取launch文件中激光雷达的扫描线，如果每有获取到N_SCANS值，那么给N_SCANS赋予默认值16线
    nh.param<int>("scan_line", N_SCANS, 16);
    //获取launch文件中的最小舍弃范围，将阈值内的点云剔除，防止扫描到的激光雷达载体数据影响后期配准，如果没有获取到数据，则取默认值0.1，单位是米
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
    //打印输出激光雷达线数
    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
    //获取激光雷达发出的数据
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);
    //发布线束激光点的数据
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    //控制callback启动
    ros::spin();

    return 0;
}
