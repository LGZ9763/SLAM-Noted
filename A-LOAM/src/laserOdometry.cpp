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
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <eigen3/Eigen/Dense>
#include <mutex>
#include <queue>

#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include "lidarFactor.hpp"

#define DISTORTION 0

//corner_correspondence为加入残差优化中的角点约束
//plane_correspondence为加入残差优化中的平面约束
int corner_correspondence = 0, plane_correspondence = 0;

//扫描一圈所需要的时间
constexpr double SCAN_PERIOD = 0.1;
//相邻两帧最近点之间的距离阈值
constexpr double DISTANCE_SQ_THRESHOLD = 25;
//在上一帧中，设置线束搜索范围，此范围用于搜索离clostPoint最近的点
constexpr double NEARBY_SCAN = 2.5;

//输出帧数
int skipFrameNum = 5;
//判断系统是否初始化完毕
bool systemInited = false;

//每帧不同类型的特征点或点击的时间戳
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;

//上一帧角点和面点构建的kdtree
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>());

//记录每一帧ROS格式转向PCL格式的存储器
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

//存储要与当前帧匹配的上一帧中的点集信息
//laserCloudCornerLast记录上一帧的cornerPointsLessSharp，因为当前帧的角点要与上一帧特征点稍大的值进行匹配
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
//laserCloudSurfLast记录上一帧的surfPointsLessFlat，因为当前帧的面点要与上一帧特征点稍小的值进行匹配
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());
//记录每一帧ROS格式转向PCL格式的存储器，这里存储一帧中所有的点
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>());

//记录laserCloudCornerLast中的点的个数
int laserCloudCornerLastNum = 0;
//记录laserCloudSurfLast中的点的个数
int laserCloudSurfLastNum = 0;

// Transformation from current frame to world frame
//当前帧坐标系向里程计坐标系转化的位姿
Eigen::Quaterniond q_w_curr(1, 0, 0, 0);
Eigen::Vector3d t_w_curr(0, 0, 0);

// q_curr_last(x, y, z, w), t_curr_last
double para_q[4] = {0, 0, 0, 1};
double para_t[3] = {0, 0, 0};

//Eigen::Map这里相当于指针一样，所以对para_q进行操作时，q_last_curr也会改变
//记录同一时刻当前帧向上一帧转化所需的位姿，也是前进过程中相邻两帧的位姿变换
Eigen::Map<Eigen::Quaterniond> q_last_curr(para_q);
Eigen::Map<Eigen::Vector3d> t_last_curr(para_t);

//获取发布者传输过来的ROS结构的所有点集
//曲率最大的特征点
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerSharpBuf;
//曲率稍大的特征点
std::queue<sensor_msgs::PointCloud2ConstPtr> cornerLessSharpBuf;
//曲率最小的特征点
std::queue<sensor_msgs::PointCloud2ConstPtr> surfFlatBuf;
//曲率稍小的特征点
std::queue<sensor_msgs::PointCloud2ConstPtr> surfLessFlatBuf;
//所有的点
std::queue<sensor_msgs::PointCloud2ConstPtr> fullPointsBuf;
//互斥量；防止写入内存数据的时间间隔大于接收数据的时间间隔导致数据写入不完整
std::mutex mBuf;

// undistort lidar point
//将所有点补偿到起始时刻，由于这里补偿取了1，所以补偿到了下一帧的起始时刻
//这里认为Lidar时匀速运动模型，所以将上一帧的位姿变化，用来这一帧的运动补偿   q_last_curr和t_last_curr为上一帧的位姿变化
void TransformToStart(PointType const *const pi, PointType *const po)
{
    //interpolation ratio
    double s;
    //DISTORTION确实是否要进行运动补偿，由于Kitti数据集已经进行了运动补偿，所以这里DISTORTION是取0的
    if (DISTORTION)
        //在scanRegistration.cpp里介绍过，发布的点集信息中intensity信息整数部分为线束，小数部分为一帧之内从起始时刻到现在位置所花的时间
        //所以这里的s是指每个点在一周旋转中所在的时刻比例 SCAN_PERIOD代表旋转一周的时间开销
        //以10HZ激光雷达为例，例如一个点在180度的位置，其距离起始位置的时间为0.05s，再除以一周的时间开销为0.1s，所以s为0.5；
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    //由于四元数的旋转采用球面插值，所以这里将旋转四元数进行了球面插值，插值到对应时刻上，计算旋转变化
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    //平移向量变换
    Eigen::Vector3d t_point_last = s * t_last_curr;
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    //计算旋转后的点，将point转化到上一帧中，变为un_point
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    //赋值
    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

// transform all lidar points to the start of the next frame
//将所有的点补偿到结束时刻
void TransformToEnd(PointType const *const pi, PointType *const po)
{
    // undistort point first
    //首先将所有点补偿到起始时刻
    pcl::PointXYZI un_point_tmp;
    TransformToStart(pi, &un_point_tmp);

    //将所有的点转移到结束时刻
    Eigen::Vector3d un_point(un_point_tmp.x, un_point_tmp.y, un_point_tmp.z);
    //推公式
    //P_curr = R_start_curr * P_curr + T_start_curr;
    //P_curr - T_start_curr = R_start_curr * P_curr;
    //P_curr = (R_start_curr)^-1 *(P_curr - T_start_curr);
    //P_end = (R_start_end)^-1 *(P_end - T_start_end);
    //P_end = R_end_start *(P_end - T_start_end);
    Eigen::Vector3d point_end = q_last_curr.inverse() * (un_point - t_last_curr);

    po->x = point_end.x();
    po->y = point_end.y();
    po->z = point_end.z();

    //Remove distortion time info
    po->intensity = int(pi->intensity);
}
//将获取的数据注入到对应的容器中
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharp2)
{
    mBuf.lock();
    cornerSharpBuf.push(cornerPointsSharp2);
    mBuf.unlock();
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2)
{
    mBuf.lock();
    cornerLessSharpBuf.push(cornerPointsLessSharp2);
    mBuf.unlock();
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlat2)
{
    mBuf.lock();
    surfFlatBuf.push(surfPointsFlat2);
    mBuf.unlock();
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2)
{
    mBuf.lock();
    surfLessFlatBuf.push(surfPointsLessFlat2);
    mBuf.unlock();
}

//receive all point cloud
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
    mBuf.lock();
    fullPointsBuf.push(laserCloudFullRes2);
    mBuf.unlock();
}

int main(int argc, char **argv)
{
    //设置节点laserOdometry，定义句柄，这是每个节点函数的必备操作
    ros::init(argc, argv, "laserOdometry");
    ros::NodeHandle nh;
    //获取aloam_velodyne_....launch中的参数mapping_skip_frame信息，如果有参数，则skipFrameNum为launch中的输入值，否则为默认的2
    nh.param<int>("mapping_skip_frame", skipFrameNum, 2);

    printf("Mapping %d Hz \n", 10 / skipFrameNum);

    //订阅节点scanRegistration中发布的信息
    //订阅曲率最大的特征点，将获取的点存入cornerSharpBuf
    ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100, laserCloudSharpHandler);
    //订阅曲率稍微大的特征点，将获取的点存入cornerLessSharpBuf
    ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100, laserCloudLessSharpHandler);
    //订阅曲率最小的特征点，将获取的点存入surfFlatBuf
    ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100, laserCloudFlatHandler);
    //订阅曲率稍微小的特征点，将获取的点存入surfLessFlatBuf
    ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100, laserCloudLessFlatHandler);
    //订阅所有的激光点，将获取的点存入fullPointsBuf
    ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100, laserCloudFullResHandler);

    //定义发布者
    ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 100);

    ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 100);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 100);

    ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init", 100);

    ros::Publisher pubLaserPath = nh.advertise<nav_msgs::Path>("/laser_odom_path", 100);
    
    //发布导航轨迹，这个主要用于rviz显示运动轨迹
    nav_msgs::Path laserPath;

    int frameCount = 0;
    ros::Rate rate(100);

    while (ros::ok())
    {
        //执行一次发布和订阅
        ros::spinOnce();
        //检测一帧数据中的特征点和所有点是否均有数据
        if (!cornerSharpBuf.empty() && !cornerLessSharpBuf.empty() &&
            !surfFlatBuf.empty() && !surfLessFlatBuf.empty() &&
            !fullPointsBuf.empty())
        {
            //获取每个数据的时间戳
            timeCornerPointsSharp = cornerSharpBuf.front()->header.stamp.toSec();
            timeCornerPointsLessSharp = cornerLessSharpBuf.front()->header.stamp.toSec();
            timeSurfPointsFlat = surfFlatBuf.front()->header.stamp.toSec();
            timeSurfPointsLessFlat = surfLessFlatBuf.front()->header.stamp.toSec();
            timeLaserCloudFullRes = fullPointsBuf.front()->header.stamp.toSec();
            //检测特征点是否与这一帧所有的点具有相同的时间戳
            if (timeCornerPointsSharp != timeLaserCloudFullRes ||
                timeCornerPointsLessSharp != timeLaserCloudFullRes ||
                timeSurfPointsFlat != timeLaserCloudFullRes ||
                timeSurfPointsLessFlat != timeLaserCloudFullRes)
            {
                printf("unsync messeage!");
                //如果时间戳不同，则跳出
                ROS_BREAK();
            }
            //获取订阅到点集信息，并将其转化为PCL格式
            mBuf.lock();
            cornerPointsSharp->clear();
            pcl::fromROSMsg(*cornerSharpBuf.front(), *cornerPointsSharp);
            cornerSharpBuf.pop();

            cornerPointsLessSharp->clear();
            pcl::fromROSMsg(*cornerLessSharpBuf.front(), *cornerPointsLessSharp);
            cornerLessSharpBuf.pop();

            surfPointsFlat->clear();
            pcl::fromROSMsg(*surfFlatBuf.front(), *surfPointsFlat);
            surfFlatBuf.pop();

            surfPointsLessFlat->clear();
            pcl::fromROSMsg(*surfLessFlatBuf.front(), *surfPointsLessFlat);
            surfLessFlatBuf.pop();

            laserCloudFullRes->clear();
            pcl::fromROSMsg(*fullPointsBuf.front(), *laserCloudFullRes);
            fullPointsBuf.pop();
            mBuf.unlock();
            
            //定义计时器，记录整个里程计完成一周期后的时间花费
            TicToc t_whole;
            // initializing
            //初始化工作，第一次不进入else中，而是对数据进行初始化
            if (!systemInited)
            {
                systemInited = true;
                std::cout << "Initialization finished \n";
            }
            else
            {
                //获取曲率最大(角点)和最小(面点)的特征点
                int cornerPointsSharpNum = cornerPointsSharp->points.size();
                int surfPointsFlatNum = surfPointsFlat->points.size();
                //定义计时器，记录优化两次的时间
                TicToc t_opt;
                //这里开始构建残差函数进行里程计位姿优化，考虑到时间，对每一帧进行了两次优化
                for (size_t opti_counter = 0; opti_counter < 2; ++opti_counter)
                {
                    corner_correspondence = 0;
                    plane_correspondence = 0;

                    //ceres::LossFunction *loss_function = NULL;
                    //这里定义了一个损失函数，防止局部突变点引起整个拟合结果产生较大的变化，具体参考最小二乘中有突变点引起的整个拟合曲线突变，
                    //这里损失函数使用了Huber损失函数，当成本函数f大于0.1时，那么将对应点的权重减小，具体权重赋值看Huber表达式
                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                    //这是是一个四元数运算的定义
                    ceres::LocalParameterization *q_parameterization =
                        new ceres::EigenQuaternionParameterization();
                    ceres::Problem::Options problem_options;

                    //定义问题
                    ceres::Problem problem(problem_options);
                    //添加参数，由于四元数与普通的向量加减法不同，所以定义q_parameterization执行四元数加减法
                    problem.AddParameterBlock(para_q, 4, q_parameterization);
                    problem.AddParameterBlock(para_t, 3);

                    //定义将点数据补偿到起点后的输出
                    pcl::PointXYZI pointSel;
                    //下面两个定义是pcl里的参数，第一个用来记录邻近点的索引，第二个记录邻近点的距离
                    std::vector<int> pointSearchInd;
                    std::vector<float> pointSearchSqDis;

                    //定义数据关联所花费的时间
                    TicToc t_data;
                    // find correspondence for corner features
                    //构建角点的残差关系
                    for (int i = 0; i < cornerPointsSharpNum; ++i)
                    {
                        //将点补偿到起始位置
                        TransformToStart(&(cornerPointsSharp->points[i]), &pointSel);
                        //从上一帧中曲率稍微大的点集(此中包含曲率最大的点)中搜索离这一帧中pointSel最近的点
                        kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                        //定义变量closestPointInd保存离这一帧中pointSel最近的点，minPointInd2为与closestPointInd对应的点最近的另一条线束的点
                        int closestPointInd = -1, minPointInd2 = -1;
                        //这里进行一个判断，如果这一帧到上一帧之间对应的最近点之间的距离小于DISTANCE_SQ_THRESHOLD时进行后续步骤
                        //由于相邻两帧时间通常较短，如果超出一定阈值的话，必定有误
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            //获取上一帧中离pointSel最近的点closestPoint所在的线束
                            int closestPointScanID = int(laserCloudCornerLast->points[closestPointInd].intensity);

                            //设置搜索上closestPoint附近其他点的距离阈值约束，这句话优点别扭，好好体会
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
                            //在上一帧中，先从索引增加的方向搜索离closestPoint最近的点
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudCornerLast->points.size(); ++j)
                            {
                                //因为要搜索其他娴熟的点
                                //所以如果搜索的点和closestPoint同一线束或小于该线束上，则跳过
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) <= closestPointScanID)
                                    continue;
                                //这里判断是否与closestPoint所在的线束差距太大？如果差距太大，则不再进行判断，因为线束差距太大已经没有意义了
                                //NEARBY_SCAN为设定搜索线束阈值
                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                //计算上一帧中closestPoint和其附近搜索点之间的距离
                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                //这里时搜索最近点的常规套路，记录最近距离和同一真中离losestPoint最近的目标点的索引
                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                            //在上一帧中，接着从索引减小的方向搜索离closestPoint最近的点
                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                //由于要找不同线束上的最近点，所以遇到相同线束或大于线束上的点，则进行跳过
                                // if in the same scan line, continue
                                if (int(laserCloudCornerLast->points[j].intensity) >= closestPointScanID)
                                    continue;
                                //只搜索NEARBY_SCAN内的点，超过这个阈值的话，就没必要进行搜索了
                                // if not in nearby scans, end the loop
                                if (int(laserCloudCornerLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) *
                                                        (laserCloudCornerLast->points[j].x - pointSel.x) +
                                                    (laserCloudCornerLast->points[j].y - pointSel.y) *
                                                        (laserCloudCornerLast->points[j].y - pointSel.y) +
                                                    (laserCloudCornerLast->points[j].z - pointSel.z) *
                                                        (laserCloudCornerLast->points[j].z - pointSel.z);
                                //寻找最小值的常规操作，记录最短距离和最近点的索引
                                if (pointSqDis < minPointSqDis2)
                                {
                                    // find nearer point
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                            }
                        }
                        //将从两个方向进行搜索后的上一帧中离closestPoint最近的点、closestPoint以及pointSel记录为last_point_b、last_point_a和curr_point
                        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
                        {
                            Eigen::Vector3d curr_point(cornerPointsSharp->points[i].x,
                                                       cornerPointsSharp->points[i].y,
                                                       cornerPointsSharp->points[i].z);
                            Eigen::Vector3d last_point_a(laserCloudCornerLast->points[closestPointInd].x,
                                                         laserCloudCornerLast->points[closestPointInd].y,
                                                         laserCloudCornerLast->points[closestPointInd].z);
                            Eigen::Vector3d last_point_b(laserCloudCornerLast->points[minPointInd2].x,
                                                         laserCloudCornerLast->points[minPointInd2].y,
                                                         laserCloudCornerLast->points[minPointInd2].z);

                            double s;
                            //这里判断是否要将每个时刻的纠正加入到残差优化当中
                            if (DISTORTION)
                                s = (cornerPointsSharp->points[i].intensity - int(cornerPointsSharp->points[i].intensity)) / SCAN_PERIOD;
                            else
                                s = 1.0;
                            ceres::CostFunction *cost_function = LidarEdgeFactor::Create(curr_point, last_point_a, last_point_b, s);
                            problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                            corner_correspondence++;
                        }
                    }

                    //构建点和面之间的残差关系
                    // find correspondence for plane features
                    for (int i = 0; i < surfPointsFlatNum; ++i)
                    {
                        //与角点构建相同的操作
                        TransformToStart(&(surfPointsFlat->points[i]), &pointSel);
                        kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);

                        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
                        {
                            closestPointInd = pointSearchInd[0];
                            //与角点构建关系相同，获取上一帧中离pointSel
                            // get closest point's scan ID
                            int closestPointScanID = int(laserCloudSurfLast->points[closestPointInd].intensity);
                            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;
                            //在上一帧中，先从索引增加的方向搜索离closestPoint最近的点
                            // search in the direction of increasing scan line
                            for (int j = closestPointInd + 1; j < (int)laserCloudSurfLast->points.size(); ++j)
                            {
                                //只搜索NEARBY_SCAN内的点，超过这个阈值的话，就没必要进行搜索了
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                                    break;
                                //计算当前帧中选定点与上一帧中的点之间的距离
                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);
                                
                                //如果前一帧中的目标点小于线束并且距离小于设定值的话，将其设为平面的点
                                // if in the same or lower scan line
                                if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                //在大于线束上寻找另外一个点，因为要想构成平面，三个点不能在同一条直线上
                                // if in the higher scan line
                                else if (int(laserCloudSurfLast->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }
                            //接下来在索引缩小方向上搜索
                            // search in the direction of decreasing scan line
                            for (int j = closestPointInd - 1; j >= 0; --j)
                            {
                                // if not in nearby scans, end the loop
                                if (int(laserCloudSurfLast->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                                    break;

                                double pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
                                                        (laserCloudSurfLast->points[j].x - pointSel.x) +
                                                    (laserCloudSurfLast->points[j].y - pointSel.y) *
                                                        (laserCloudSurfLast->points[j].y - pointSel.y) +
                                                    (laserCloudSurfLast->points[j].z - pointSel.z) *
                                                        (laserCloudSurfLast->points[j].z - pointSel.z);

                                // if in the same or higher scan line
                                if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                                {
                                    minPointSqDis2 = pointSqDis;
                                    minPointInd2 = j;
                                }
                                else if (int(laserCloudSurfLast->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                                {
                                    // find nearer point
                                    minPointSqDis3 = pointSqDis;
                                    minPointInd3 = j;
                                }
                            }
                            //如果待选平面上所有点的索引都存在，则保存为Eigen格式
                            if (minPointInd2 >= 0 && minPointInd3 >= 0)
                            {

                                Eigen::Vector3d curr_point(surfPointsFlat->points[i].x,
                                                            surfPointsFlat->points[i].y,
                                                            surfPointsFlat->points[i].z);
                                Eigen::Vector3d last_point_a(laserCloudSurfLast->points[closestPointInd].x,
                                                                laserCloudSurfLast->points[closestPointInd].y,
                                                                laserCloudSurfLast->points[closestPointInd].z);
                                Eigen::Vector3d last_point_b(laserCloudSurfLast->points[minPointInd2].x,
                                                                laserCloudSurfLast->points[minPointInd2].y,
                                                                laserCloudSurfLast->points[minPointInd2].z);
                                Eigen::Vector3d last_point_c(laserCloudSurfLast->points[minPointInd3].x,
                                                                laserCloudSurfLast->points[minPointInd3].y,
                                                                laserCloudSurfLast->points[minPointInd3].z);

                                double s;
                                if (DISTORTION)
                                    s = (surfPointsFlat->points[i].intensity - int(surfPointsFlat->points[i].intensity)) / SCAN_PERIOD;
                                else
                                    s = 1.0;
                                //残差优化块输入
                                ceres::CostFunction *cost_function = LidarPlaneFactor::Create(curr_point, last_point_a, last_point_b, last_point_c, s);
                                problem.AddResidualBlock(cost_function, loss_function, para_q, para_t);
                                plane_correspondence++;
                            }
                        }
                    }

                    //printf("coner_correspondance %d, plane_correspondence %d \n", corner_correspondence, plane_correspondence);
                    printf("data association time %f ms \n", t_data.toc());
                    //如果特征点较少的话，就会打印信息进行提示
                    if ((corner_correspondence + plane_correspondence) < 10)
                    {
                        printf("less correspondence! *************************************************\n");
                    }
                    //进行残差优化
                    TicToc t_solver;
                    ceres::Solver::Options options;
                    //使用稠密QR
                    options.linear_solver_type = ceres::DENSE_QR;
                    //设置迭代次数
                    options.max_num_iterations = 4;
                    options.minimizer_progress_to_stdout = false;
                    ceres::Solver::Summary summary;
                    ceres::Solve(options, &problem, &summary);
                    printf("solver time %f ms \n", t_solver.toc());
                }
                printf("optimization twice time %f \n", t_opt.toc());

                //更新里程计的旋转和平移信息
                //这里使用的是一个关于旋转和平移相乘的公式
                //Matrix_result = Matrix(q_last_curr,t_last_curr;0,1)* Matrix(q_w_curr,t_w_curr;0,1);
                t_w_curr = t_w_curr + q_w_curr * t_last_curr;
                q_w_curr = q_w_curr * q_last_curr;
            }

            TicToc t_pub;

            // publish odometry
            //发布里程计信息
            nav_msgs::Odometry laserOdometry;
            laserOdometry.header.frame_id = "camera_init";
            laserOdometry.child_frame_id = "/laser_odom";
            laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
            laserOdometry.pose.pose.orientation.x = q_w_curr.x();
            laserOdometry.pose.pose.orientation.y = q_w_curr.y();
            laserOdometry.pose.pose.orientation.z = q_w_curr.z();
            laserOdometry.pose.pose.orientation.w = q_w_curr.w();
            laserOdometry.pose.pose.position.x = t_w_curr.x();
            laserOdometry.pose.pose.position.y = t_w_curr.y();
            laserOdometry.pose.pose.position.z = t_w_curr.z();
            pubLaserOdometry.publish(laserOdometry);

            //发布路径信息，用于在rviz中可视化
            geometry_msgs::PoseStamped laserPose;
            laserPose.header = laserOdometry.header;
            laserPose.pose = laserOdometry.pose.pose;
            laserPath.header.stamp = laserOdometry.header.stamp;
            laserPath.poses.push_back(laserPose);
            laserPath.header.frame_id = "camera_init";
            pubLaserPath.publish(laserPath);

            //将所有点移到每一帧的结尾处，这里没有执行
            // transform corner features and plane features to the scan end point
            if (0)
            {
                int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
                for (int i = 0; i < cornerPointsLessSharpNum; i++)
                {
                    TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
                }

                int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
                for (int i = 0; i < surfPointsLessFlatNum; i++)
                {
                    TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
                }

                int laserCloudFullResNum = laserCloudFullRes->points.size();
                for (int i = 0; i < laserCloudFullResNum; i++)
                {
                    TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
                }
            }
            //数据互换，将当前帧记录在上一帧中，进行下一步配准
            pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
            cornerPointsLessSharp = laserCloudCornerLast;
            laserCloudCornerLast = laserCloudTemp;

            laserCloudTemp = surfPointsLessFlat;
            surfPointsLessFlat = laserCloudSurfLast;
            laserCloudSurfLast = laserCloudTemp;

            laserCloudCornerLastNum = laserCloudCornerLast->points.size();
            laserCloudSurfLastNum = laserCloudSurfLast->points.size();

            // std::cout << "the size of corner last is " << laserCloudCornerLastNum << ", and the size of surf last is " << laserCloudSurfLastNum << '\n';
            //将移动到上一帧的数据构建kdtree，PCL里的函数，将曲率稍大的点集进行kdtree管理
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

            //这里的判断是按照帧率进行输出
            if (frameCount % skipFrameNum == 0)
            {
                frameCount = 0;
                //发布已经里程计粗配准完成的帧，即上一帧，这里是角点
                sensor_msgs::PointCloud2 laserCloudCornerLast2;
                pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
                laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudCornerLast2.header.frame_id = "/camera";
                pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

                //发布已经里程计粗配准完成的帧，即上一帧，这里是面点
                sensor_msgs::PointCloud2 laserCloudSurfLast2;
                pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
                laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudSurfLast2.header.frame_id = "/camera";
                pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
                //发布已经里程计粗配准完成的帧，即上一帧，这里是全部点
                sensor_msgs::PointCloud2 laserCloudFullRes3;
                pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
                laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
                laserCloudFullRes3.header.frame_id = "/camera";
                pubLaserCloudFullRes.publish(laserCloudFullRes3);
            }
            printf("publication time %f ms \n", t_pub.toc());
            printf("whole laserOdometry time %f ms \n \n", t_whole.toc());
            if(t_whole.toc() > 100)
                ROS_WARN("odometry process over 100ms");

            frameCount++;
        }
        rate.sleep();
    }
    return 0;
}
