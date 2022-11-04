// Author:   Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk
// Notes：   Liu Guangzu            LGZ9763@163.com

#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <geometry_msgs/PoseStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <eigen3/Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    //以二进制的形式读取雷达文件，kitti中激光雷达数据由(x,y,z,r)四个元素组成，其中r这里表示反射率
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    //将指针移到文件末尾
    lidar_data_file.seekg(0, std::ios::end);
    //通过获取文件长度除以每个元素所占的字节获取总共有多少元素
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    //将指针移回开头
    lidar_data_file.seekg(0, std::ios::beg);
    //定义存放元素的容器
    std::vector<float> lidar_data_buffer(num_elements);
    //读取数据
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

int main(int argc, char** argv)
{
    //初始化节点，节点名称为“kitti_helper”，rosnode list会有kitti_helper
    ros::init(argc, argv, "kitti_helper");
    //节点句柄初始化，后面的发布和接收都是这个句柄里的函数
    ros::NodeHandle n("~");
    //定义三个string 分别是输入文件目录dataset_folder，
    //不同的kitti数据文件名sequence_number，如00 01.....
    //输出bag包文件路径 output_bag_file
    std::string dataset_folder, sequence_number, output_bag_file;
    //从kitti_helper.launch中提取设置的参数
    n.getParam("dataset_folder", dataset_folder);
    n.getParam("sequence_number", sequence_number);
    std::cout << "Reading sequence " << sequence_number << " from " << dataset_folder << '\n';
    bool to_bag;
    //获取是否生成bag包
    n.getParam("to_bag", to_bag);
    if (to_bag)
        //从launch文件中读取output_bag_file文件
        n.getParam("output_bag_file", output_bag_file);
    //定义了一个发布间隔，根据后面rosRate可知，public_delay和发布帧率之间的关系为r=10/public_delay，当publish_delay为1时，发布频率为每秒10帧。
    int publish_delay;
    n.getParam("publish_delay", publish_delay);
    publish_delay = publish_delay <= 0 ? 1 : publish_delay;//三目运算符，如果延迟时间小于等于0时，则发布间隔为0.1s，其余时间为0.public_delay

    //定义雷达话题发布者，在rostopic list 会有 /velodyne_points，用于发布雷达数据，发布者实时保存最近发布的两帧雷达数据
    ros::Publisher pub_laser_cloud = n.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 2);

    //定义图像话题发布者，在rostopic list会有 /image_left，/image_right，用于发布图像消息，发布者实时保存最近发布的两帧图像数据
    image_transport::ImageTransport it(n);
    image_transport::Publisher pub_image_left = it.advertise("/image_left", 2);
    image_transport::Publisher pub_image_right = it.advertise("/image_right", 2);
    //定义真实方向和位置话题发布者，在rostopic list会有/odometry_gt，用于发布真实有箭头的方向话题，该话题是由一个四元数和一个平移向量组成
    ros::Publisher pubOdomGT = n.advertise<nav_msgs::Odometry> ("/odometry_gt", 5);
    //定义一个真实方向和位置的消息集合，用于/odometry_gt发布使用
    nav_msgs::Odometry odomGT;
    //定义真实位姿的ID
    odomGT.header.frame_id = "camera_init";
    //此处应该定义速度的ID，为什么代码的ID为真实地面呢？？？？？？？？？？？
    odomGT.child_frame_id = "ground_truth";

    //定义一个轨迹消息发布者，主要用来发布路径到rviz，在rostopic list会有/path_gt
    ros::Publisher pubPathGT = n.advertise<nav_msgs::Path> ("/path_gt", 5);
    //定义路径的消息集合
    nav_msgs::Path pathGT;
    //定义路径的ID
    pathGT.header.frame_id = "camera_init";

    //读取时间戳数据
    std::string timestamp_path = "sequences/" + sequence_number + "/times.txt";
    std::ifstream timestamp_file(dataset_folder + timestamp_path, std::ifstream::in);
    //读取真实轨迹数据
    std::string ground_truth_path = "results/" + sequence_number + ".txt";
    std::ifstream ground_truth_file(dataset_folder + ground_truth_path, std::ifstream::in);

    rosbag::Bag bag_out;
    if (to_bag)
        //设置bag包的输出路径
        bag_out.open(output_bag_file, rosbag::bagmode::Write);
    
    //设置一个由GPS坐标系向相机坐标系转化的矩阵，相机坐标系为右X，下y，上z； 激光雷达和GPS是前x，左y，上z
    //真实位姿是由GPS测得的
    Eigen::Matrix3d R_transform;
    R_transform << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    //旋转矩阵转四元数
    Eigen::Quaterniond q_transform(R_transform);

    std::string line;
    std::size_t line_num = 0;
    //根据launch中获取的publish_delay的值设置帧率，括号的设置的是频率
    ros::Rate r(10.0 / publish_delay);
    //根据ros节点是否存在和读取行数确定是否停止
    while (std::getline(timestamp_file, line) && ros::ok())
    {
        //获取一个时间戳
        float timestamp = stof(line);
        //获取左视和右视的相片各一张
        std::stringstream left_image_path, right_image_path;
        left_image_path << dataset_folder << "sequences/" + sequence_number + "/image_0/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat left_image = cv::imread(left_image_path.str(), cv::IMREAD_GRAYSCALE);
        right_image_path << dataset_folder << "sequences/" + sequence_number + "/image_1/" << std::setfill('0') << std::setw(6) << line_num << ".png";
        cv::Mat right_image = cv::imread(left_image_path.str(), cv::IMREAD_GRAYSCALE);

        //读取一行真实轨迹数据
        std::getline(ground_truth_file, line);
        std::stringstream pose_stream(line);
        std::string s;
        //将一行真实轨迹数据转化为3*4的位姿矩阵，矩阵由一个3*3的旋转矩阵R和一个3*1的位置向量T组成。
        /*
             R1  R2  R3  T1
             R4  R5  R6  T2
             R7  R8  R9  T3
        */
        Eigen::Matrix<double, 3, 4> gt_pose;
        for (std::size_t i = 0; i < 3; ++i)
        {
            for (std::size_t j = 0; j < 4; ++j)
            {
                std::getline(pose_stream, s, ' ');
                gt_pose(i, j) = stof(s);
            }
        }
        //获取3*4位姿矩阵中的旋转矩阵R
        Eigen::Quaterniond q_w_i(gt_pose.topLeftCorner<3, 3>());
        //将旋转矩阵转R化为四元数
        Eigen::Quaterniond q = q_transform * q_w_i;
        //将四元数单位化
        q.normalize();
        //获取3*4位姿矩阵中的位置向量T
        Eigen::Vector3d t = q_transform * gt_pose.topRightCorner<3, 1>();

        //给当前时间戳的点赋予时间戳、位置和姿态信息
        odomGT.header.stamp = ros::Time().fromSec(timestamp);
        odomGT.pose.pose.orientation.x = q.x();
        odomGT.pose.pose.orientation.y = q.y();
        odomGT.pose.pose.orientation.z = q.z();
        odomGT.pose.pose.orientation.w = q.w();
        odomGT.pose.pose.position.x = t(0);
        odomGT.pose.pose.position.y = t(1);
        odomGT.pose.pose.position.z = t(2);
        //利用/odometry_gt 话题发布一次当前时间点的信息
        pubOdomGT.publish(odomGT);

        //获取开始到当前时间点的移动轨迹
        geometry_msgs::PoseStamped poseGT;
        poseGT.header = odomGT.header;
        poseGT.pose = odomGT.pose.pose;
        pathGT.header.stamp = odomGT.header.stamp;
        pathGT.poses.push_back(poseGT);
        //发布一次移动轨迹（相当于更新）
        pubPathGT.publish(pathGT);

        // read lidar point cloud
        std::stringstream lidar_data_path;
        //路径赋值
        lidar_data_path << dataset_folder << "velodyne/sequences/" + sequence_number + "/velodyne/" 
                        << std::setfill('0') << std::setw(6) << line_num << ".bin";
        //读取二进制雷达数据
        std::vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        std::cout << "totally " << lidar_data.size() / 4.0 << " points in this lidar frame \n";
        
        std::vector<Eigen::Vector3d> lidar_points;//存放雷达点的容器
        std::vector<float> lidar_intensities;//存放雷达强度的容器
        pcl::PointCloud<pcl::PointXYZI> laser_cloud;//以pcl中的格式存储雷达信息
        //将输入的雷达数据存成pcl格式
        for (std::size_t i = 0; i < lidar_data.size(); i += 4)
        {
            lidar_points.emplace_back(lidar_data[i], lidar_data[i+1], lidar_data[i+2]);
            lidar_intensities.push_back(lidar_data[i+3]);

            pcl::PointXYZI point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            point.intensity = lidar_data[i + 3];
            laser_cloud.push_back(point);
        }
        //数据转化为ros格式发布
        sensor_msgs::PointCloud2 laser_cloud_msg;
        pcl::toROSMsg(laser_cloud, laser_cloud_msg);
        laser_cloud_msg.header.stamp = ros::Time().fromSec(timestamp);
        laser_cloud_msg.header.frame_id = "camera_init";
        //发布雷达消息
        pub_laser_cloud.publish(laser_cloud_msg);

        //发布图片消息
        sensor_msgs::ImagePtr image_left_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", left_image).toImageMsg();
        sensor_msgs::ImagePtr image_right_msg = cv_bridge::CvImage(laser_cloud_msg.header, "mono8", right_image).toImageMsg();
        pub_image_left.publish(image_left_msg);
        pub_image_right.publish(image_right_msg);

        //将信息写入bag包
        if (to_bag)
        {
            bag_out.write("/image_left", ros::Time::now(), image_left_msg);
            bag_out.write("/image_right", ros::Time::now(), image_right_msg);
            bag_out.write("/velodyne_points", ros::Time::now(), laser_cloud_msg);
            bag_out.write("/path_gt", ros::Time::now(), pathGT);
            bag_out.write("/odometry_gt", ros::Time::now(), odomGT);
        }

        line_num ++;
        r.sleep();
    }
    bag_out.close();
    std::cout << "Done \n";


    return 0;
}
