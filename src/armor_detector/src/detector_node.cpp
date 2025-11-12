// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#include <cv_bridge/cv_bridge.h>
#include <rmw/qos_profiles.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_ros/create_timer_ros.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/qos.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// STD
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "armor_detector/detector_node.hpp"
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{
// ==================== 构造函数 ====================
ArmorDetectorNode::ArmorDetectorNode(const rclcpp::NodeOptions & options)
: Node("armor_detector", options)
{
    RCLCPP_INFO(this->get_logger(), "Starting DetectorNode!");

    //设置需要探测的颜色
    declare_parameter("detect_color", RED);

    // 只使用 AI 检测器
    ai_detector_ = initAIDetector();

    //提取相机内参
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
            cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
            cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
            pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
            cam_info_sub_.reset();  //取消订阅
        });

    // 订阅 GPU 图像，使用同进程零拷贝
    img_sub_ = this->create_subscription<armor_detector::GpuImage>(
        "/image_gpu", rclcpp::SensorDataQoS(),
        std::bind(&ArmorDetectorNode::imageCallback, this, std::placeholders::_1));

    // 初始化Armors Publisher
    armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
        "/detector/armors", rclcpp::SensorDataQoS());

    // 初始化Cars Publisher
    cars_pub_ = this->create_publisher<auto_aim_interfaces::msg::Cars>(
        "/detector/cars", rclcpp::SensorDataQoS());

    //tf2
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(
        this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);

    // Debug
    debug_ = this->declare_parameter("debug", false);
    if (debug_) {
        createDebugPublishers();
    }
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ =
        debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter & p) {
            debug_ = p.as_bool();
            debug_ ? createDebugPublishers() : destroyDebugPublishers();
        });
}

// ==================== 初始化功能 ====================


std::unique_ptr<AIDetector> ArmorDetectorNode::initAIDetector()
{
    // 声明AI检测器相关参数
    auto model_path = ament_index_cpp::get_package_share_directory("armor_detector") +
                      this->declare_parameter("ai_model_path", "/model/0526.engine");
    auto device = this->declare_parameter("ai_device", "GPU");  // TensorRT 只支持 GPU
    auto conf_threshold = this->declare_parameter("ai_conf_threshold", 0.65);
    auto nms_threshold = this->declare_parameter("ai_nms_threshold", 0.45);

    // 创建AI检测器实例
    auto ai_detector = std::make_unique<AIDetector>(
        model_path, device, static_cast<float>(conf_threshold), static_cast<float>(nms_threshold));

    RCLCPP_INFO(this->get_logger(), "AI Detector initialized with model: %s", model_path.c_str());

    return ai_detector;
}

// ==================== 核心处理功能 ====================
void ArmorDetectorNode::imageCallback(armor_detector::GpuImage::UniquePtr img_msg)
{
    // 如果当前模式为打符模式，不进行装甲板检测
    if (!enable_) {
        return;
    }

    // 直接使用 GPU 帧进行 AI 检测
    if (!img_msg->gpu || img_msg->gpu->empty()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Empty GpuImage received");
        return;
    }
    std::vector<Armor> armors = ai_detector_->detect(*img_msg->gpu, get_parameter("detect_color").as_int());

    // 提取from base_link to gimbal的坐标系变换
    if (!updateTransform(img_msg->header.frame_id, "base_link", img_msg->header.stamp)) {
        return;
    }

    if (pnp_solver_ == nullptr) return;  //如果pnp解算未初始化

    //初始化消息，包括装甲板信息和调试信息
    armors_msg_.header = img_msg->header;
    armors_msg_.armors.clear();
    if (debug_) {
        armor_marker_.header = text_marker_.header = img_msg->header;
        marker_array_.markers.clear();
        armor_marker_.id = 0;
        text_marker_.id = 0;
    }

    // 进行位姿解算和降自由度优化

    //将装甲板分类存放，无效装甲板的装甲板数标为-1
    // std::string         int           std::vector<int>
    // 装甲板类别      这类装甲板的数目    这类装甲板的编号
    std::map<std::string, std::pair<int, std::vector<int>>> armor_num_map;
    std::vector<Car> cars_detected;
    std::vector<int> valid_armors;  //筛选出能解算，符合先验的有效装甲板，并储存编号
    for (size_t i = 0; i < armors.size(); i++) {
        cv::Mat rvec, tvec;
        bool success =
            pnp_solver_->solvePnP(armors[i], rvec, tvec);  //通过pnp解算获得两个装甲板位姿解
        if (success) {
            if (armor_num_map[armors[i].number].first == -1)
                continue;  // 如果发生了解算失败，这一帧的装甲板数据宁可放弃
            // 装甲板先验可知roll为0，pitch为15度
            // 通过装甲板的类型所决定的先验信息，在两个解中选择较好的那个
            // 同样通过这个先验信息，将原本三自由度的装甲板压到只有yaw一个自由度
            chooseBestPose(armors[i], rvec, tvec);
            armor_num_map[armors[i].number].first++;
            armor_num_map[armors[i].number].second.push_back(i);
            if (armor_num_map[armors[i].number].first > 2) {
                armor_num_map[armors[i].number].first = -1;
                RCLCPP_ERROR(this->get_logger(), "More than 2 armors detected!");
            }
        } else {
            armor_num_map[armors[i].number].first = -1;
            RCLCPP_ERROR(this->get_logger(), "PnP failed!");
        }
    }
    for (auto & armor_num : armor_num_map) {
        if (armor_num.second.first == 2) {
            cars_detected.push_back(Car{(armors[armor_num.second.second[0]].x + 
                                        armors[armor_num.second.second[1]].x) / 2,
                                        (armors[armor_num.second.second[0]].y + 
                                        armors[armor_num.second.second[1]].y) / 2,
                                        (armors[armor_num.second.second[0]].z + 
                                        armors[armor_num.second.second[1]].z) / 2,
                                        armor_num.first});
        }
        else if (armor_num.second.first == 1) {
            cars_detected.push_back(Car{armors[armor_num.second.second[0]].x, 
                                        armors[armor_num.second.second[0]].y,
                                        armors[armor_num.second.second[0]].z,
                                        armor_num.first});
        }
        if (armor_num.second.first != -1) {
            //视为有效装甲板
            valid_armors.insert(
                valid_armors.end(), armor_num.second.second.begin(), armor_num.second.second.end());
            cars_detected.push_back(Car{armors[armor_num.second.second[0]].x, 
                                        armors[armor_num.second.second[0]].y,
                                        armors[armor_num.second.second[0]].z,
                                        armor_num.first});
        }
    }
    //填充车辆信息到消息中，发送给控制节点
    for (auto & car : cars_detected) {
        auto car_msg = auto_aim_interfaces::msg::Car();
        car_msg.x = car.x;
        car_msg.y = car.y;
        car_msg.z = car.z;
        car_msg.type = car.type;
        cars_msg_.cars.emplace_back(car_msg);
    }
    cars_msg_.header = img_msg->header;
    cars_pub_->publish(cars_msg_);
    
    //填充有效装甲板到消息中，发送给调试
    if (debug_) {
        cv::Mat cpu_img;
        auto_aim_interfaces::msg::Armor armor_msg;
        for (auto & index : valid_armors) {
            // Fill basic info
            armor_msg.type = ARMOR_TYPE_STR[static_cast<int>(armors[index].type)];
            armor_msg.number = armors[index].number;

            // Fill pose
            Eigen::Quaterniond eigen_quat(armors[index].r_camera_armor);
            tf2::Quaternion tf2_q(eigen_quat.x(), eigen_quat.y(), eigen_quat.z(), eigen_quat.w());
            armor_msg.pose.orientation = tf2::toMsg(tf2_q);
            armor_msg.pose.position.x = armors[index].t_camera_armor(0);
            armor_msg.pose.position.y = armors[index].t_camera_armor(1);
            armor_msg.pose.position.z = armors[index].t_camera_armor(2);

            // Fill the distance to image center
            armor_msg.distance_to_image_center =
                pnp_solver_->calculateDistanceToCenter(armors[index].center);

            // Fill the classification result
            armors_msg_.armors.emplace_back(armor_msg);

            armor_marker_.id++;
            armor_marker_.scale.y = armors[index].type == ArmorType::SMALL ? 0.135 : 0.23;
            armor_marker_.pose = armor_msg.pose;
            text_marker_.id++;
            text_marker_.pose.position = armor_msg.pose.position;
            text_marker_.pose.position.y -= 0.1;
            text_marker_.text = armors[index].classfication_result;
            marker_array_.markers.emplace_back(armor_marker_);
            marker_array_.markers.emplace_back(text_marker_);
        }
        // draw results（若 cpu_img 为空，则从 GPU 下载一份临时图像供绘制）
        if (img_msg->gpu && !img_msg->gpu->empty()) {
            img_msg->gpu->download(cpu_img);
        }
        if (!cpu_img.empty()) {
            drawResults(img_msg->header, cpu_img, armors);
        }

        // Publishing marker
        publishMarkers();
        // Publishing detected armors
        armors_pub_->publish(armors_msg_);
    }
}

// ==================== 坐标变换和位姿处理 ====================
bool ArmorDetectorNode::updateTransform(
    std::string target_frame, std::string source_frame, rclcpp::Time timestamp)
{
        try {
        auto latest_tf =
            tf2_buffer_->lookupTransform(target_frame, source_frame, tf2::TimePointZero);
        const rclcpp::Time & target_time = timestamp;
        rclcpp::Time latest_time = latest_tf.header.stamp;
        
        // 计算并记录时间差
        double time_diff = (target_time - latest_time).seconds();
        RCLCPP_DEBUG(this->get_logger(), 
            "TF time diff: %.3f ms (target: %.6f, latest: %.6f)", 
            time_diff * 1000, target_time.seconds(), latest_time.seconds());
        
        // 比较时间戳
        geometry_msgs::msg::TransformStamped odom_to_camera_tf;
        if (target_time > latest_time) {
            // 使用最新变换
            odom_to_camera_tf = latest_tf;
            RCLCPP_DEBUG(this->get_logger(), "Using latest TF");
        } else {
            // 查找指定时间的变换
            RCLCPP_DEBUG(this->get_logger(), "Looking up TF at specific time");
            odom_to_camera_tf = tf2_buffer_->lookupTransform(
                target_frame, source_frame, target_time,
                rclcpp::Duration::from_nanoseconds(1000000));
        }
        auto msg_q = odom_to_camera_tf.transform.rotation;
        tf2::Quaternion tf_q;
        tf2::fromMsg(msg_q, tf_q);
        tf2::Matrix3x3 tf2_matrix = tf2::Matrix3x3(tf_q);
        r_odom_to_camera << tf2_matrix.getRow(0)[0], tf2_matrix.getRow(0)[1],
            tf2_matrix.getRow(0)[2], tf2_matrix.getRow(1)[0], tf2_matrix.getRow(1)[1],
            tf2_matrix.getRow(1)[2], tf2_matrix.getRow(2)[0], tf2_matrix.getRow(2)[1],
            tf2_matrix.getRow(2)[2];
        t_odom_to_camera = Eigen::Vector3d(
            odom_to_camera_tf.transform.translation.x, odom_to_camera_tf.transform.translation.y,
            odom_to_camera_tf.transform.translation.z);
        return 1;
    } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(this->get_logger(), 
            "TF lookup failed: [%s] -> [%s] at time %.6f. Error: %s",
            source_frame.c_str(), target_frame.c_str(), 
            timestamp.seconds(), ex.what());
        return 0;
    } catch (const std::exception & ex) {
        RCLCPP_WARN(this->get_logger(), 
            "Exception in lookupTransform [%s] -> [%s]: %s",
            source_frame.c_str(), target_frame.c_str(), ex.what());
        return 0;
    } catch (...) {
        RCLCPP_WARN(this->get_logger(), 
            "Unknown error in lookupTransform: [%s] -> [%s] at time %.6f", 
            source_frame.c_str(), target_frame.c_str(), timestamp.seconds());
        return 0;
    }
}

void ArmorDetectorNode::chooseBestPose(Armor & armor, const cv::Mat & rvec, const cv::Mat & tvec)
{
    //提取云台系欧拉角
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    Eigen::Matrix3d rotation_matrix_eigen;
    cv::cv2eigen(rotation_matrix, rotation_matrix_eigen);
    Eigen::Matrix3d R_camera_to_gimble;
    R_camera_to_gimble << 0, 0, 1, -1, 0, 0, 0, -1, 0;
    Eigen::Vector3d rpy = (R_camera_to_gimble * rotation_matrix_eigen).eulerAngles(0, 1, 2);

    //对于云台系来说，将yaw归一到-pi/2 到 pi/2中后：左侧装甲板yaw角为负，右侧装甲板yaw角为正
    rpy(1) = std::atan2(std::sin(rpy(2)), std::cos(rpy(2)));
    if (abs(rpy(2)) > M_PI / 2) {
        rpy(0) = std::atan2(std::sin(M_PI + rpy(0)), std::cos(M_PI + rpy(0)));  // 旋转roll 180度
        rpy(1) = std::atan2(std::sin(M_PI - rpy(1)), std::cos(M_PI - rpy(1)));  // pitch, 使用补角
        rpy(2) = std::atan2(std::sin(M_PI + rpy(2)), std::cos(M_PI + rpy(2)));  // 旋转yaw 180度
    }
    //前哨站装甲板负倾角
    armor.sign = (armor.left_light.tilt_angle + armor.right_light.tilt_angle) * 0.5 <= 0.0;
    if (armor.number == "outpost") armor.sign = !armor.sign;
    // armor.sign 为0则为右侧装甲板，为1则为左侧装甲板
    if (!armor.sign) {
        rpy = Eigen::Vector3d(rpy(0), rpy(1), abs(rpy(2)));  //右侧
    } else {
        rpy = Eigen::Vector3d(rpy(0), rpy(1), -abs(rpy(2)));  //左侧
    }

    //构造装甲板的旋转平移矩阵
    armor.r_odom_armor = r_odom_to_camera.inverse() * R_camera_to_gimble.inverse() *
                         (Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX()) *
                          Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()) *
                          Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()))
                             .toRotationMatrix();
    armor.t_odom_armor =
        r_odom_to_camera.inverse() *
        (Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)) -
         t_odom_to_camera);
    //设置相机系下的装甲板位姿
    armor.setCameraArmor(r_odom_to_camera, t_odom_to_camera);
    //设置车辆中心位姿
    armor.setCarPose(rpy(2));
    if (abs(rpy(0)) >= 0.26) {
        RCLCPP_WARN(this->get_logger(), "The car is on the slope");
    }
}

// ==================== 可视化和调试功能 ====================
void ArmorDetectorNode::drawResults(
    const std_msgs::msg::Header & header, cv::Mat & img, const std::vector<Armor> & armors)
{
    //计算延迟
    auto final_time = this->now();
    auto latency = (final_time - header.stamp).seconds() * 1000;
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms");
    if (!debug_) {
        return;
    }
    // 只使用 AI 检测器
    ai_detector_->drawResults(img);
    // Show yaw, pitch, roll
    for (const auto & armor : armors) {
        Eigen::Vector3d rpy = armor.r_odom_armor.eulerAngles(0, 1, 2);  //提取欧拉角
        // 归一化
        if (abs(rpy(1)) > M_PI / 2) {
            rpy(0) =
                std::atan2(std::sin(M_PI + rpy(0)), std::cos(M_PI + rpy(0)));  // 旋转roll 180度
            rpy(1) =
                std::atan2(std::sin(M_PI - rpy(1)), std::cos(M_PI - rpy(1)));  // pitch, 使用补角
            rpy(2) = std::atan2(std::sin(M_PI + rpy(2)), std::cos(M_PI + rpy(2)));  // 旋转yaw 180度
        }
        double distance = armor.t_camera_armor.norm();
        cv::putText(
            img, "y: " + std::to_string(int(rpy(2) / CV_PI * 180)) + " deg",
            cv::Point(armor.left_light.bottom.x, armor.left_light.bottom.y + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
        cv::putText(
            img, "d: " + std::to_string(distance) + "m",
            cv::Point(armor.left_light.bottom.x, armor.left_light.bottom.y + 60),
            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);
    }
    // Draw camera center
    cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(
        img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    result_img_pub_.publish(cv_bridge::CvImage(header, "rgb8", img).toImageMsg());
}

void ArmorDetectorNode::createDebugPublishers()
{
    marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("/detector/marker", 10);
    result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
}

void ArmorDetectorNode::destroyDebugPublishers()
{
    marker_pub_.reset();
    result_img_pub_.shutdown();
}

void ArmorDetectorNode::publishMarkers()
{
    using Marker = visualization_msgs::msg::Marker;
    armor_marker_.action = armors_msg_.armors.empty() ? Marker::DELETE : Marker::ADD;
    marker_array_.markers.emplace_back(armor_marker_);
    marker_pub_->publish(marker_array_);
}

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorNode)
