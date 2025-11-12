#ifndef VIDEO_READER__VIDEO_READER_NODE_HPP_
#define VIDEO_READER__VIDEO_READER_NODE_HPP_

#include "armor_detector/gpu_image_type_adapter.hpp"  // include first to expose specialization
#include "armor_detector/gpu_image.hpp"

#include <cv_bridge/cv_bridge.h>

#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace video_reader {

class VideoReaderNode : public rclcpp::Node {
public:
    VideoReaderNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

private:
    void timerCallback();

    cv::VideoCapture cap_;
    image_transport::CameraPublisher camera_pub_;
    rclcpp::Publisher<armor_detector::GpuImage>::SharedPtr gpu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::string camera_name_;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
    sensor_msgs::msg::CameraInfo camera_info_msg_;

    // 发布模式：cpu | gpu | both
    std::string publish_mode_ = "gpu";
    
    // frame_id for camera optical frame
    std::string frame_id_;
};

}  // namespace video_reader

#endif  // VIDEO_READER__VIDEO_READER_NODE_HPP_