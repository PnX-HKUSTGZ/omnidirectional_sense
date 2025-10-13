#ifndef VIDEO_READER__VIDEO_READER_NODE_HPP_
#define VIDEO_READER__VIDEO_READER_NODE_HPP_

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
    rclcpp::TimerBase::SharedPtr timer_;
    std::string camera_name_;
    std::unique_ptr<camera_info_manager::CameraInfoManager> camera_info_manager_;
    sensor_msgs::msg::CameraInfo camera_info_msg_;
};

}  // namespace video_reader

#endif  // VIDEO_READER__VIDEO_READER_NODE_HPP_