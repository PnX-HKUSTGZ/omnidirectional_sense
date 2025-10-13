#include "video_reader/video_reader_node.hpp"

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace video_reader {

VideoReaderNode::VideoReaderNode(const rclcpp::NodeOptions& options)
    : Node("video_reader_node", options) {
    auto video_path =
        ament_index_cpp::get_package_share_directory("video_reader") + "/docs/test_two_car.mp4";
    RCLCPP_INFO(this->get_logger(), "Video path: %s", video_path.c_str());

    cap_.open(video_path);
    if (!cap_.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open video file");
        rclcpp::shutdown();
        return;
    }

    // 获取视频的帧率
    double fps = cap_.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get FPS from video file");
        rclcpp::shutdown();
    }

    bool use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", true);
    auto qos = use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default;
    camera_pub_ = image_transport::create_camera_publisher(this, "image_raw", qos);

    // Load camera info
    camera_name_ = this->declare_parameter("camera_name", "video_camera");
    camera_info_manager_ =
        std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_);
    auto camera_info_url = this->declare_parameter(
        "camera_info_url", "package://video_reader/config/camera_info.yaml");
    if (camera_info_manager_->validateURL(camera_info_url)) {
        camera_info_manager_->loadCameraInfo(camera_info_url);
        camera_info_msg_ = camera_info_manager_->getCameraInfo();
    } else {
        RCLCPP_WARN(this->get_logger(), "Invalid camera info URL: %s", camera_info_url.c_str());
    }

    timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000 / fps)),
                                     std::bind(&VideoReaderNode::timerCallback, this));
}

void VideoReaderNode::timerCallback() {
    cv::Mat frame;
    if (cap_.read(frame)) {
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "rgb8", frame).toImageMsg();
        msg->header.frame_id = "camera_optical_frame";
        msg->header.stamp = this->now();
        camera_info_msg_.header = msg->header;
        camera_pub_.publish(*msg, camera_info_msg_);
    } else {
        RCLCPP_INFO(this->get_logger(), "End of video file reached, restarting video");
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);  // 重置视频到开头
    }
}

}  // namespace video_reader

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(video_reader::VideoReaderNode)