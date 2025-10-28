#include "video_reader/video_reader_node.hpp"

#include "video_reader/gpu_image_type_adapter.hpp"  // NOLINT: ensure adapter is compiled

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
    camera_pub_ = image_transport::CameraPublisher(); // disable image_transport CPU camera publisher

    // 发布模式：cpu|gpu|both（默认 cpu）
    publish_mode_ = this->declare_parameter<std::string>("publish_mode", "gpu");
    gpu_pub_ = this->create_publisher<GpuImage>("/image_gpu", rclcpp::SensorDataQoS());
    cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera_info", rclcpp::SensorDataQoS());

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

    // 从参数读取 frame_id，默认为 camera_optical_frame
    frame_id_ = this->declare_parameter("frame_id", "camera_optical_frame");

    timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<int>(1000 / fps)),
                                     std::bind(&VideoReaderNode::timerCallback, this));
}

void VideoReaderNode::timerCallback() {
    cv::Mat frame;
    if (cap_.read(frame)) {
        // 统一转换为 RGB
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);

        // 时间戳与帧名
        auto stamp = this->now();

        if (gpu_pub_) {
            // 上传到 GPU 并以零拷贝对象发布
            auto g = std::make_shared<cv::cuda::GpuMat>();
            g->upload(frame);

            auto gpu_msg = std::make_unique<GpuImage>();
            gpu_msg->gpu = std::move(g);
            gpu_msg->encoding = "rgb8";
            gpu_msg->width = frame.cols;
            gpu_msg->height = frame.rows;
            gpu_msg->step = frame.step;
            gpu_msg->header.frame_id = frame_id_;
            gpu_msg->header.stamp = stamp;

            gpu_pub_->publish(std::move(gpu_msg));
            camera_info_msg_.header.frame_id = frame_id_;
            camera_info_msg_.header.stamp = stamp;
            cam_info_pub_->publish(camera_info_msg_);
        }
    } else {
        RCLCPP_INFO(this->get_logger(), "End of video file reached, restarting video");
        cap_.set(cv::CAP_PROP_POS_FRAMES, 0);  // 重置视频到开头
    }
}

}  // namespace video_reader

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(video_reader::VideoReaderNode)