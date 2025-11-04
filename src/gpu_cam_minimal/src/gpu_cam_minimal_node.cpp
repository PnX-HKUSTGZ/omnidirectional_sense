#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#ifdef GPU_CAM_HAS_OPENCV_CUDA
  #include <opencv2/core/cuda.hpp>
#endif

using namespace std::chrono_literals;

class GpuCamMinimalNode : public rclcpp::Node {
public:
  GpuCamMinimalNode()
  : Node("gpu_cam_minimal")
  {
    // Parameters
    device_id_ = this->declare_parameter<int>("device_id", 0);
    width_ = this->declare_parameter<int>("width", 640);
    height_ = this->declare_parameter<int>("height", 480);
    fps_ = this->declare_parameter<int>("fps", 30);
    frame_id_ = this->declare_parameter<std::string>("frame_id", "camera");
    topic_ = this->declare_parameter<std::string>("image_topic", "image_raw");

    pub_ = this->create_publisher<sensor_msgs::msg::Image>(topic_, rclcpp::SensorDataQoS());

    // Open camera
    openCamera();

    // Timer at ~fps
    auto period_ms = (fps_ > 0) ? (1000 / fps_) : 33; // default ~30fps
    timer_ = this->create_wall_timer(std::chrono::milliseconds(period_ms),
      std::bind(&GpuCamMinimalNode::tick, this));
  }

private:
  void openCamera()
  {
    // Prefer V4L2 backend on Linux
    cap_.open(device_id_, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open camera device %d", device_id_);
      throw std::runtime_error("camera open failed");
    }

    if (width_ > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH, width_);
    if (height_ > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    if (fps_ > 0) cap_.set(cv::CAP_PROP_FPS, fps_);

    // Read back actual settings
    width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    fps_ = static_cast<int>(cap_.get(cv::CAP_PROP_FPS));

    RCLCPP_INFO(get_logger(), "Camera opened: %dx%d @ %d fps (device %d)", width_, height_, fps_, device_id_);

#ifdef GPU_CAM_HAS_OPENCV_CUDA
    // Pre-allocate GPU buffer with expected size and type (we expect 8UC3 from OpenCV)
    d_frame_ = cv::cuda::GpuMat(height_, width_, CV_8UC3);
    RCLCPP_INFO(get_logger(), "OpenCV CUDA detected: will upload frames to GPU");
#else
    RCLCPP_WARN(get_logger(), "OpenCV CUDA not detected at build time: publishing without GPU upload");
#endif
  }

  void tick()
  {
    cv::Mat frame;
    if (!cap_.read(frame)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to read frame");
      return;
    }

#ifdef GPU_CAM_HAS_OPENCV_CUDA
    // Upload to GPU (no format conversion)
    if (d_frame_.empty() || d_frame_.rows != frame.rows || d_frame_.cols != frame.cols || d_frame_.type() != frame.type()) {
      d_frame_.release();
      d_frame_ = cv::cuda::GpuMat(frame.size(), frame.type());
    }
    d_frame_.upload(frame);
#endif

    // Publish as sensor_msgs/Image (no conversion; assuming BGR8 from OpenCV)
    auto msg = sensor_msgs::msg::Image();
    msg.header.stamp = this->now();
    msg.header.frame_id = frame_id_;
    msg.height = static_cast<uint32_t>(frame.rows);
    msg.width = static_cast<uint32_t>(frame.cols);
    msg.encoding = "bgr8"; // OpenCV default without manual conversion
    msg.is_bigendian = false;
    msg.step = static_cast<uint32_t>(frame.step);
    size_t size_bytes = frame.total() * frame.elemSize();
    msg.data.resize(size_bytes);
    std::memcpy(msg.data.data(), frame.data, size_bytes);

    pub_->publish(std::move(msg));
  }

private:
  int device_id_;
  int width_;
  int height_;
  int fps_;
  std::string frame_id_;
  std::string topic_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  cv::VideoCapture cap_;
#ifdef GPU_CAM_HAS_OPENCV_CUDA
  cv::cuda::GpuMat d_frame_;
#endif
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<GpuCamMinimalNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("gpu_cam_minimal"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
