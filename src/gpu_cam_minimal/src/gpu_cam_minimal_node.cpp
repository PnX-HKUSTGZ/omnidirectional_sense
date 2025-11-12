#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#ifdef GPU_CAM_HAS_OPENCV_CUDA
  #include <opencv2/core/cuda.hpp>
  // GPU message type adapter from video_reader (if available in workspace)
  #if __has_include(<video_reader/gpu_image.hpp>) && __has_include(<video_reader/gpu_image_type_adapter.hpp>)
    #include <video_reader/gpu_image.hpp>
    #include <video_reader/gpu_image_type_adapter.hpp>
    #define GPU_CAM_HAS_VIDEO_READER 1
  #else
    #define GPU_CAM_HAS_VIDEO_READER 0
  #endif
#endif

using namespace std::chrono_literals;

class GpuCamMinimalNode : public rclcpp::Node {
public:
  GpuCamMinimalNode()
  : Node("gpu_cam_minimal")
  {
    // Parameters
    camera_name_ = this->declare_parameter<std::string>("camera_name", "default_cam");
    camera_info_url_ = this->declare_parameter<std::string>("camera_info_url", "");
    frame_id_ = this->declare_parameter<std::string>("frame_id", "default_cam");
    framerate_ = this->declare_parameter<double>("framerate", 30.0);
    image_width_ = this->declare_parameter<int>("image_width", 640);
    image_height_ = this->declare_parameter<int>("image_height", 480);
    video_device_ = this->declare_parameter<std::string>("video_device", "/dev/video0");
  publish_mode_ = this->declare_parameter<std::string>("publish_mode", "cpu"); // cpu|gpu
  pixel_format_ = this->declare_parameter<std::string>("pixel_format", "mjpeg"); // only mjpeg supported

    // Publishers to match usb_cam external topics
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", rclcpp::SensorDataQoS());
    cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS());

#if defined(GPU_CAM_HAS_OPENCV_CUDA) && GPU_CAM_HAS_VIDEO_READER
    if (publish_mode_ == "gpu") {
      gpu_image_pub_ = this->create_publisher<video_reader::GpuImage>("/image_gpu", rclcpp::SensorDataQoS());
      gpu_cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera_info", rclcpp::SensorDataQoS());
    }
#endif

    // Open camera
    openCamera();

    // Timer at ~fps
    auto period_ms = (framerate_ > 0.0) ? static_cast<int>(1000.0 / framerate_) : 33; // default ~30fps
    timer_ = this->create_wall_timer(std::chrono::milliseconds(period_ms),
      std::bind(&GpuCamMinimalNode::tick, this));
  }

private:
  void openCamera()
  {
    // Map video_device to numeric id if possible (e.g., /dev/video0 -> 0)
    int device_id = parse_device_id(video_device_);
    // Prefer V4L2 backend on Linux
    cap_.open(device_id, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open camera device %s (id=%d)", video_device_.c_str(), device_id);
      throw std::runtime_error("camera open failed");
    }

    // Enforce MJPG if requested (best-effort)
    if (pixel_format_ == "mjpeg" || pixel_format_ == "MJPG") {
      bool ok = cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
      RCLCPP_INFO(get_logger(), "Request MJPG pixel format: %s", ok ? "OK" : "Not supported by backend");
    } else {
      RCLCPP_WARN(get_logger(), "Only 'mjpeg' pixel_format is supported by gpu_cam_minimal; got '%s'", pixel_format_.c_str());
    }

    if (image_width_ > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
    if (image_height_ > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
    if (framerate_ > 0.0) cap_.set(cv::CAP_PROP_FPS, framerate_);

    // Read back actual settings
    image_width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    image_height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    framerate_ = cap_.get(cv::CAP_PROP_FPS);

    RCLCPP_INFO(get_logger(), "Camera opened: %dx%d @ %.1f fps (%s)", image_width_, image_height_, framerate_, video_device_.c_str());

#ifdef GPU_CAM_HAS_OPENCV_CUDA
    // Pre-allocate GPU buffer with expected size and type (we expect 8UC3 from OpenCV)
    d_frame_ = cv::cuda::GpuMat(image_height_, image_width_, CV_8UC3);
    RCLCPP_INFO(get_logger(), "OpenCV CUDA detected: will upload frames to GPU (mode=%s)", publish_mode_.c_str());
#else
    RCLCPP_WARN(get_logger(), "OpenCV CUDA not detected at build time: publishing without GPU upload");
#endif

    // CameraInfo manager (minimal defaults if no URL)
    cinfo_mgr_ = std::make_unique<camera_info_manager::CameraInfoManager>(this, camera_name_, camera_info_url_);
    sensor_msgs::msg::CameraInfo ci;
    if (!cinfo_mgr_->isCalibrated()) {
      ci.header.frame_id = frame_id_;
      ci.width = static_cast<uint32_t>(image_width_);
      ci.height = static_cast<uint32_t>(image_height_);
      cinfo_mgr_->setCameraInfo(ci);
    }
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

    // Prepare camera info
    auto ci = cinfo_mgr_->getCameraInfo();
    ci.header.stamp = this->now();
    ci.header.frame_id = frame_id_;

#if defined(GPU_CAM_HAS_OPENCV_CUDA) && GPU_CAM_HAS_VIDEO_READER
    if (publish_mode_ == "gpu" && gpu_image_pub_) {
      // Publish GPU image with type adapter; no conversions
      video_reader::GpuImage gpu_msg;
      gpu_msg.header = ci.header;
      gpu_msg.encoding = "bgr8";
      gpu_msg.width = static_cast<uint32_t>(frame.cols);
      gpu_msg.height = static_cast<uint32_t>(frame.rows);
      gpu_msg.step = static_cast<uint32_t>(frame.step);
      gpu_msg.gpu = std::make_shared<cv::cuda::GpuMat>(d_frame_);
      gpu_image_pub_->publish(gpu_msg);
      if (gpu_cam_info_pub_) {
        gpu_cam_info_pub_->publish(ci);
      }
      return;
    }
#endif

    // CPU publish path: sensor_msgs/Image (no conversion; assuming BGR8)
    auto msg = sensor_msgs::msg::Image();
    msg.header = ci.header;
    msg.height = static_cast<uint32_t>(frame.rows);
    msg.width = static_cast<uint32_t>(frame.cols);
    msg.encoding = "bgr8"; // OpenCV default without manual conversion
    msg.is_bigendian = false;
    msg.step = static_cast<uint32_t>(frame.step);
    size_t size_bytes = frame.total() * frame.elemSize();
    msg.data.resize(size_bytes);
    std::memcpy(msg.data.data(), frame.data, size_bytes);

    image_pub_->publish(msg);
    cam_info_pub_->publish(ci);
  }

private:
  // usb_cam-aligned params
  std::string camera_name_;
  std::string camera_info_url_;
  std::string frame_id_;
  double framerate_;
  int image_width_;
  int image_height_;
  std::string video_device_;
  std::string publish_mode_;
  std::string pixel_format_;
  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
#if defined(GPU_CAM_HAS_OPENCV_CUDA)
  // Only valid if GPU_CAM_HAS_VIDEO_READER is true, but keep declaration guarded simply by CUDA
  rclcpp::Publisher<video_reader::GpuImage>::SharedPtr gpu_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr gpu_cam_info_pub_;
#endif
  rclcpp::TimerBase::SharedPtr timer_;

  cv::VideoCapture cap_;
#ifdef GPU_CAM_HAS_OPENCV_CUDA
  cv::cuda::GpuMat d_frame_;
#endif
  std::unique_ptr<camera_info_manager::CameraInfoManager> cinfo_mgr_;

  static int parse_device_id(const std::string & dev)
  {
    // try to extract trailing digits
    int id = 0;
    try {
      size_t pos = dev.find_last_not_of("0123456789");
      if (pos != std::string::npos && pos + 1 < dev.size()) {
        id = std::stoi(dev.substr(pos + 1));
      }
    } catch (...) {
      id = 0;
    }
    return id;
  }
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
