#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

// armor_detector GPU image type (align topics with video_reader)
#if __has_include(<armor_detector/gpu_image.hpp>) && __has_include(<armor_detector/gpu_image_type_adapter.hpp>)
#include <armor_detector/gpu_image.hpp>
#include <armor_detector/gpu_image_type_adapter.hpp>
#define GPU_CAM_HAS_ARMOR_GPU_IMAGE 1
#else
#define GPU_CAM_HAS_ARMOR_GPU_IMAGE 0
#endif

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>

#ifdef GPU_CAM_HAS_OPENCV_CUDA
  #include <opencv2/core/cuda.hpp>
#endif

#if defined(GPU_CAM_HAS_JETSON_MMA)
#include "gpu_cam_minimal/jetson_mjpeg_decoder.hpp"
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
  publish_mode_ = this->declare_parameter<std::string>("publish_mode", "gpu"); // prefer gpu
  pixel_format_ = this->declare_parameter<std::string>("pixel_format", "mjpeg"); // only mjpeg supported
  use_jetson_hw_decode_ = this->declare_parameter<bool>("use_jetson_hw_decode", false);

  // Publishers aligned with video_reader: /image_gpu + /camera_info
#if defined(GPU_CAM_HAS_OPENCV_CUDA) && GPU_CAM_HAS_ARMOR_GPU_IMAGE
  gpu_image_pub_ = this->create_publisher<armor_detector::GpuImage>("/image_gpu", rclcpp::SensorDataQoS());
  cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera_info", rclcpp::SensorDataQoS());
#else
  cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera_info", rclcpp::SensorDataQoS());
  RCLCPP_WARN(get_logger(), "GPU image publishing not available (no CUDA or armor_detector headers). Node will not publish /image_gpu");
#endif

    // Open camera or init Jetson HW decoder
    if (use_jetson_hw_decode_) {
      initJetsonDecoder();
    } else {
      openCamera();
    }

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
  RCLCPP_INFO(get_logger(), "OpenCV CUDA detected: mode=%s", publish_mode_.c_str());
#else
  RCLCPP_WARN(get_logger(), "OpenCV CUDA not detected at build time: GPU publishing unavailable");
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
    
    // Optional container for GPU frame when using Jetson decoder
#if defined(GPU_CAM_HAS_OPENCV_CUDA)
    cv::cuda::GpuMat gpu_rgb_frame;
#endif
    bool have_frame = false;

    if (use_jetson_hw_decode_) {
#if defined(GPU_CAM_HAS_OPENCV_CUDA)
      if (jetson_decoder_ready_ && jetson_decoder_ && jetson_decoder_->grab_gpu_rgb(gpu_rgb_frame)) {
        // Download to CPU for header size/step if needed (we publish GPU below);
        // also prepare a CPU Mat with same geometry for metadata (avoid extra copies).
        frame.create(gpu_rgb_frame.rows, gpu_rgb_frame.cols, CV_8UC3);
        have_frame = true;
        // We'll not download; we only need cols/rows/step; set step via placeholder
        frame = cv::Mat(gpu_rgb_frame.rows, gpu_rgb_frame.cols, CV_8UC3);
        // Publish using GPU path directly below
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Jetson HW decode failed to get frame");
        have_frame = false;
      }
#else
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "OpenCV CUDA not available; Jetson HW decode disabled");
      have_frame = false;
#endif
    } else {
      if (cap_.read(frame)) {
        have_frame = true;
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to read frame");
        have_frame = false;
      }
    }

    if (!have_frame) return;

    // Prepare camera info
    auto ci = cinfo_mgr_->getCameraInfo();
    ci.header.stamp = this->now();
    ci.header.frame_id = frame_id_;

#if defined(GPU_CAM_HAS_OPENCV_CUDA) && GPU_CAM_HAS_ARMOR_GPU_IMAGE
    if (publish_mode_ == "gpu" && gpu_image_pub_) {
      armor_detector::GpuImage gpu_msg;
      if (use_jetson_hw_decode_) {
        // Jetson path: grab_gpu_rgb already produced GPU RGB frame
        if (gpu_rgb_frame.empty()) {
          // Fallback: convert CPU frame to GPU (until decoder implemented)
          cv::Mat frame_rgb;
          cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
          auto g = std::make_shared<cv::cuda::GpuMat>();
          g->upload(frame_rgb);
          gpu_msg.gpu = std::move(g);
          gpu_msg.width = frame_rgb.cols;
          gpu_msg.height = frame_rgb.rows;
          gpu_msg.step = frame_rgb.step;
        } else {
          gpu_msg.gpu = std::make_shared<cv::cuda::GpuMat>(std::move(gpu_rgb_frame));
          gpu_msg.width = gpu_msg.gpu->cols;
          gpu_msg.height = gpu_msg.gpu->rows;
          gpu_msg.step = static_cast<uint32_t>(gpu_msg.gpu->step);
        }
      } else {
        // OpenCV path: BGR->RGB then upload
        cv::Mat frame_rgb;
        cv::cvtColor(frame, frame_rgb, cv::COLOR_BGR2RGB);
        auto g = std::make_shared<cv::cuda::GpuMat>();
        g->upload(frame_rgb);
        gpu_msg.gpu = std::move(g);
        gpu_msg.width = frame_rgb.cols;
        gpu_msg.height = frame_rgb.rows;
        gpu_msg.step = frame_rgb.step;
      }

      gpu_msg.encoding = "rgb8";
      gpu_msg.header = ci.header;
      gpu_image_pub_->publish(gpu_msg);
      cam_info_pub_->publish(ci);
      return;
    }
#endif

    // If GPU path not available, do nothing or add CPU fallback if needed
    (void)frame; // suppress unused warning when CPU path disabled
  }

private:
  void initJetsonDecoder()
  {
#if defined(GPU_CAM_HAS_OPENCV_CUDA)
    (void)jetson_decoder_ready_;
#endif
#if defined(GPU_CAM_HAS_JETSON_MMA)
    gpu_cam_minimal::JetsonMjpegDecoder::Options opt;
    opt.device = video_device_;
    opt.width = image_width_;
    opt.height = image_height_;
    opt.fps = static_cast<int>(framerate_);
    jetson_decoder_ = std::make_unique<gpu_cam_minimal::JetsonMjpegDecoder>(opt, this->get_logger());
    jetson_decoder_ready_ = jetson_decoder_ && jetson_decoder_->is_ready();
    if (!jetson_decoder_ready_) {
      RCLCPP_WARN(get_logger(), "Jetson MMAPI decoder not ready; falling back to OpenCV V4L2 path");
      openCamera();
      use_jetson_hw_decode_ = false;
    }
#else
    RCLCPP_WARN(get_logger(), "Jetson Multimedia API headers not found; set use_jetson_hw_decode=false");
    openCamera();
    use_jetson_hw_decode_ = false;
#endif
  }
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
  bool use_jetson_hw_decode_ {false};
  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
#if defined(GPU_CAM_HAS_OPENCV_CUDA) && GPU_CAM_HAS_ARMOR_GPU_IMAGE
  rclcpp::Publisher<armor_detector::GpuImage>::SharedPtr gpu_image_pub_;
#endif
  rclcpp::TimerBase::SharedPtr timer_;

  cv::VideoCapture cap_;
#if defined(GPU_CAM_HAS_JETSON_MMA)
  std::unique_ptr<gpu_cam_minimal::JetsonMjpegDecoder> jetson_decoder_;
  bool jetson_decoder_ready_{false};
#endif
  // no persistent GPU buffer; upload per frame after color conversion
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
