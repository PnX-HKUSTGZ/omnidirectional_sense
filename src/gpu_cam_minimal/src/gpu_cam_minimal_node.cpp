#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>

// 新的 NVDEC 封装
#include "gpu_cam_minimal/nvdec_mjpeg_decoder.hpp"
#include <opencv2/core/cuda.hpp>
#include <armor_detector/gpu_image.hpp>
#include <armor_detector/gpu_image_type_adapter.hpp>

using namespace std::chrono_literals;

class GpuCamMinimalNode : public rclcpp::Node {
public:
  GpuCamMinimalNode()
  : Node("gpu_cam_minimal")
  {
    // Parameters
    camera_name_ = this->declare_parameter<std::string>("camera_name", "cam_0");
    camera_info_url_ = this->declare_parameter<std::string>("camera_info_url", "");
    frame_id_ = this->declare_parameter<std::string>("frame_id", "cam_0");
    framerate_ = this->declare_parameter<double>("framerate", 30.0);
    image_width_ = this->declare_parameter<int>("image_width", 640);
    image_height_ = this->declare_parameter<int>("image_height", 480);
    video_device_ = this->declare_parameter<std::string>("video_device", "/dev/video0");
    publish_mode_ = this->declare_parameter<std::string>("publish_mode", "cpu"); // cpu|gpu
    pixel_format_ = this->declare_parameter<std::string>("pixel_format", "mjpeg"); // only mjpeg supported
    debug_enabled_ = this->declare_parameter<bool>("debug", false);
    control_params_.brightness = this->declare_parameter<int>("brightness", 0);
    control_params_.contrast = this->declare_parameter<int>("contrast", 32);
    control_params_.saturation = this->declare_parameter<int>("saturation", 64);
    control_params_.hue = this->declare_parameter<int>("hue", 0);
    control_params_.white_balance_automatic = this->declare_parameter<bool>("white_balance_automatic", true);
    control_params_.gamma = this->declare_parameter<int>("gamma", 300);
    control_params_.gain = this->declare_parameter<int>("gain", 32);
    control_params_.power_line_frequency = this->declare_parameter<int>("power_line_frequency", 1);
    control_params_.white_balance_temperature = this->declare_parameter<int>("white_balance_temperature", 4600);
    control_params_.sharpness = this->declare_parameter<int>("sharpness", 32);
    control_params_.backlight_compensation = this->declare_parameter<int>("backlight_compensation", 0);
    control_params_.auto_exposure = this->declare_parameter<int>("auto_exposure", 3);
    control_params_.exposure_time_absolute = this->declare_parameter<int>("exposure_time_absolute", 313);

    // Publishers to match usb_cam external topics
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("image_raw", rclcpp::SensorDataQoS());
    cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::SensorDataQoS());

    if (publish_mode_ == "gpu") {
      gpu_image_pub_ = this->create_publisher<armor_detector::GpuImage>("/image_gpu", rclcpp::SensorDataQoS());
      gpu_cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("/camera_info", rclcpp::SensorDataQoS());
    }
    if (debug_enabled_) {
      debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("debug_image_raw", rclcpp::SensorDataQoS());
    }

    apply_camera_controls();
    // Open camera (may choose HW MJPEG decode path on Jetson)
    openCamera();

    // Timer at ~fps
    auto period_ms = (framerate_ > 0.0) ? static_cast<int>(1000.0 / framerate_) : 33; // default ~30fps
    timer_ = this->create_wall_timer(std::chrono::milliseconds(period_ms),
      std::bind(&GpuCamMinimalNode::tick, this));
  }

private:
  void openCamera()
  {
    // Try to use Jetson NVDEC path for MJPEG -> NV12 -> RGB on GPU
    use_hw_mjpeg_ = (publish_mode_ == "gpu" || publish_mode_ == "gpu_hw") &&
                    (pixel_format_ == "mjpeg" || pixel_format_ == "MJPG") &&
                    gpu_cam_minimal::NvdecMjpegDecoder::is_supported();

    if (use_hw_mjpeg_) {
      nvdec_ = std::make_unique<gpu_cam_minimal::NvdecMjpegDecoder>();
      if (!nvdec_->open(video_device_, image_width_, image_height_, framerate_)) {
        RCLCPP_WARN(get_logger(), "Falling back to OpenCV VideoCapture; NVDEC open failed");
        use_hw_mjpeg_ = false;
        nvdec_.reset();
      } else {
        RCLCPP_INFO(get_logger(), "Using Jetson NVDEC MJPEG hardware decode path");
      }
    }

    if (!use_hw_mjpeg_) {
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

      RCLCPP_INFO(get_logger(), "Camera opened (OpenCV): %dx%d @ %.1f fps (%s)", image_width_, image_height_, framerate_, video_device_.c_str());
    }
    if (!use_hw_mjpeg_) {
      // Pre-allocate GPU buffer with expected size and type (we expect 8UC3 from OpenCV)
      d_frame_rgb_ = cv::cuda::GpuMat(image_height_, image_width_, CV_8UC3);
      RCLCPP_INFO(get_logger(), "OpenCV CUDA detected: will upload frames to GPU (mode=%s)", publish_mode_.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "OpenCV CUDA detected: using Jetson NVDEC path for MJPEG");
    }

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

  void apply_camera_controls()
  {
    int fd = ::open(video_device_.c_str(), O_RDWR);
    if (fd < 0) {
      RCLCPP_WARN(get_logger(), "Failed to open %s for V4L2 control setup: %s",
                  video_device_.c_str(), std::strerror(errno));
      return;
    }

    auto set_ctrl = [&](const char * name, int control_id, int value) {
      v4l2_control ctrl{};
      ctrl.id = control_id;
      ctrl.value = value;
      if (::ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
        if (errno == EINVAL || errno == ENOTTY) {
          RCLCPP_DEBUG(get_logger(), "Control %s not supported: %s", name, std::strerror(errno));
        } else {
          RCLCPP_WARN(get_logger(), "Failed to set %s to %d: %s", name, value, std::strerror(errno));
        }
        return false;
      }
      return true;
    };

    set_ctrl("brightness", V4L2_CID_BRIGHTNESS, control_params_.brightness);
    set_ctrl("contrast", V4L2_CID_CONTRAST, control_params_.contrast);
    set_ctrl("saturation", V4L2_CID_SATURATION, control_params_.saturation);
    set_ctrl("hue", V4L2_CID_HUE, control_params_.hue);
    set_ctrl("white_balance_automatic", V4L2_CID_AUTO_WHITE_BALANCE,
             control_params_.white_balance_automatic ? 1 : 0);
    if (!control_params_.white_balance_automatic) {
      set_ctrl("white_balance_temperature", V4L2_CID_WHITE_BALANCE_TEMPERATURE,
               control_params_.white_balance_temperature);
    }
    set_ctrl("gamma", V4L2_CID_GAMMA, control_params_.gamma);
    set_ctrl("gain", V4L2_CID_GAIN, control_params_.gain);
    set_ctrl("power_line_frequency", V4L2_CID_POWER_LINE_FREQUENCY,
             control_params_.power_line_frequency);
    set_ctrl("sharpness", V4L2_CID_SHARPNESS, control_params_.sharpness);
    set_ctrl("backlight_compensation", V4L2_CID_BACKLIGHT_COMPENSATION,
             control_params_.backlight_compensation);
    set_ctrl("auto_exposure", V4L2_CID_EXPOSURE_AUTO, control_params_.auto_exposure);
    if (control_params_.auto_exposure == V4L2_EXPOSURE_MANUAL) {
      set_ctrl("exposure_time_absolute", V4L2_CID_EXPOSURE_ABSOLUTE,
               control_params_.exposure_time_absolute);
    }

    ::close(fd);
  }

  void tick()
  {
    rclcpp::Time timestamp;
    cv::Mat frame_bgr;
    cv::Mat frame_rgb_cpu;
    cv::cuda::GpuMat gpu_rgb;

    if (use_hw_mjpeg_) {
      timestamp = this->now();
      if (!nvdec_ || !nvdec_->read_rgb(gpu_rgb)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "NVDEC read/decode failed");
        return;
      }
    } else {
      timestamp = this->now();
      if (!cap_.read(frame_bgr)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to read frame");
        return;
      }
      cv::cvtColor(frame_bgr, frame_rgb_cpu, cv::COLOR_BGR2RGB);
      if (frame_rgb_cpu.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Converted RGB frame is empty");
        return;
      }
      if (d_frame_rgb_.empty() || d_frame_rgb_.rows != frame_rgb_cpu.rows ||
          d_frame_rgb_.cols != frame_rgb_cpu.cols || d_frame_rgb_.type() != frame_rgb_cpu.type()) {
        d_frame_rgb_.release();
        d_frame_rgb_ = cv::cuda::GpuMat(frame_rgb_cpu.size(), frame_rgb_cpu.type());
      }
      d_frame_rgb_.upload(frame_rgb_cpu);
      gpu_rgb = d_frame_rgb_;
    }

    if (gpu_rgb.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "GPU RGB frame is empty");
      return;
    }

    // Prepare camera info
    auto ci = cinfo_mgr_->getCameraInfo();
    ci.header.stamp = timestamp;
    ci.header.frame_id = frame_id_;

    auto publish_debug_if_requested = [&](const sensor_msgs::msg::CameraInfo & info) {
      if (!debug_enabled_ || !debug_image_pub_ || gpu_rgb.empty()) {
        return;
      }
      cv::Mat debug_cpu;
      gpu_rgb.download(debug_cpu);
      if (debug_cpu.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Debug RGB download failed");
        return;
      }
      sensor_msgs::msg::Image debug_msg;
      debug_msg.header = info.header;
      debug_msg.encoding = "rgb8";
      debug_msg.is_bigendian = false;
      debug_msg.height = static_cast<uint32_t>(debug_cpu.rows);
      debug_msg.width = static_cast<uint32_t>(debug_cpu.cols);
      debug_msg.step = static_cast<uint32_t>(debug_cpu.step);
      size_t debug_size = debug_cpu.total() * debug_cpu.elemSize();
      debug_msg.data.resize(debug_size);
      std::memcpy(debug_msg.data.data(), debug_cpu.data, debug_size);
      debug_image_pub_->publish(debug_msg);
    };

    if ((publish_mode_ == "gpu" || publish_mode_ == "gpu_hw") && gpu_image_pub_) {
      // Publish GPU image with type adapter; no conversions
      armor_detector::GpuImage gpu_msg;
      gpu_msg.header = ci.header;
      gpu_msg.encoding = "rgb8";
      gpu_msg.width = static_cast<uint32_t>(gpu_rgb.cols);
      gpu_msg.height = static_cast<uint32_t>(gpu_rgb.rows);
      gpu_msg.step = static_cast<uint32_t>(gpu_rgb.step);
      gpu_msg.gpu = std::make_shared<cv::cuda::GpuMat>(gpu_rgb);
      gpu_image_pub_->publish(gpu_msg);
      if (gpu_cam_info_pub_) {
        gpu_cam_info_pub_->publish(ci);
      }
      publish_debug_if_requested(ci);
      return;
    }

    // CPU publish path: sensor_msgs/Image (already RGB)
    auto msg = sensor_msgs::msg::Image();
    msg.header = ci.header;
    msg.encoding = "rgb8";
    msg.is_bigendian = false;

    if (!use_hw_mjpeg_) {
      msg.height = static_cast<uint32_t>(frame_rgb_cpu.rows);
      msg.width = static_cast<uint32_t>(frame_rgb_cpu.cols);
      msg.step = static_cast<uint32_t>(frame_rgb_cpu.step);
      size_t size_bytes = frame_rgb_cpu.total() * frame_rgb_cpu.elemSize();
      msg.data.resize(size_bytes);
      std::memcpy(msg.data.data(), frame_rgb_cpu.data, size_bytes);
    } else {
      // Download minimal copy to publish when GPU transport is not available
      cv::Mat rgb_cpu;
      gpu_rgb.download(rgb_cpu);
      msg.height = static_cast<uint32_t>(rgb_cpu.rows);
      msg.width = static_cast<uint32_t>(rgb_cpu.cols);
      msg.step = static_cast<uint32_t>(rgb_cpu.step);
      size_t size_bytes = rgb_cpu.total() * rgb_cpu.elemSize();
      msg.data.resize(size_bytes);
      std::memcpy(msg.data.data(), rgb_cpu.data, size_bytes);
    }

    image_pub_->publish(msg);
    cam_info_pub_->publish(ci);
    publish_debug_if_requested(ci);
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
  bool debug_enabled_ {false};
  struct CameraControlParams {
    int brightness {0};
    int contrast {32};
    int saturation {64};
    int hue {0};
    bool white_balance_automatic {true};
    int gamma {300};
    int gain {32};
    int power_line_frequency {1};
    int white_balance_temperature {4600};
    int sharpness {32};
    int backlight_compensation {0};
    int auto_exposure {V4L2_EXPOSURE_APERTURE_PRIORITY};
    int exposure_time_absolute {313};
  } control_params_;
  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_pub_;
  // GPU publishers only when armor_detector GPU message is available
  rclcpp::Publisher<armor_detector::GpuImage>::SharedPtr gpu_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr gpu_cam_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  cv::VideoCapture cap_;
  cv::cuda::GpuMat d_frame_rgb_;
  std::unique_ptr<camera_info_manager::CameraInfoManager> cinfo_mgr_;

  bool use_hw_mjpeg_ {false};

  // NVDEC 解码器实例（按需）
  std::unique_ptr<gpu_cam_minimal::NvdecMjpegDecoder> nvdec_;

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
