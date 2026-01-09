#include "gpu_cam_minimal/gpu_cam_minimal_node.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <algorithm>
#include <armor_detector/gpu_image_type_adapter.hpp>
#include <cerrno>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <thread>

using namespace std::chrono_literals;

GpuCamMinimalNode::GpuCamMinimalNode(const rclcpp::NodeOptions & options)
: Node("gpu_cam_minimal", options)
{
    // Parameters
    camera_name_ = this->declare_parameter<std::string>("camera_name", "cam_0");
    camera_info_url_ = this->declare_parameter<std::string>("camera_info_url", "");
    frame_id_ = this->declare_parameter<std::string>("frame_id", "cam_0");
    framerate_ = this->declare_parameter<double>("framerate", 30.0);
    image_width_ = this->declare_parameter<int>("image_width", 640);
    image_height_ = this->declare_parameter<int>("image_height", 480);
    video_device_ = this->declare_parameter<std::string>("video_device", "/dev/video0");
    publish_mode_ = "gpu";    // fixed mode
    pixel_format_ = "mjpeg";  // pipeline assumes MJPEG input
    debug_enabled_ = this->declare_parameter<bool>("debug", false);
    use_v4l2_buffer_timestamps_ =
        this->declare_parameter<bool>("use_v4l2_buffer_timestamps", true);
    double timestamp_offset_sec = this->declare_parameter<double>("timestamp_offset", 0.0);
    timestamp_offset_ = rclcpp::Duration::from_seconds(timestamp_offset_sec);
    tsc_offset_ = this->declare_parameter<int64_t>("tsc_offset_ns", 0);
    nvdec_v4l2_buffer_count_ = this->declare_parameter<int>("nvdec_v4l2_buffer_count", 4);
    nvdec_capture_buffer_padding_ = this->declare_parameter<int>("nvdec_capture_buffer_padding", 2);
    nvdec_drop_late_frames_ = this->declare_parameter<bool>("nvdec_drop_late_frames", true);
    control_params_.brightness = this->declare_parameter<int>("brightness", 0);
    control_params_.contrast = this->declare_parameter<int>("contrast", 32);
    control_params_.saturation = this->declare_parameter<int>("saturation", 64);
    control_params_.hue = this->declare_parameter<int>("hue", 0);
    control_params_.white_balance_automatic =
        this->declare_parameter<bool>("white_balance_automatic", true);
    control_params_.gamma = this->declare_parameter<int>("gamma", 300);
    control_params_.gain = this->declare_parameter<int>("gain", 32);
    control_params_.power_line_frequency = this->declare_parameter<int>("power_line_frequency", 1);
    control_params_.white_balance_temperature =
        this->declare_parameter<int>("white_balance_temperature", 4600);
    control_params_.sharpness = this->declare_parameter<int>("sharpness", 32);
    control_params_.backlight_compensation =
        this->declare_parameter<int>("backlight_compensation", 0);
    control_params_.auto_exposure = this->declare_parameter<int>("auto_exposure", 3);
    control_params_.exposure_time_absolute =
        this->declare_parameter<int>("exposure_time_absolute", 313);
    cuda_device_id_ = this->declare_parameter<int>("cuda_device_id", 0);

    setTSCOffset();

    initializeCudaDevice();

    // Publishers to match usb_cam external topics
    image_pub_ =
        this->create_publisher<sensor_msgs::msg::Image>("image_raw", rclcpp::SensorDataQoS());
    cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
        "camera_info", rclcpp::SensorDataQoS());

    if (publish_mode_ == "gpu") {
        gpu_image_pub_ =
            this->create_publisher<armor_detector::GpuImage>("/image_gpu", rclcpp::SensorDataQoS());
        gpu_cam_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>(
            "/camera_info", rclcpp::SensorDataQoS());
    }
    if (debug_enabled_) {
        debug_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "debug_image_raw", rclcpp::SensorDataQoS());
    }

    apply_camera_controls();
    // Open camera (may choose HW MJPEG decode path on Jetson)
    openCamera();

    if (use_hw_mjpeg_) {
        startNvdecCaptureLoop();
    } else {
        startCpuTimer();
    }
}

GpuCamMinimalNode::~GpuCamMinimalNode() { stopNvdecCaptureLoop(); }

void GpuCamMinimalNode::initializeCudaDevice()
{
    auto err = cudaSetDevice(cuda_device_id_);
    if (err != cudaSuccess) {
        RCLCPP_FATAL(
            get_logger(), "cudaSetDevice(%d) failed: %s", cuda_device_id_, cudaGetErrorString(err));
        throw std::runtime_error("Failed to initialize CUDA device context");
    }
}

void GpuCamMinimalNode::openCamera()
{
    // Try to use Jetson NVDEC path for MJPEG -> NV12 -> RGB on GPU
    use_hw_mjpeg_ = (publish_mode_ == "gpu" || publish_mode_ == "gpu_hw") &&
                    (pixel_format_ == "mjpeg" || pixel_format_ == "MJPG") &&
                    gpu_cam_minimal::NvdecMjpegDecoder::is_supported();

    if (use_hw_mjpeg_) {
        nvdec_ = std::make_unique<gpu_cam_minimal::NvdecMjpegDecoder>();
        gpu_cam_minimal::NvdecMjpegDecoder::Config config;
        config.v4l2_buffer_count = static_cast<uint32_t>(std::max(4, nvdec_v4l2_buffer_count_));
        config.capture_buffer_padding =
            static_cast<uint32_t>(std::max(2, nvdec_capture_buffer_padding_));
        config.drop_late_frames = nvdec_drop_late_frames_;
        nvdec_->set_config(config);
        if (!nvdec_->open(video_device_, image_width_, image_height_, framerate_)) {
            RCLCPP_WARN(get_logger(), "Falling back to OpenCV VideoCapture; NVDEC open failed");
            use_hw_mjpeg_ = false;
            nvdec_.reset();
        } else {
            RCLCPP_INFO(get_logger(), "Using Jetson NVDEC MJPEG hardware decode path");
        }
    }

    if (!use_hw_mjpeg_) {
        if (!openCpuCapture()) {
            throw std::runtime_error("camera open failed");
        }
    } else {
        RCLCPP_INFO(get_logger(), "OpenCV CUDA detected: using Jetson NVDEC path for MJPEG");
    }

    // CameraInfo manager (minimal defaults if no URL)
    cinfo_mgr_ = std::make_unique<camera_info_manager::CameraInfoManager>(
        this, camera_name_, camera_info_url_);
    sensor_msgs::msg::CameraInfo ci;
    if (!cinfo_mgr_->isCalibrated()) {
        ci.header.frame_id = frame_id_;
        ci.width = static_cast<uint32_t>(image_width_);
        ci.height = static_cast<uint32_t>(image_height_);
        cinfo_mgr_->setCameraInfo(ci);
    }
}

void GpuCamMinimalNode::apply_camera_controls()
{
    int fd = ::open(video_device_.c_str(), O_RDWR);
    if (fd < 0) {
        RCLCPP_WARN(
            get_logger(), "Failed to open %s for V4L2 control setup: %s", video_device_.c_str(),
            std::strerror(errno));
        return;
    }

    auto set_ctrl = [&](const char * name, int control_id, int value) {
        v4l2_control ctrl{};
        ctrl.id = control_id;
        ctrl.value = value;
        if (::ioctl(fd, VIDIOC_S_CTRL, &ctrl) < 0) {
            if (errno == EINVAL || errno == ENOTTY) {
                RCLCPP_DEBUG(
                    get_logger(), "Control %s not supported: %s", name, std::strerror(errno));
            } else {
                RCLCPP_WARN(
                    get_logger(), "Failed to set %s to %d: %s", name, value, std::strerror(errno));
            }
            return false;
        }
        return true;
    };

    set_ctrl("brightness", V4L2_CID_BRIGHTNESS, control_params_.brightness);
    set_ctrl("contrast", V4L2_CID_CONTRAST, control_params_.contrast);
    set_ctrl("saturation", V4L2_CID_SATURATION, control_params_.saturation);
    set_ctrl("hue", V4L2_CID_HUE, control_params_.hue);
    set_ctrl(
        "white_balance_automatic", V4L2_CID_AUTO_WHITE_BALANCE,
        control_params_.white_balance_automatic ? 1 : 0);
    if (!control_params_.white_balance_automatic) {
        set_ctrl(
            "white_balance_temperature", V4L2_CID_WHITE_BALANCE_TEMPERATURE,
            control_params_.white_balance_temperature);
    }
    set_ctrl("gamma", V4L2_CID_GAMMA, control_params_.gamma);
    set_ctrl("gain", V4L2_CID_GAIN, control_params_.gain);
    set_ctrl(
        "power_line_frequency", V4L2_CID_POWER_LINE_FREQUENCY,
        control_params_.power_line_frequency);
    set_ctrl("sharpness", V4L2_CID_SHARPNESS, control_params_.sharpness);
    set_ctrl(
        "backlight_compensation", V4L2_CID_BACKLIGHT_COMPENSATION,
        control_params_.backlight_compensation);
    set_ctrl("auto_exposure", V4L2_CID_EXPOSURE_AUTO, control_params_.auto_exposure);
    if (control_params_.auto_exposure == V4L2_EXPOSURE_MANUAL) {
        set_ctrl(
            "exposure_time_absolute", V4L2_CID_EXPOSURE_ABSOLUTE,
            control_params_.exposure_time_absolute);
    }

    ::close(fd);
}

int64_t GpuCamMinimalNode::timeval_to_ns(const timeval & tv)
{
    constexpr int64_t kSecToNs = 1000000000LL;
    constexpr int64_t kUsecToNs = 1000LL;
    return static_cast<int64_t>(tv.tv_sec) * kSecToNs +
           static_cast<int64_t>(tv.tv_usec) * kUsecToNs;
}

rclcpp::Time GpuCamMinimalNode::convert_v4l2_timestamp(const timeval & tv, bool is_monotonic)
{
    if (!use_v4l2_buffer_timestamps_) {
        return system_now_with_offset();
    }

    (void)is_monotonic;

    int64_t ts_ns = timeval_to_ns(tv);
    if (ts_ns <= 0) {
        return system_now_with_offset();
    }

    int64_t stamp_ns = ts_ns + getTimeOffset() - tsc_offset_;

    if (stamp_ns < 0) {
        stamp_ns = 0;
    }

    rclcpp::Time stamp(stamp_ns, get_clock()->get_clock_type());
    stamp = stamp + timestamp_offset_;
    return stamp;
}

int64_t GpuCamMinimalNode::getTimeOffset() const
{
    timespec system_sample{};
    timespec monotonic_sample{};
    if (clock_gettime(CLOCK_REALTIME, &system_sample) != 0 ||
        clock_gettime(CLOCK_MONOTONIC, &monotonic_sample) != 0) {
        return 0;
    }

    constexpr int64_t kSecToNs = 1000000000LL;
    int64_t system_ns = static_cast<int64_t>(system_sample.tv_sec) * kSecToNs +
                        static_cast<int64_t>(system_sample.tv_nsec);
    int64_t monotonic_ns = static_cast<int64_t>(monotonic_sample.tv_sec) * kSecToNs +
                           static_cast<int64_t>(monotonic_sample.tv_nsec);
    return system_ns - monotonic_ns;
}

void GpuCamMinimalNode::setTSCOffset()
{
#if defined(__aarch64__) || defined(__arm__)
    if (tsc_offset_ != 0) {
        RCLCPP_INFO(
            get_logger(), "Using user provided tsc_offset_ns=%lld", static_cast<long long>(tsc_offset_));
        return;
    }

    const char * env_offset = std::getenv("TSC_OFFSET_NS");
    if (env_offset != nullptr) {
        char * end = nullptr;
        errno = 0;
        long long parsed = std::strtoll(env_offset, &end, 10);
        if (end != env_offset && errno == 0) {
            tsc_offset_ = static_cast<int64_t>(parsed);
            RCLCPP_INFO(
                get_logger(), "Using TSC_OFFSET_NS env override: %lld ns",
                static_cast<long long>(tsc_offset_));
            return;
        }
    }

    // Default to zero; can be overridden via parameter or env when Jetson kernels expose TSC skew.
    tsc_offset_ = 0;
#else
    (void)tsc_offset_;
#endif
}

rclcpp::Time GpuCamMinimalNode::system_now_with_offset() const
{
    return this->now() + timestamp_offset_;
}

void GpuCamMinimalNode::tick()
{
    if (use_hw_mjpeg_) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "tick() invoked while NVDEC mode active; ignoring timer callback");
        return;
    }

    rclcpp::Time timestamp = system_now_with_offset();
    cv::Mat frame_bgr;
    cv::Mat frame_rgb_cpu;

    nvdec_failure_count_ = 0;
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

    publishFrame(d_frame_rgb_, &frame_rgb_cpu, timestamp);
}

void GpuCamMinimalNode::startCpuTimer()
{
    if (timer_) {
        return;
    }
    auto period_ms = (framerate_ > 0.0) ? static_cast<int>(1000.0 / framerate_) : 33;
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(period_ms), std::bind(&GpuCamMinimalNode::tick, this));
}

void GpuCamMinimalNode::startNvdecCaptureLoop()
{
    if (nvdec_thread_running_.load()) {
        return;
    }
    nvdec_thread_stop_.store(false);
    nvdec_thread_running_.store(true);
    nvdec_thread_ = std::thread([this]() { nvdecCaptureLoop(); });
}

void GpuCamMinimalNode::stopNvdecCaptureLoop()
{
    nvdec_thread_stop_.store(true);
    if (nvdec_thread_.joinable()) {
        nvdec_thread_.join();
    }
    nvdec_thread_running_.store(false);
}

void GpuCamMinimalNode::nvdecCaptureLoop()
{
    cv::cuda::GpuMat gpu_rgb;
    while (rclcpp::ok() && !nvdec_thread_stop_.load()) {
        if (!use_hw_mjpeg_ || !nvdec_) {
            break;
        }

        rclcpp::Time timestamp;
        if (readNvdecFrame(gpu_rgb, timestamp)) {
            nvdec_failure_count_ = 0;
            publishFrame(gpu_rgb, nullptr, timestamp);
            continue;
        }

        int err = errno;
        if (err == EAGAIN || err == EWOULDBLOCK) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        handleNvdecFailure("read/decode failed (errno=" + std::string(strerror(err)) + ")");
        if (!use_hw_mjpeg_) {
            break;
        }
    }
    nvdec_thread_running_.store(false);
}

bool GpuCamMinimalNode::readNvdecFrame(cv::cuda::GpuMat & gpu_rgb, rclcpp::Time & timestamp)
{
    if (!nvdec_) {
        errno = ENODEV;
        return false;
    }
    struct timeval capture_tv
    {
    };
    bool ts_monotonic = false;
    if (!nvdec_->read_rgb(gpu_rgb, &capture_tv, &ts_monotonic)) {
        return false;
    }
    if (gpu_rgb.empty()) {
        errno = EIO;
        return false;
    }
    timestamp = convert_v4l2_timestamp(capture_tv, ts_monotonic);
    return true;
}

void GpuCamMinimalNode::publishFrame(
    const cv::cuda::GpuMat & gpu_rgb, const cv::Mat * cpu_rgb, const rclcpp::Time & timestamp)
{
    if (gpu_rgb.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "GPU RGB frame is empty");
        return;
    }

    auto ci = cinfo_mgr_->getCameraInfo();
    ci.header.stamp = timestamp;
    ci.header.frame_id = frame_id_;

    const bool gpu_publish =
        (publish_mode_ == "gpu" || publish_mode_ == "gpu_hw") && gpu_image_pub_;

    if (gpu_publish) {
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
    } else {
        if (!image_pub_ || !cam_info_pub_) {
            return;
        }
        cv::Mat rgb_cpu_storage;
        const cv::Mat * src_cpu = cpu_rgb;
        if (!src_cpu) {
            gpu_rgb.download(rgb_cpu_storage);
            src_cpu = &rgb_cpu_storage;
        }
        if (!src_cpu || src_cpu->empty()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000, "CPU RGB frame unavailable for publish");
            return;
        }

        sensor_msgs::msg::Image msg;
        msg.header = ci.header;
        msg.encoding = "rgb8";
        msg.is_bigendian = false;
        msg.height = static_cast<uint32_t>(src_cpu->rows);
        msg.width = static_cast<uint32_t>(src_cpu->cols);
        msg.step = static_cast<uint32_t>(src_cpu->step);
        const size_t size_bytes = src_cpu->total() * src_cpu->elemSize();
        msg.data.resize(size_bytes);
        std::memcpy(msg.data.data(), src_cpu->data, size_bytes);

        image_pub_->publish(msg);
        cam_info_pub_->publish(ci);
    }

    const auto publish_time = this->now();
    updateDebugStats(timestamp, publish_time);
    publishDebugImage(ci, gpu_rgb, cpu_rgb);
}

void GpuCamMinimalNode::publishDebugImage(
    const sensor_msgs::msg::CameraInfo & info, const cv::cuda::GpuMat & gpu_rgb,
    const cv::Mat * cpu_rgb)
{
    if (!debug_enabled_ || !debug_image_pub_ || gpu_rgb.empty()) {
        return;
    }

    cv::Mat debug_cpu_storage;
    const cv::Mat * src_cpu = cpu_rgb;
    if (!src_cpu) {
        gpu_rgb.download(debug_cpu_storage);
        src_cpu = &debug_cpu_storage;
    }
    if (!src_cpu || src_cpu->empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Debug RGB download failed");
        return;
    }

    sensor_msgs::msg::Image debug_msg;
    debug_msg.header = info.header;
    debug_msg.encoding = "rgb8";
    debug_msg.is_bigendian = false;
    debug_msg.height = static_cast<uint32_t>(src_cpu->rows);
    debug_msg.width = static_cast<uint32_t>(src_cpu->cols);
    debug_msg.step = static_cast<uint32_t>(src_cpu->step);
    const size_t debug_size = src_cpu->total() * src_cpu->elemSize();
    debug_msg.data.resize(debug_size);
    std::memcpy(debug_msg.data.data(), src_cpu->data, debug_size);
    debug_image_pub_->publish(debug_msg);
}

void GpuCamMinimalNode::updateDebugStats(
    const rclcpp::Time & capture_ts, const rclcpp::Time & publish_ts)
{
    if (!debug_enabled_) {
        return;
    }

    const double latency_ms = static_cast<double>((publish_ts - capture_ts).nanoseconds()) / 1e6;
    std::lock_guard<std::mutex> lock(debug_stats_mutex_);
    if (debug_window_start_.nanoseconds() == 0) {
        debug_window_start_ = publish_ts;
        debug_window_frames_ = 0;
        debug_window_latency_ms_ = 0.0;
    }

    debug_window_frames_ += 1;
    debug_window_latency_ms_ += latency_ms;

    const double window_ms =
        static_cast<double>((publish_ts - debug_window_start_).nanoseconds()) / 1e6;
    if (window_ms >= 1000.0) {
        const double avg_latency =
            debug_window_frames_ > 0 ? debug_window_latency_ms_ / debug_window_frames_ : 0.0;
        const double fps = (window_ms > 0.0) ? (debug_window_frames_ * 1000.0 / window_ms) : 0.0;
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "debug stats: fps=%.2f avg_latency=%.2f ms over %.0f ms", fps, avg_latency, window_ms);
        debug_window_start_ = publish_ts;
        debug_window_frames_ = 0;
        debug_window_latency_ms_ = 0.0;
    }
}

int GpuCamMinimalNode::parse_device_id(const std::string & dev)
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

bool GpuCamMinimalNode::openCpuCapture()
{
    int device_id = parse_device_id(video_device_);
    cap_.release();
    cap_.open(device_id, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
        RCLCPP_ERROR(
            get_logger(), "Failed to open camera device %s (id=%d)", video_device_.c_str(),
            device_id);
        return false;
    }

    if (pixel_format_ == "mjpeg" || pixel_format_ == "MJPG") {
        bool ok = cap_.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        RCLCPP_INFO(
            get_logger(), "Request MJPG pixel format: %s", ok ? "OK" : "Not supported by backend");
    } else {
        RCLCPP_WARN(
            get_logger(),
            "Only 'mjpeg' pixel_format is supported by "
            "gpu_cam_minimal; got '%s'",
            pixel_format_.c_str());
    }

    if (image_width_ > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
    if (image_height_ > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
    if (framerate_ > 0.0) cap_.set(cv::CAP_PROP_FPS, framerate_);

    image_width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    image_height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    framerate_ = cap_.get(cv::CAP_PROP_FPS);

    d_frame_rgb_.release();
    d_frame_rgb_ = cv::cuda::GpuMat(image_height_, image_width_, CV_8UC3);

    RCLCPP_INFO(
        get_logger(), "Camera opened (OpenCV): %dx%d @ %.1f fps (%s)", image_width_, image_height_,
        framerate_, video_device_.c_str());
    RCLCPP_INFO(
        get_logger(), "OpenCV CUDA detected: will upload frames to GPU (mode=%s)",
        publish_mode_.c_str());
    return true;
}

void GpuCamMinimalNode::handleNvdecFailure(const std::string & reason)
{
    ++nvdec_failure_count_;
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "NVDEC failure (%d/%d): %s", nvdec_failure_count_,
        kNvdecFailureThreshold, reason.c_str());
    if (nvdec_failure_count_ >= kNvdecFailureThreshold && kNvdecFailureThreshold != -1) {
        fallbackToCpuCapture("Exceeded NVDEC failure threshold. Last error: " + reason);
    }
}

void GpuCamMinimalNode::fallbackToCpuCapture(const std::string & reason)
{
    if (!use_hw_mjpeg_) {
        return;
    }

    stopNvdecCaptureLoop();
    RCLCPP_ERROR(
        get_logger(), "Disabling NVDEC path: %s. Falling back to OpenCV pipeline.", reason.c_str());
    if (nvdec_) {
        nvdec_->close_decoder();
        nvdec_.reset();
    }
    use_hw_mjpeg_ = false;
    nvdec_failure_count_ = 0;

    if (!openCpuCapture()) {
        RCLCPP_FATAL(
            get_logger(), "CPU fallback failed to open device %s; shutting down.",
            video_device_.c_str());
        rclcpp::shutdown();
    } else {
        startCpuTimer();
    }
}

#ifndef GPU_CAM_MINIMAL_COMPONENT_ONLY
int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<GpuCamMinimalNode>(rclcpp::NodeOptions{});
        rclcpp::spin(node);
    } catch (const std::exception & e) {
        RCLCPP_FATAL(rclcpp::get_logger("gpu_cam_minimal"), "Exception: %s", e.what());
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
#endif

RCLCPP_COMPONENTS_REGISTER_NODE(GpuCamMinimalNode)
