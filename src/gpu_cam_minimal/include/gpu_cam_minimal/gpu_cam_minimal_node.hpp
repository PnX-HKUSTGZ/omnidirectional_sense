#pragma once

#include <linux/videodev2.h>
#include <sys/time.h>

#include <armor_detector/gpu_image.hpp>
#include <armor_detector/gpu_image_type_adapter.hpp>
#include <atomic>
#include <camera_info_manager/camera_info_manager.hpp>
#include <cstdint>
#include <memory>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include <thread>

#include "gpu_cam_minimal/nvdec_mjpeg_decoder.hpp"

class GpuCamMinimalNode : public rclcpp::Node
{
public:
    explicit GpuCamMinimalNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
    ~GpuCamMinimalNode() override;

private:
    bool openCpuCapture();
    void handleNvdecFailure(const std::string & reason);
    void fallbackToCpuCapture(const std::string & reason);
    void initializeCudaDevice();
    void openCamera();
    void apply_camera_controls();
    static int64_t timeval_to_ns(const timeval & tv);
    rclcpp::Time convert_v4l2_timestamp(const timeval & tv, bool is_monotonic);
    int64_t getTimeOffset() const;
    void setTSCOffset();
    rclcpp::Time system_now_with_offset() const;
    void tick();
    void startCpuTimer();
    void startNvdecCaptureLoop();
    void stopNvdecCaptureLoop();
    void nvdecCaptureLoop();
    bool readNvdecFrame(cv::cuda::GpuMat & gpu_rgb, rclcpp::Time & timestamp);
    void publishFrame(
        const cv::cuda::GpuMat & gpu_rgb, const cv::Mat * cpu_rgb, const rclcpp::Time & timestamp);
    void publishDebugImage(
        const sensor_msgs::msg::CameraInfo & info, const cv::cuda::GpuMat & gpu_rgb,
        const cv::Mat * cpu_rgb);
    void updateDebugStats(const rclcpp::Time & capture_ts, const rclcpp::Time & publish_ts);
    static int parse_device_id(const std::string & dev);

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
    bool flip_image_{false};
    int cuda_device_id_{0};
    bool debug_enabled_{false};
    bool use_v4l2_buffer_timestamps_{true};
    rclcpp::Duration timestamp_offset_{0, 0};
    int64_t tsc_offset_{0};
    int nvdec_v4l2_buffer_count_{4};
    int nvdec_capture_buffer_padding_{2};
    bool nvdec_drop_late_frames_{true};
    static constexpr int kNvdecFailureThreshold = -1;
    int nvdec_failure_count_{0};
    struct CameraControlParams
    {
        int brightness{0};
        int contrast{32};
        int saturation{64};
        int hue{0};
        bool white_balance_automatic{true};
        int gamma{300};
        int gain{32};
        int power_line_frequency{1};
        int white_balance_temperature{4600};
        int sharpness{32};
        int backlight_compensation{0};
        int auto_exposure{V4L2_EXPOSURE_APERTURE_PRIORITY};
        int exposure_time_absolute{313};
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

    bool use_hw_mjpeg_{false};

    // NVDEC 解码器实例（按需）
    std::unique_ptr<gpu_cam_minimal::NvdecMjpegDecoder> nvdec_;
    std::thread nvdec_thread_;
    std::atomic<bool> nvdec_thread_running_{false};
    std::atomic<bool> nvdec_thread_stop_{false};
    std::mutex debug_stats_mutex_;
    rclcpp::Time debug_window_start_;
    size_t debug_window_frames_{0};
    double debug_window_latency_ms_{0.0};
};
