#ifndef GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_HPP
#define GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_HPP

#include <memory>
#include <string>
#include <sys/time.h>

#include <opencv2/core/cuda.hpp>

namespace gpu_cam_minimal {

class NvdecMjpegDecoderImpl;

// 简单的 Jetson MJPEG→NV12→RGB(GPU) 解码器包装。
// 头文件不做条件编译，实际可用性由实现文件中的 is_supported() 和 open() 返回值决定。
class NvdecMjpegDecoder {
public:
  struct Config {
    uint32_t v4l2_buffer_count{4};
    uint32_t capture_buffer_padding{2};
    bool drop_late_frames{false};
  };

  NvdecMjpegDecoder();
  ~NvdecMjpegDecoder();

  // 试图打开并初始化硬解码器；返回是否成功。
  // width/height/fps 为期望配置；驱动/解码器可能会调整为实际值。
  bool open(const std::string& video_device, int width, int height, double fps);

  // 配置底层缓冲策略，需在 open() 之前调用。
  void set_config(const Config & config);

  // 读取一帧并在 GPU 上输出 RGB 格式；成功返回 true。
  // out_rgb 会被就地写入/重用。
  // 如 capture_time 非空，则返回底层 V4L2 时间戳；timestamp_monotonic 指示是否基于 monotonic clock。
  bool read_rgb(cv::cuda::GpuMat& out_rgb, struct timeval* capture_time = nullptr,
                bool* timestamp_monotonic = nullptr);

  // 关闭资源。
  void close_decoder();

  bool is_open() const;

  // 返回当前构建/运行环境是否具备 NVDEC 路径（编译期检测结果）。
  static bool is_supported();

private:
  std::unique_ptr<NvdecMjpegDecoderImpl> impl_;
  Config config_{};
};

} // namespace gpu_cam_minimal
#endif