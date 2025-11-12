#pragma once

#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#ifdef GPU_CAM_HAS_OPENCV_CUDA
#include <opencv2/core/cuda.hpp>
#endif

namespace gpu_cam_minimal {

class JetsonMjpegDecoder {
public:
  struct Options {
    std::string device = "/dev/video0";
    int width = 640;
    int height = 480;
    int fps = 30;
  };

  explicit JetsonMjpegDecoder(const Options &opt, const rclcpp::Logger &logger);
  ~JetsonMjpegDecoder();

  bool is_ready() const noexcept { return ready_; }

#ifdef GPU_CAM_HAS_OPENCV_CUDA
  // Decode one frame from camera and output RGB8 on GPU
  bool grab_gpu_rgb(cv::cuda::GpuMat &out_rgb);
#else
  bool grab_gpu_rgb(...) { return false; }
#endif

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool ready_{false};
  rclcpp::Logger logger_;
};

} // namespace gpu_cam_minimal
