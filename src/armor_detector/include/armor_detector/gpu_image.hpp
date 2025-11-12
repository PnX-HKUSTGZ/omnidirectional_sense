#ifndef ARMOR_DETECTOR__GPU_IMAGE_HPP_
#define ARMOR_DETECTOR__GPU_IMAGE_HPP_

#include <memory>
#include <string>

#include <opencv2/core/cuda.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

namespace armor_detector {

// 轻量封装：将 GPU 图像数据与必要元数据打包
struct GpuImage {
  using GpuMatPtr = std::shared_ptr<cv::cuda::GpuMat>;
  using UniquePtr = std::unique_ptr<GpuImage>;

  GpuMatPtr gpu;               // GPU 图像数据（共享指针保证生命周期）
  std::string encoding;        // 如 "rgb8"/"bgr8"/"mono8"
  uint32_t width{0};
  uint32_t height{0};
  uint32_t step{0};
  std_msgs::msg::Header header; // frame_id + stamp

  int device_id{0};            // 可选：CUDA 设备 ID（如使用多 GPU）
  // 可选：如需携带 CUDA 流，可在不增加依赖的情况下使用不透明指针
  void* stream{nullptr};
};

}  // namespace armor_detector

#endif  // ARMOR_DETECTOR__GPU_IMAGE_HPP_
