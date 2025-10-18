#ifndef VIDEO_READER__GPU_IMAGE_TYPE_ADAPTER_HPP_
#define VIDEO_READER__GPU_IMAGE_TYPE_ADAPTER_HPP_

#include "video_reader/gpu_image.hpp"

#include <rclcpp/type_adapter.hpp>

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>

namespace rclcpp {

// 将自定义的 video_reader::GpuImage 与标准 sensor_msgs::msg::Image 互转
template<>
struct TypeAdapter<video_reader::GpuImage, sensor_msgs::msg::Image> {
  using is_specialized = std::true_type;
  using custom_type = video_reader::GpuImage;
  using ros_message_type = sensor_msgs::msg::Image;

  static void convert_to_ros_message(const custom_type & src, ros_message_type & dst) {
    // 仅在跨进程/录包时触发：下载 GPU -> CPU
    cv::Mat cpu;
    if (src.gpu) {
      src.gpu->download(cpu);
    } else {
      cpu = cv::Mat(src.height, src.width, CV_8UC3);  // 兜底，占位
      cpu.setTo(cv::Scalar(0, 0, 0));
    }
    auto cv_msg = cv_bridge::CvImage(src.header, src.encoding, cpu);
    dst = *cv_msg.toImageMsg();
  }

  static void convert_to_custom(const ros_message_type & src, custom_type & dst) {
    // 仅在需要从 CPU 源上传到 GPU 的场景
    auto cv_ptr = cv_bridge::toCvCopy(src, src.encoding);
    auto g = std::make_shared<cv::cuda::GpuMat>();
    g->upload(cv_ptr->image);

    dst.gpu = std::move(g);
    dst.encoding = src.encoding;
    dst.width = src.width;
    dst.height = src.height;
    dst.step = src.step;
    dst.header = src.header;
  }
};

}  // namespace rclcpp

// 声明：允许直接将自定义类型作为 ROS 类型使用（用于 Publisher/Subscription 模板参数）
RCLCPP_USING_CUSTOM_TYPE_AS_ROS_MESSAGE_TYPE(video_reader::GpuImage, sensor_msgs::msg::Image);

namespace video_reader {
// 可选别名：并非必须
using GpuImageAdapter = rclcpp::TypeAdapter<video_reader::GpuImage, sensor_msgs::msg::Image>;
}

#endif  // VIDEO_READER__GPU_IMAGE_TYPE_ADAPTER_HPP_
