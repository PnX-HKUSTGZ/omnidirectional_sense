#pragma once

#include <opencv2/core/cuda.hpp>

namespace gpu_cam_minimal
{
void cudaFlip(
    const cv::cuda::GpuMat & src, cv::cuda::GpuMat & dst, int flip_code,
    cv::cuda::Stream stream = cv::cuda::Stream::Null());
}
