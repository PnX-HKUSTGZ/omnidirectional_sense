#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace rm_auto_aim
{

void launch_bgr8_to_rgb_nchw_fp32(const unsigned char* bgr_dev, float* dst_dev, int W, int H, cudaStream_t stream);
void launch_bgr8_to_rgb_nchw_fp16(const unsigned char* bgr_dev, __half* dst_dev, int W, int H, cudaStream_t stream);

} // namespace rm_auto_aim
