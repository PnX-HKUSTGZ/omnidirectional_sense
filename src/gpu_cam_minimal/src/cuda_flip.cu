#include "gpu_cam_minimal/cuda_flip.hpp"

#include <cuda_runtime.h>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <stdexcept>
#include <string>

namespace gpu_cam_minimal
{
namespace
{
__global__ void flipKernel(
    const uchar3 * src, size_t src_step, uchar3 * dst, size_t dst_step, int width, int height,
    int flip_code)
{
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }

    int src_x = x;
    int src_y = y;
    if (flip_code == 0) {
        src_y = height - 1 - y;  // vertical flip
    } else if (flip_code > 0) {
        src_x = width - 1 - x;   // horizontal flip
    } else {
        src_x = width - 1 - x;   // both axes
        src_y = height - 1 - y;
    }

    const unsigned char * src_row = reinterpret_cast<const unsigned char *>(src) + src_step * src_y;
    unsigned char * dst_row = reinterpret_cast<unsigned char *>(dst) + dst_step * y;

    reinterpret_cast<uchar3 *>(dst_row)[x] = reinterpret_cast<const uchar3 *>(src_row)[src_x];
}
}  // namespace

void cudaFlip(
    const cv::cuda::GpuMat & src, cv::cuda::GpuMat & dst, int flip_code, cv::cuda::Stream stream)
{
    if (src.empty()) {
        return;
    }
    if (src.type() != CV_8UC3) {
        throw std::invalid_argument("cudaFlip supports CV_8UC3 only");
    }

    const bool inplace =
        dst.data != nullptr && dst.data == src.data && dst.step == src.step && dst.type() == src.type() &&
        dst.rows == src.rows && dst.cols == src.cols;

    cv::cuda::GpuMat output;
    if (inplace) {
        output.create(src.size(), src.type());
    } else {
        dst.create(src.size(), src.type());
        output = dst;
    }

    dim3 block(32, 8);
    dim3 grid((src.cols + block.x - 1) / block.x, (src.rows + block.y - 1) / block.y);
    auto cuda_stream = cv::cuda::StreamAccessor::getStream(stream);

    flipKernel<<<grid, block, 0, cuda_stream>>>(
        reinterpret_cast<const uchar3 *>(src.data), src.step, reinterpret_cast<uchar3 *>(output.data),
        output.step, src.cols, src.rows, flip_code);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaFlip kernel launch failed: ") + cudaGetErrorString(err));
    }

    if (inplace) {
        output.copyTo(dst, stream);
    }
}
}  // namespace gpu_cam_minimal
