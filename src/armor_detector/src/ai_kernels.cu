#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace rm_auto_aim
{

// CUDA kernel: BGR8 (HWC) -> RGB (NCHW) with normalization [0,1]
template <typename T>
__global__ void bgr8_to_rgb_nchw_kernel(const unsigned char* __restrict__ bgr,
                                        T* __restrict__ dst,
                                        int W, int H)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    int idx_hwc = (y * W + x) * 3;
    unsigned char b = bgr[idx_hwc + 0];
    unsigned char g = bgr[idx_hwc + 1];
    unsigned char r = bgr[idx_hwc + 2];
    float rf = static_cast<float>(r) * (1.f / 255.f);
    float gf = static_cast<float>(g) * (1.f / 255.f);
    float bf = static_cast<float>(b) * (1.f / 255.f);
    int HW = H * W;
    int base = y * W + x;
    if constexpr (std::is_same<T, float>::value) {
        dst[0 * HW + base] = rf;
        dst[1 * HW + base] = gf;
        dst[2 * HW + base] = bf;
    } else {
        dst[0 * HW + base] = __float2half(rf);
        dst[1 * HW + base] = __float2half(gf);
        dst[2 * HW + base] = __float2half(bf);
    }
}

static inline dim3 make_block()
{
    return dim3(16, 16, 1);
}

static inline dim3 make_grid(int W, int H)
{
    dim3 block = make_block();
    return dim3((W + block.x - 1) / block.x, (H + block.y - 1) / block.y, 1);
}

void launch_bgr8_to_rgb_nchw_fp32(const unsigned char* bgr_dev, float* dst_dev, int W, int H, cudaStream_t stream)
{
    dim3 block = make_block();
    dim3 grid = make_grid(W, H);
    bgr8_to_rgb_nchw_kernel<float><<<grid, block, 0, stream>>>(bgr_dev, dst_dev, W, H);
}

void launch_bgr8_to_rgb_nchw_fp16(const unsigned char* bgr_dev, __half* dst_dev, int W, int H, cudaStream_t stream)
{
    dim3 block = make_block();
    dim3 grid = make_grid(W, H);
    bgr8_to_rgb_nchw_kernel<__half><<<grid, block, 0, stream>>>(bgr_dev, dst_dev, W, H);
}

} // namespace rm_auto_aim
