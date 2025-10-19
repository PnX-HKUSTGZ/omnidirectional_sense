#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <type_traits>
#include "armor_detector/ai_kernels.hpp"

namespace rm_auto_aim
{

// ---------------- GPU Post Process (sigmoid/threshold/argmax/compact) ----------------

__device__ inline float sigmoidf_fast(float x)
{
    return 1.f / (1.f + __expf(-x));
}

__global__ void postprocess_kernel_fp32(const float* __restrict__ out,
                                   int num_det, float conf_th,
                                   int detect_color, float sx, float sy,
                                   PostDet* __restrict__ out_dets, int max_out,
                                   int* __restrict__ out_count)
{
    const int K = 22; // attributes per det
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_det) return;

    const float* row = out + i * K;

    // confidence
    float conf = sigmoidf_fast(*(row + 8));
    if (conf < conf_th) return;

    // color argmax over cols [9, 13)
    int best_color = 0;
    float best_color_val = -1e30f;
    for (int c = 0; c < 4; ++c) {
        float v = *(row + 9 + c);
        if (v > best_color_val) { best_color_val = v; best_color = c; }
    }
    if (best_color >= 2) return; // 只保留红/蓝
    if ((detect_color == 0 && best_color == 1) || (detect_color == 1 && best_color == 0)) return;

    // number argmax over cols [13, 22)
    int best_label = 0;
    float best_num_val = -1e30f;
    for (int c = 0; c < 9; ++c) {
        float v = *(row + 13 + c);
        if (v > best_num_val) { best_num_val = v; best_label = c; }
    }

    // landmarks (scaled to original image by sx/sy)
    float lms[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float v = *(row + j);
        lms[j] = v * ((j & 1) ? sy : sx);
    }

    // bbox from landmarks
    float minx = 1e9f, maxx = 0.f, miny = 1e9f, maxy = 0.f;
    for (int k = 0; k < 8; k += 2) {
        float x = lms[k];
        float y = lms[k + 1];
        minx = fminf(minx, x);
        maxx = fmaxf(maxx, x);
        miny = fminf(miny, y);
        maxy = fmaxf(maxy, y);
    }
    float w = maxx - minx;
    float h = maxy - miny;
    if (w <= 0.f || h <= 0.f) return;

    int widx = atomicAdd(out_count, 1);
    if (widx >= max_out) return;

    PostDet pd;
    #pragma unroll
    for (int j = 0; j < 8; ++j) pd.landmarks[j] = lms[j];
    pd.x = minx; pd.y = miny; pd.w = w; pd.h = h;
    pd.label = best_label; pd.color = best_color;
    pd.prob = conf; pd.score_num = best_num_val;
    out_dets[widx] = pd;
}

void launch_postprocess_fp32(const float* out_dev, int num_det, float conf_th,
                             int detect_color, float sx, float sy,
                             PostDet* out_dets_dev, int max_out, int* out_count_dev,
                             cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_det + threads - 1) / threads;
    postprocess_kernel_fp32<<<blocks, threads, 0, stream>>>(out_dev, num_det, conf_th,
        detect_color, sx, sy, out_dets_dev, max_out, out_count_dev);
}

static inline dim3 make_block()
{
    return dim3(16, 16, 1);
}

// NOTE: simple non-resize BGR->RGB kernel and its launcher removed as unused.

// Fused resize + BGR->RGB + normalize to NCHW FP16
__global__ void resize_bgr8_to_rgb_nchw_fp16_kernel(const unsigned char* __restrict__ src,
                                                    size_t src_pitch,
                                                    int srcW, int srcH,
                                                    __half* __restrict__ dst,
                                                    int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // output x
    int y = blockIdx.y * blockDim.y + threadIdx.y; // output y
    if (x >= dstW || y >= dstH) return;

    // map to source with bilinear
    float scaleX = static_cast<float>(srcW) / dstW;
    float scaleY = static_cast<float>(srcH) / dstH;
    float fx = (x + 0.5f) * scaleX - 0.5f;
    float fy = (y + 0.5f) * scaleY - 0.5f;
    int x0 = floorf(fx);
    int y0 = floorf(fy);
    float dx = fx - x0;
    float dy = fy - y0;
    int x1 = min(x0 + 1, srcW - 1);
    int y1 = min(y0 + 1, srcH - 1);
    x0 = max(0, x0);
    y0 = max(0, y0);

    const unsigned char* row0 = src + y0 * src_pitch;
    const unsigned char* row1 = src + y1 * src_pitch;

    // read 4 neighbors (BGR)
    const unsigned char* p00 = row0 + x0 * 3;
    const unsigned char* p10 = row0 + x1 * 3;
    const unsigned char* p01 = row1 + x0 * 3;
    const unsigned char* p11 = row1 + x1 * 3;

    float b00 = static_cast<float>(p00[0]);
    float g00 = static_cast<float>(p00[1]);
    float r00 = static_cast<float>(p00[2]);
    float b10 = static_cast<float>(p10[0]);
    float g10 = static_cast<float>(p10[1]);
    float r10 = static_cast<float>(p10[2]);
    float b01 = static_cast<float>(p01[0]);
    float g01 = static_cast<float>(p01[1]);
    float r01 = static_cast<float>(p01[2]);
    float b11 = static_cast<float>(p11[0]);
    float g11 = static_cast<float>(p11[1]);
    float r11 = static_cast<float>(p11[2]);

    // bilinear interpolate
    float w00 = (1 - dx) * (1 - dy);
    float w10 = dx * (1 - dy);
    float w01 = (1 - dx) * dy;
    float w11 = dx * dy;
    float bf = (b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11) * (1.f/255.f);
    float gf = (g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11) * (1.f/255.f);
    float rf = (r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11) * (1.f/255.f);

    int HW = dstW * dstH;
    int base = y * dstW + x;
    dst[0 * HW + base] = __float2half(rf);
    dst[1 * HW + base] = __float2half(gf);
    dst[2 * HW + base] = __float2half(bf);
}

void launch_resize_bgr8_to_rgb_nchw_fp16(const unsigned char* src_bgr_dev, size_t src_pitch,
                                         int srcW, int srcH,
                                         __half* dst_dev, int dstW, int dstH,
                                         cudaStream_t stream)
{
    dim3 block = make_block();
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y, 1);
    resize_bgr8_to_rgb_nchw_fp16_kernel<<<grid, block, 0, stream>>>(src_bgr_dev, src_pitch, srcW, srcH,
                                                                    dst_dev, dstW, dstH);
}

} // namespace rm_auto_aim
