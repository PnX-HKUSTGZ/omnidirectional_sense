#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace rm_auto_aim
{

// 融合预处理：从任意尺寸 BGR8 源做双线性缩放 + BGR->RGB + NCHW + 归一化到 FP16
// 支持带 pitch 的源图（以字节为单位）。
void launch_resize_bgr8_to_rgb_nchw_fp16(const unsigned char* src_bgr_dev, size_t src_pitch,
										 int srcW, int srcH,
										 __half* dst_dev, int dstW, int dstH,
										 cudaStream_t stream);

// 紧凑的候选检测结构（GPU->CPU 回传）
struct PostDet
{
	float landmarks[8];   // 已按原图尺度缩放
	float x, y, w, h;     // bbox（由四点包围盒得到）
	int   label;          // 数字类别索引 [0..8]
	int   color;          // 颜色类别索引 [0..1]
	float prob;           // 置信度（sigmoid 后）
	float score_num;      // 数字类别的最大分数（用于 NMS 打分）
};

// GPU 端后处理（当前模型输出为 FP32）：sigmoid / 阈值 / argmax / 过滤 / 压缩(compact)
// out_dev: 模型输出 (N x 22)，按行存储；num_det = N
// detect_color: 0 红 / 1 蓝；sx, sy 为从网络输入到原图的尺度因子
// out_dets_dev: 设备端输出数组（容量为 max_out）；out_count_dev: 设备端计数器（调用前需要清零）
void launch_postprocess_fp32(const float* out_dev, int num_det, float conf_th,
							 int detect_color, float sx, float sy,
							 PostDet* out_dets_dev, int max_out, int* out_count_dev,
							 cudaStream_t stream);

} // namespace rm_auto_aim
