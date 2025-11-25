#include <cuda_runtime.h>
#include "gpu_cam_minimal/yuv2rgb.cuh"

__device__ inline float clamp(float val, float mn, float mx)
{
	return (val >= mn)? ((val <= mx)? val : mx) : mn;
}

__global__ void gpuConvertYUYVtoRGB_kernel(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx*2 >= width) {
		return;
	}

	for (int i = 0; i < height; ++i) {
		int y0 = src[i*width*2+idx*4+0];
		int cb = src[i*width*2+idx*4+1];
		int y1 = src[i*width*2+idx*4+2];
		int cr = src[i*width*2+idx*4+3];

		dst[i*width*3+idx*6+0] = clamp(1.164f * (y0 - 16) + 1.596f * (cr - 128)                      , 0.0f, 255.0f);
		dst[i*width*3+idx*6+1] = clamp(1.164f * (y0 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f);
		dst[i*width*3+idx*6+2] = clamp(1.164f * (y0 - 16)                       + 2.018f * (cb - 128), 0.0f, 255.0f);

		dst[i*width*3+idx*6+3] = clamp(1.164f * (y1 - 16) + 1.596f * (cr - 128)                      , 0.0f, 255.0f);
		dst[i*width*3+idx*6+4] = clamp(1.164f * (y1 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f);
		dst[i*width*3+idx*6+5] = clamp(1.164f * (y1 - 16)                       + 2.018f * (cb - 128), 0.0f, 255.0f);
	}
}

__global__ void gpuConvertYUV422MToRGB_kernel(const unsigned char *y_plane,
		const unsigned char *u_plane,
		const unsigned char *v_plane,
		unsigned char *dst,
		size_t y_pitch,
		size_t u_pitch,
		size_t v_pitch,
		size_t dst_pitch,
		unsigned int width,
		unsigned int height)
{
	unsigned int x_pair = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (x_pair >= width || row >= height) {
		return;
	}

	const size_t chroma_width = (width + 1) / 2;
	const unsigned int chroma_col = x_pair / 2;
	if (chroma_col >= chroma_width) {
		return;
	}

	const unsigned char *y_row = y_plane + row * y_pitch;
	const unsigned char *u_row = u_plane + row * u_pitch;
	const unsigned char *v_row = v_plane + row * v_pitch;
	unsigned char *dst_row = dst + row * dst_pitch + x_pair * 3;

	int y0 = y_row[x_pair];
	int cb = u_row[chroma_col];
	int cr = v_row[chroma_col];
	dst_row[0] = static_cast<unsigned char>(clamp(1.164f * (y0 - 16) + 1.596f * (cr - 128)                      , 0.0f, 255.0f));
	dst_row[1] = static_cast<unsigned char>(clamp(1.164f * (y0 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f));
	dst_row[2] = static_cast<unsigned char>(clamp(1.164f * (y0 - 16)                       + 2.018f * (cb - 128), 0.0f, 255.0f));

	if (x_pair + 1 < width) {
		int y1 = y_row[x_pair + 1];
		dst_row[3] = static_cast<unsigned char>(clamp(1.164f * (y1 - 16) + 1.596f * (cr - 128)                      , 0.0f, 255.0f));
		dst_row[4] = static_cast<unsigned char>(clamp(1.164f * (y1 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f));
		dst_row[5] = static_cast<unsigned char>(clamp(1.164f * (y1 - 16)                       + 2.018f * (cb - 128), 0.0f, 255.0f));
	}
}

void gpuConvertYUYVtoRGB(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height)
{
	unsigned char *d_src = NULL;
	unsigned char *d_dst = NULL;
	size_t planeSize = width * height * sizeof(unsigned char);

	unsigned int flags;
	bool srcIsMapped = (cudaHostGetFlags(&flags, src) == cudaSuccess) && (flags & cudaHostAllocMapped);
	bool dstIsMapped = (cudaHostGetFlags(&flags, dst) == cudaSuccess) && (flags & cudaHostAllocMapped);

	if (srcIsMapped) {
		d_src = src;
		cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);
	} else {
		cudaMalloc(&d_src, planeSize * 2);
		cudaMemcpy(d_src, src, planeSize * 2, cudaMemcpyHostToDevice);
	}
	if (dstIsMapped) {
		d_dst = dst;
		cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);
	} else {
		cudaMalloc(&d_dst, planeSize * 3);
	}

	unsigned int blockSize = 1024;
	unsigned int numBlocks = (width / 2 + blockSize - 1) / blockSize;
	gpuConvertYUYVtoRGB_kernel<<<numBlocks, blockSize>>>(d_src, d_dst, width, height);
	cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);
	cudaStreamSynchronize(NULL);

	if (!srcIsMapped) {
		cudaMemcpy(dst, d_dst, planeSize * 3, cudaMemcpyDeviceToHost);
		cudaFree(d_src);
	}
	if (!dstIsMapped) {
		cudaFree(d_dst);
	}
}

cudaError_t gpuConvertYUV422MToRGB(const unsigned char *y_plane,
		const unsigned char *u_plane,
		const unsigned char *v_plane,
		unsigned char *dst,
		size_t y_pitch,
		size_t u_pitch,
		size_t v_pitch,
		size_t dst_pitch,
		unsigned int width,
		unsigned int height,
		cudaStream_t stream)
{
	if (!y_plane || !u_plane || !v_plane || !dst || width == 0 || height == 0) {
		return cudaErrorInvalidValue;
	}
	dim3 block(32, 8);
	unsigned int pairs = (width + 1) / 2;
	dim3 grid((pairs + block.x - 1) / block.x,
		(height + block.y - 1) / block.y);
	gpuConvertYUV422MToRGB_kernel<<<grid, block, 0, stream>>>(
			y_plane,
			u_plane,
			v_plane,
			dst,
			y_pitch,
			u_pitch,
			v_pitch,
			dst_pitch,
			width,
			height);
	return cudaGetLastError();
}
