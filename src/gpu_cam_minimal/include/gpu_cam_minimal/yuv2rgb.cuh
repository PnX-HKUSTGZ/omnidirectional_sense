#ifndef __YUV2RGB_CUH__
#define __YUV2RGB_CUH__

#include <cuda_runtime.h>

void gpuConvertYUYVtoRGB(unsigned char *src, unsigned char *dst,
                         unsigned int width, unsigned int height);

cudaError_t gpuConvertYUV422MToRGB(
    const unsigned char *y_plane, const unsigned char *u_plane,
    const unsigned char *v_plane, unsigned char *dst, size_t y_pitch,
    size_t u_pitch, size_t v_pitch, size_t dst_pitch, unsigned int width,
    unsigned int height, cudaStream_t stream = 0);

#endif