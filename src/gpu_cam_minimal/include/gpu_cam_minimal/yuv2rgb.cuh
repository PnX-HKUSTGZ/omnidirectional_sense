#ifndef __YUV2RGB_CUH__
#define __YUV2RGB_CUH__

void gpuConvertYUYVtoRGB(unsigned char *src, unsigned char *dst,
		unsigned int width, unsigned int height);

#endif