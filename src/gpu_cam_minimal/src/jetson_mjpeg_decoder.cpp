#include "gpu_cam_minimal/jetson_mjpeg_decoder.hpp"

#include <stdexcept>

#include <linux/videodev2.h>

#ifdef GPU_CAM_HAS_JETSON_MMA
#include <NvVideoDecoder.h>
#include <NvBufSurface.h>
#include <nvbuf_utils.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace gpu_cam_minimal {

struct JetsonMjpegDecoder::Impl {
  Options opt;
  int fd{-1};
  bool open_ok{false};

#ifdef GPU_CAM_HAS_JETSON_MMA
  std::unique_ptr<NvVideoDecoder> dec;
  EGLDisplay egl_display{EGL_NO_DISPLAY};
  bool egl_inited{false};
#endif
};

JetsonMjpegDecoder::JetsonMjpegDecoder(const Options &opt, const rclcpp::Logger &logger)
    : impl_(std::make_unique<Impl>()), logger_(logger) {
  impl_->opt = opt;

#ifdef GPU_CAM_HAS_JETSON_MMA
  // Open decoder in MJPEG mode. Real implementation needs V4L2 capture of compressed MJPEG buffers.
  try {
    impl_->dec.reset(NvVideoDecoder::createVideoDecoder("mjpeg-decoder", V4L2_PIX_FMT_MJPEG));
  } catch (...) {
    RCLCPP_ERROR(logger_, "Failed to create NvVideoDecoder (MJPEG)");
  }
  ready_ = (impl_->dec != nullptr);
#else
  ready_ = false;
#endif
}

JetsonMjpegDecoder::~JetsonMjpegDecoder() = default;

#ifdef GPU_CAM_HAS_OPENCV_CUDA
bool JetsonMjpegDecoder::grab_gpu_rgb(cv::cuda::GpuMat &out_rgb) {
#ifdef GPU_CAM_HAS_JETSON_MMA
  // TODO: Implement V4L2 capture of MJPEG, push to NvVideoDecoder, map decoded NvBuffer to CUDA and convert to RGB
  // Placeholder: return false until full implementation is added
  (void)out_rgb;
  return false;
#else
  (void)out_rgb;
  return false;
#endif
}
#endif

} // namespace gpu_cam_minimal
