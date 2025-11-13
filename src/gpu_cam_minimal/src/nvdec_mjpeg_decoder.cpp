#include "gpu_cam_minimal/nvdec_mjpeg_decoder.hpp"

#include <vector>
#include <thread>
#include <cstring>
#include <unistd.h>


#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <EGL/egl.h>
#include <NvVideoDecoder.h>
#include <cudaEGL.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>

namespace gpu_cam_minimal {

struct NvdecMjpegDecoder::Impl {
  // 通用成员（即使在不支持时也存在）
  std::string device;
  int width{0};
  int height{0};
  double fps{0.0};
  bool opened{false};
  int v4l2_fd{-1};
  NvVideoDecoder* dec{nullptr};
  EGLDisplay egl_display{EGL_NO_DISPLAY};
  std::vector<unsigned char> enc_buf;

  static bool set_v4l2_mjpeg(int fd, int w, int h, double f)
  {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
      return false;
    }
    if (f > 0.0) {
      v4l2_streamparm parm{};
      parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      parm.parm.capture.timeperframe.numerator = 1;
      parm.parm.capture.timeperframe.denominator = static_cast<unsigned int>(f);
      ioctl(fd, VIDIOC_S_PARM, &parm);
    }
    return true;
  }
};

NvdecMjpegDecoder::NvdecMjpegDecoder() : impl_(new Impl) {}
NvdecMjpegDecoder::~NvdecMjpegDecoder() { close_decoder(); }

bool NvdecMjpegDecoder::open(const std::string& video_device, int width, int height, double fps)
{
  impl_->device = video_device;
  impl_->width = width;
  impl_->height = height;
  impl_->fps = fps;

  // 打开 V4L2 以读取 MJPEG 编码数据
  impl_->v4l2_fd = ::open(video_device.c_str(), O_RDONLY | O_NONBLOCK);
  if (impl_->v4l2_fd < 0) {
    impl_->opened = false;
    return false;
  }
  (void)Impl::set_v4l2_mjpeg(impl_->v4l2_fd, width, height, fps); // 尽力设置

  // 创建 Jetson 硬件解码器
  impl_->dec = NvVideoDecoder::createVideoDecoder("dec0");
  if (!impl_->dec) {
    close_decoder();
    return false;
  }
  impl_->dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0);
  impl_->dec->setOutputPlaneFormat(V4L2_PIX_FMT_MJPEG, 2 * 1024 * 1024);
  int w = width > 0 ? width : 640;
  int h = height > 0 ? height : 480;
  impl_->dec->setCapturePlaneFormat(V4L2_PIX_FMT_NV12M, w, h);
  impl_->dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
  impl_->dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
  impl_->dec->output_plane.setStreamStatus(true);
  impl_->dec->capture_plane.setStreamStatus(true);

  // 初始化 EGL 显示
  impl_->egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (impl_->egl_display == EGL_NO_DISPLAY) {
    close_decoder();
    return false;
  }
  if (!eglInitialize(impl_->egl_display, nullptr, nullptr)) {
    close_decoder();
    return false;
  }

  impl_->enc_buf.resize(2 * 1024 * 1024);
  impl_->width = w;
  impl_->height = h;
  impl_->opened = true;
  return true;
}

bool NvdecMjpegDecoder::read_bgr(cv::cuda::GpuMat& out_bgr)
{
    if (impl_->v4l2_fd < 0 || !impl_->dec) return false;

    // ---- 1. 从 V4L2 读取 MJPEG 编码帧 ----
    ssize_t len = ::read(impl_->v4l2_fd, impl_->enc_buf.data(), impl_->enc_buf.size());
    if (len <= 0) return false;

    // ---- 2. 投喂到 NvVideoDecoder output plane ----
    struct v4l2_buffer v4l2_buf{};
    struct v4l2_plane planes[VIDEO_MAX_PLANES]{};
    v4l2_buf.m.planes = planes;

    NvBuffer* nvbuf = impl_->dec->output_plane.getNthBuffer(0);
    if (!nvbuf) return false;

    std::memcpy(nvbuf->planes[0].data, impl_->enc_buf.data(), static_cast<size_t>(len));
    v4l2_buf.m.planes[0].bytesused = len;

    if (impl_->dec->output_plane.qBuffer(v4l2_buf, nullptr) < 0)
        return false;

    // ---- 3. 从 capture plane 取出解码后的帧 ----
    if (impl_->dec->capture_plane.dqBuffer(v4l2_buf, &nvbuf, nullptr, 1000) != 0)
        return false;

    // ---- 4. 映射为 EGLImage ----
    NvBufSurface* surf = reinterpret_cast<NvBufSurface*>(nvbuf);
    if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
        return false;
    }

    EGLImageKHR egl_image = surf->surfaceList[0].mappedAddr.eglImage;


    // ---- 5. 注册到 CUDA 并获取 CUeglFrame ----
    CUgraphicsResource cuda_resource{};
    if (cuGraphicsEGLRegisterImage(&cuda_resource, egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE) != CUDA_SUCCESS) {
        NvBufSurfaceUnMapEglImage(surf, 0);
        return false;
    }

    CUeglFrame eglFrame{};
    if (cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuda_resource, 0, 0) != CUDA_SUCCESS) {
        cuGraphicsUnregisterResource(cuda_resource);
        NvBufSurfaceUnMapEglImage(surf, 0);
        return false;
    }

    // ---- 6. 从 CUeglFrame 构造 NV12 并转换 BGR ----
    cv::cuda::GpuMat y(impl_->height, impl_->width, CV_8UC1, eglFrame.frame.pPitch[0]);
    cv::cuda::GpuMat uv(impl_->height / 2, impl_->width / 2, CV_8UC2, eglFrame.frame.pPitch[1]);

    cv::cuda::GpuMat nv12(impl_->height * 3 / 2, impl_->width, CV_8UC1);
    y.copyTo(nv12.rowRange(0, impl_->height));
    uv.copyTo(nv12.rowRange(impl_->height, impl_->height * 3 / 2));

    cv::cuda::cvtColor(nv12, out_bgr, cv::COLOR_YUV2BGR_NV12);

    // ---- 7. 清理 ----
    cuGraphicsUnregisterResource(cuda_resource);
    NvBufSurfaceUnMapEglImage(surf, 0);
    // NvBufSurfaceDestroy(surf); // 如果 surf 是 capture plane buffer，不需要 destroy

    return !out_bgr.empty();
}


void NvdecMjpegDecoder::close_decoder()
{
  if (!impl_->opened) return;
  if (impl_->dec) {
    // NvVideoDecoder 无显式 destroy API；依照样例通常让进程回收。
    // 在此尽量停止流并释放 fd。
    try {
      impl_->dec->output_plane.setStreamStatus(false);
      impl_->dec->capture_plane.setStreamStatus(false);
    } catch(...) {}
  }
  if (impl_->v4l2_fd >= 0) {
    ::close(impl_->v4l2_fd);
    impl_->v4l2_fd = -1;
  }
  if (impl_->egl_display != EGL_NO_DISPLAY) {
    eglTerminate(impl_->egl_display);
    impl_->egl_display = EGL_NO_DISPLAY;
  }
  impl_->opened = false;
}

bool NvdecMjpegDecoder::is_open() const { return impl_->opened; }

bool NvdecMjpegDecoder::is_supported()
{
  return true;
}

} // namespace gpu_cam_minimal
