#include "gpu_cam_minimal/jetson_mjpeg_decoder.hpp"

#include <stdexcept>

#include <linux/videodev2.h>

#ifdef GPU_CAM_HAS_JETSON_MMA
#include <NvVideoDecoder.h>
#if __has_include(<NvBufSurface.h>)
#include <NvBufSurface.h>
#elif __has_include(<libnvbufsurface.h>)
#include <libnvbufsurface.h>
#else
#warning "Neither NvBufSurface.h nor libnvbufsurface.h found in include paths"
#endif
#if __has_include(<nvbuf_utils.h>)
#include <nvbuf_utils.h>
#elif __has_include(<libnvbuf_utils.h>)
#include <libnvbuf_utils.h>
#else
#warning "Neither nvbuf_utils.h nor libnvbuf_utils.h found in include paths"
#endif
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cudaEGL.h>
#include <cuda_runtime.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <cstring>
#include <vector>

namespace gpu_cam_minimal {

struct JetsonMjpegDecoder::Impl {
  Options opt;
  int cam_fd{-1};
  bool cam_ok{false};

#ifdef GPU_CAM_HAS_JETSON_MMA
  std::unique_ptr<NvVideoDecoder> dec;
  EGLDisplay egl_display{EGL_NO_DISPLAY};
  bool egl_inited{false};
  bool cap_setup{false};
  int dst_dma_fd{-1};
#endif

  struct V4L2Buffer { void* start{nullptr}; size_t length{0}; v4l2_buffer vbuf{}; };
  std::vector<V4L2Buffer> cam_buffers;  
};

JetsonMjpegDecoder::JetsonMjpegDecoder(const Options &opt, const rclcpp::Logger &logger)
    : impl_(std::make_unique<Impl>()), logger_(logger) {
  impl_->opt = opt;

#ifdef GPU_CAM_HAS_JETSON_MMA
  // 1) Open V4L2 camera in MJPEG mode
  impl_->cam_fd = ::open(opt.device.c_str(), O_RDWR | O_NONBLOCK);
  if (impl_->cam_fd < 0) {
    RCLCPP_ERROR(logger_, "Failed to open %s", opt.device.c_str());
    ready_ = false;
    return;
  }
  // Set format to MJPEG
  struct v4l2_format fmt{};
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = opt.width;
  fmt.fmt.pix.height = opt.height;
  fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  if (ioctl(impl_->cam_fd, VIDIOC_S_FMT, &fmt) < 0) {
    RCLCPP_ERROR(logger_, "VIDIOC_S_FMT failed (MJPEG)");
  }
  // Request buffers
  struct v4l2_requestbuffers req{};
  req.count = 6;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (ioctl(impl_->cam_fd, VIDIOC_REQBUFS, &req) < 0) {
    RCLCPP_ERROR(logger_, "VIDIOC_REQBUFS failed");
  } else {
    impl_->cam_buffers.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
      struct v4l2_buffer buf{};
      buf.type = req.type;
      buf.memory = req.memory;
      buf.index = i;
      if (ioctl(impl_->cam_fd, VIDIOC_QUERYBUF, &buf) < 0) {
        RCLCPP_ERROR(logger_, "VIDIOC_QUERYBUF failed");
        continue;
      }
      void* start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, impl_->cam_fd, buf.m.offset);
      if (start == MAP_FAILED) {
        RCLCPP_ERROR(logger_, "mmap failed for camera buffer %u", i);
        continue;
      }
      impl_->cam_buffers[i].start = start;
      impl_->cam_buffers[i].length = buf.length;
      impl_->cam_buffers[i].vbuf = buf;
    }
    // Queue all buffers
    for (uint32_t i = 0; i < impl_->cam_buffers.size(); ++i) {
      struct v4l2_buffer buf = impl_->cam_buffers[i].vbuf;
      if (ioctl(impl_->cam_fd, VIDIOC_QBUF, &buf) < 0) {
        RCLCPP_ERROR(logger_, "VIDIOC_QBUF failed on camera buffer %u", i);
      }
    }
    // Stream on
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(impl_->cam_fd, VIDIOC_STREAMON, &type) == 0) {
      impl_->cam_ok = true;
    } else {
      RCLCPP_ERROR(logger_, "VIDIOC_STREAMON failed on camera");
    }
  }

  // 2) Create decoder in MJPEG mode
  try {
    impl_->dec.reset(NvVideoDecoder::createVideoDecoder("mjpeg-decoder", V4L2_PIX_FMT_MJPEG));
  } catch (...) {
    RCLCPP_ERROR(logger_, "Failed to create NvVideoDecoder (MJPEG)");
  }
  if (!impl_->dec) { ready_ = false; return; }

  // Configure decoder output plane for chunked input
  const uint32_t kChunkSize = 2 * 1024 * 1024;
  if (impl_->dec->setOutputPlaneFormat(V4L2_PIX_FMT_MJPEG, kChunkSize) < 0) {
    RCLCPP_ERROR(logger_, "setOutputPlaneFormat failed");
  }
  // Allow partial chunks
  impl_->dec->setFrameInputMode(1);
  // Setup output plane buffers
  if (impl_->dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false) < 0) {
    RCLCPP_ERROR(logger_, "decoder output_plane setupPlane failed");
  }
  if (impl_->dec->output_plane.setStreamStatus(true) < 0) {
    RCLCPP_ERROR(logger_, "decoder output_plane stream on failed");
  }

  ready_ = impl_->cam_ok;
#else
  ready_ = false;
#endif
}

JetsonMjpegDecoder::~JetsonMjpegDecoder() = default;

#ifdef GPU_CAM_HAS_OPENCV_CUDA
bool JetsonMjpegDecoder::grab_gpu_rgb(cv::cuda::GpuMat &out_rgb) {
#ifdef GPU_CAM_HAS_JETSON_MMA
  if (!impl_->cam_ok || !impl_->dec) return false;

  // 1) Dequeue one compressed MJPEG frame from camera
  struct v4l2_buffer cam_buf{}; struct v4l2_plane cam_planes[1]{};
  cam_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  cam_buf.memory = V4L2_MEMORY_MMAP;
  cam_buf.m.planes = cam_planes;
  if (ioctl(impl_->cam_fd, VIDIOC_DQBUF, &cam_buf) < 0) {
    if (errno != EAGAIN) {
      RCLCPP_WARN(logger_, "VIDIOC_DQBUF camera failed: %s", strerror(errno));
    }
    return false;
  }
  auto &local_cam = impl_->cam_buffers[cam_buf.index];

  // 2) Get an available decoder output plane buffer
  struct v4l2_buffer v4l2_out{}; struct v4l2_plane out_planes[MAX_PLANES]{};
  v4l2_out.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  v4l2_out.memory = V4L2_MEMORY_MMAP;
  v4l2_out.m.planes = out_planes;
  NvBuffer *outbuf = nullptr;
  if (impl_->dec->output_plane.getNumQueuedBuffers() < impl_->dec->output_plane.getNumBuffers()) {
    // Use next not-queued buffer index
    uint32_t idx = impl_->dec->output_plane.getNumQueuedBuffers();
    outbuf = impl_->dec->output_plane.getNthBuffer(idx);
    v4l2_out.index = idx;
  } else {
    // Dequeue one to reuse
    if (impl_->dec->output_plane.dqBuffer(v4l2_out, &outbuf, nullptr, 0) < 0) {
      // Requeue camera buffer and bail
      ioctl(impl_->cam_fd, VIDIOC_QBUF, &cam_buf);
      return false;
    }
  }
  if (!outbuf) {
    ioctl(impl_->cam_fd, VIDIOC_QBUF, &cam_buf);
    return false;
  }

  // Copy compressed bytes into decoder buffer
  size_t copy_bytes = std::min((size_t)local_cam.vbuf.bytesused, (size_t)outbuf->planes[0].length);
  std::memcpy(outbuf->planes[0].data, local_cam.start, copy_bytes);
  v4l2_out.m.planes[0].bytesused = copy_bytes;
  // Queue decoder buffer
  if (impl_->dec->output_plane.qBuffer(v4l2_out, nullptr) < 0) {
    RCLCPP_WARN(logger_, "decoder qBuffer failed");
    ioctl(impl_->cam_fd, VIDIOC_QBUF, &cam_buf);
    return false;
  }

  // Requeue camera buffer immediately
  if (ioctl(impl_->cam_fd, VIDIOC_QBUF, &cam_buf) < 0) {
    RCLCPP_WARN(logger_, "VIDIOC_QBUF camera failed: %s", strerror(errno));
  }

  // 3) On first decoded frames, set up capture plane
  if (!impl_->cap_setup) {
    struct v4l2_event ev{};
    int ret = impl_->dec->dqEvent(ev, 100 /*ms*/);
    if (ret == 0 && ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
      struct v4l2_format format{}; struct v4l2_crop crop{};
      impl_->dec->capture_plane.getFormat(format);
      impl_->dec->capture_plane.getCrop(crop);

      // Set capture plane format
      if (impl_->dec->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                    format.fmt.pix_mp.width, format.fmt.pix_mp.height) < 0) {
        RCLCPP_ERROR(logger_, "setCapturePlaneFormat failed");
        return false;
      }
      int32_t min_cap = 0; impl_->dec->getMinimumCapturePlaneBuffers(min_cap);
      if (impl_->dec->capture_plane.setupPlane(V4L2_MEMORY_MMAP, min_cap + 4, false, false) < 0) {
        RCLCPP_ERROR(logger_, "capture_plane setupPlane failed");
        return false;
      }
      if (impl_->dec->capture_plane.setStreamStatus(true) < 0) {
        RCLCPP_ERROR(logger_, "capture_plane stream on failed");
        return false;
      }
      // Queue all capture buffers
      for (uint32_t i = 0; i < impl_->dec->capture_plane.getNumBuffers(); ++i) {
        struct v4l2_buffer cap_v4l2{}; struct v4l2_plane cap_planes[MAX_PLANES]{};
        cap_v4l2.index = i; cap_v4l2.m.planes = cap_planes;
        if (impl_->dec->capture_plane.qBuffer(cap_v4l2, nullptr) < 0) {
          RCLCPP_ERROR(logger_, "capture qBuffer failed at %u", i);
          return false;
        }
      }

      // Allocate destination pitch-linear RGB surface
      NvBufSurf::NvCommonAllocateParams params{};
      params.memType = NVBUF_MEM_SURFACE_ARRAY;
      params.width = format.fmt.pix_mp.width;
      params.height = format.fmt.pix_mp.height;
      params.layout = NVBUF_LAYOUT_PITCH;
      params.colorFormat = NVBUF_COLOR_FORMAT_RGB;
      params.memtag = NvBufSurfaceTag_VIDEO_CONVERT;
      if (impl_->dst_dma_fd != -1) { NvBufSurf::NvDestroy(impl_->dst_dma_fd); impl_->dst_dma_fd = -1; }
      if (NvBufSurf::NvAllocate(&params, 1, &impl_->dst_dma_fd) < 0) {
        RCLCPP_ERROR(logger_, "NvAllocate dst surface failed");
        return false;
      }
      impl_->cap_setup = true;
    }
  }

  if (!impl_->cap_setup) {
    return false; // wait for setup
  }

  // 4) Try to dequeue a decoded frame
  struct v4l2_buffer cap_v4l2{}; struct v4l2_plane cap_planes[MAX_PLANES]{}; NvBuffer *dec_buffer = nullptr;
  cap_v4l2.m.planes = cap_planes;
  if (impl_->dec->capture_plane.dqBuffer(cap_v4l2, &dec_buffer, nullptr, 0) < 0) {
    return false;
  }

  // 5) Blocklinear -> RGB pitch-linear via NvTransform
  NvBufSurf::NvCommonTransformParams t{};
  // full frame
  t.src_top = t.src_left = t.dst_top = t.dst_left = 0;
  // Query size from capture plane format
  struct v4l2_format cf{}; impl_->dec->capture_plane.getFormat(cf);
  t.src_width = t.dst_width = cf.fmt.pix_mp.width;
  t.src_height = t.dst_height = cf.fmt.pix_mp.height;
  t.flag = NVBUFSURF_TRANSFORM_FILTER;
  t.flip = NvBufSurfTransform_None;
  t.filter = NvBufSurfTransformInter_Nearest;

  if (NvBufSurf::NvTransform(&t, dec_buffer->planes[0].fd, impl_->dst_dma_fd) < 0) {
    RCLCPP_WARN(logger_, "NvTransform failed");
    // Requeue and return
    impl_->dec->capture_plane.qBuffer(cap_v4l2, nullptr);
    return false;
  }

  // 6) Map CPU pointer and upload to GPU as fallback (still HW decoded)
  NvBufSurface *surf = nullptr;
  if (NvBufSurfaceFromFd(impl_->dst_dma_fd, (void**)&surf) != 0) {
    RCLCPP_WARN(logger_, "NvBufSurfaceFromFd failed");
    impl_->dec->capture_plane.qBuffer(cap_v4l2, nullptr);
    return false;
  }
  if (NvBufSurfaceMap(surf, 0, 0, NVBUF_MAP_READ) != 0) {
    RCLCPP_WARN(logger_, "NvBufSurfaceMap failed");
    impl_->dec->capture_plane.qBuffer(cap_v4l2, nullptr);
    return false;
  }
  NvBufSurfaceSyncForCpu(surf, 0, 0);
  int rows = surf->surfaceList[0].height;
  int cols = surf->surfaceList[0].width;
  int pitch = surf->surfaceList[0].pitch;
  void* cpu_ptr = surf->surfaceList[0].mappedAddr.addr[0];
  cv::Mat rgb_cpu(rows, cols, CV_8UC3, cpu_ptr, pitch);
  // Ensure out_rgb is allocated and copy
  out_rgb.create(rows, cols, CV_8UC3);
  cv::cuda::GpuMat src_gpu; src_gpu.create(rows, cols, CV_8UC3);
  src_gpu.upload(rgb_cpu);
  src_gpu.copyTo(out_rgb);
  NvBufSurfaceUnMap(surf, 0, 0);

  // 7) Requeue capture buffer
  impl_->dec->capture_plane.qBuffer(cap_v4l2, nullptr);

  ++frames_decoded_;
  return true;
#else
  (void)out_rgb;
  return false;
#endif
}
#endif

} // namespace gpu_cam_minimal
