#include "gpu_cam_minimal/nvdec_mjpeg_decoder.hpp"

#include <drm/drm_fourcc.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <NvBuffer.h>
#include <NvVideoDecoder.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <libv4l2.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <rcutils/logging_macros.h>
#include <stdarg.h>

#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include "NvBufSurface.h"
#include "NvUtils.h"
#include "gpu_cam_minimal/nvdec_mjpeg_decoder_impl.hpp"
#include "gpu_cam_minimal/yuv2rgb.cuh"

namespace gpu_cam_minimal
{

NvdecMjpegDecoder::NvdecMjpegDecoder() : impl_(new NvdecMjpegDecoderImpl) {}
NvdecMjpegDecoder::~NvdecMjpegDecoder() { close_decoder(); }

void NvdecMjpegDecoder::set_config(const Config & config) { config_ = config; }

bool NvdecMjpegDecoder::open(const std::string & video_device, int width, int height, double fps)
{
    impl_->device = video_device;
    impl_->width = width;
    impl_->height = height;
    impl_->fps = fps;
    impl_->requested_v4l2_buffers = std::max<uint32_t>(2, config_.v4l2_buffer_count);
    impl_->capture_buffer_padding = std::max<uint32_t>(1, config_.capture_buffer_padding);
    impl_->drop_late_frames = config_.drop_late_frames;

    // 打开 V4L2 camera（MJPEG bitstream）
    impl_->v4l2_fd = ::open(video_device.c_str(), O_RDWR | O_NONBLOCK);
    if (impl_->v4l2_fd < 0) {
        impl_->opened = false;
        RCUTILS_LOG_ERROR_NAMED(
            "nvdec_mjpeg_decoder", "Failed to open V4L2 device %s: %s", video_device.c_str(),
            strerror(errno));
        return false;
    }

    if (!impl_->initEglExtensions()) {
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to initialize EGL KHR extensions");
        return false;
    }

    // 尽力设置分辨率
    (void)NvdecMjpegDecoderImpl::set_v4l2_mjpeg(impl_->v4l2_fd, width, height, fps);

    // 确认支持 STREAMING 能力
    v4l2_capability cap{};
    if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_QUERYCAP, &cap) == 0) {
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder", "V4L2 device does not support STREAMING API");
            close_decoder();
            return false;
        }
    }

    // 初始化 V4L2 MMAP 缓冲并开启 STREAMON
    {
        v4l2_requestbuffers req{};
        req.count = impl_->requested_v4l2_buffers;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_REQBUFS, &req) < 0) {
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder", "VIDIOC_REQBUFS failed: %s", strerror(errno));
            close_decoder();
            return false;
        }
        if (req.count < 2) {
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder", "Insufficient V4L2 buffers allocated: %u", req.count);
            close_decoder();
            return false;
        }
        impl_->v4l2_bufs.resize(req.count);
        for (uint32_t i = 0; i < req.count; ++i) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_QUERYBUF, &buf) < 0) {
                RCUTILS_LOG_ERROR_NAMED(
                    "nvdec_mjpeg_decoder", "VIDIOC_QUERYBUF failed: %s", strerror(errno));
                close_decoder();
                return false;
            }
            void * start = mmap(
                nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, impl_->v4l2_fd,
                buf.m.offset);
            if (start == MAP_FAILED) {
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "mmap failed: %s", strerror(errno));
                close_decoder();
                return false;
            }
            impl_->v4l2_bufs[i].start = start;
            impl_->v4l2_bufs[i].length = buf.length;

            if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
                RCUTILS_LOG_ERROR_NAMED(
                    "nvdec_mjpeg_decoder", "VIDIOC_QBUF failed: %s", strerror(errno));
                close_decoder();
                return false;
            }
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_STREAMON, &type) < 0) {
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder", "VIDIOC_STREAMON failed: %s", strerror(errno));
            close_decoder();
            return false;
        }
        impl_->v4l2_streaming = true;
    }

    // ---- 创建 Jetson NVDEC MJPEG 解码器 ----
    impl_->dec.reset(NvVideoDecoder::createVideoDecoder("dec0"));
    if (!impl_->dec) {
        close_decoder();
        return false;
    }

    // 订阅分辨率变化事件
    if (impl_->dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0) < 0) {
        RCUTILS_LOG_WARN_NAMED(
            "nvdec_mjpeg_decoder",
            "subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE) failed; will use fallback without "
            "event.");
    }

    // 设置 OUTPUT 平面格式（输入单帧 JPEG 码流）
    // 使用 V4L2_PIX_FMT_MJPEG 能在部分 Jetson NVDEC 版本上更稳定触发内部解析，避免后续 capture_plane.getFormat EINVAL。
    if (impl_->dec->setOutputPlaneFormat(V4L2_PIX_FMT_MJPEG, 2 * 1024 * 1024) < 0) {
        RCUTILS_LOG_ERROR_NAMED(
            "nvdec_mjpeg_decoder", "setOutputPlaneFormat(V4L2_PIX_FMT_MJPEG) failed");
        close_decoder();
        return false;
    }

    // 允许以 bitstream 方式输入 MJPEG（需在设置 plane format 之后调用）
    impl_->dec->setFrameInputMode(1);

    // 配置 OUTPUT 平面（必须在 streamon 前）
    if (impl_->dec->output_plane.setupPlane(V4L2_MEMORY_MMAP, 2, true, false) < 0)
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "output_plane.setupPlane failed");

    // 开启 OUTPUT 流（必须在 capture 前）
    if (impl_->dec->output_plane.setStreamStatus(true) < 0)
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "output_plane.streamon failed");
    else
        impl_->start_output_reclaim_thread();

    // ---- EGL 初始化 ----
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
    impl_->capture_configured = false;
    impl_->capture_num_planes = 0;
    impl_->capture_pixfmt = 0;
    impl_->dec_w = 0;
    impl_->dec_h = 0;
    impl_->frames_fed = 0;
    impl_->out_next_idx = 0;
    impl_->out_in_use[0] = false;
    impl_->out_in_use[1] = false;
    impl_->opened = true;
    return true;
}

bool NvdecMjpegDecoder::read_rgb(
    cv::cuda::GpuMat & out_rgb, struct timeval * capture_time, bool * timestamp_monotonic)
{
    if (impl_->v4l2_fd < 0 || !impl_->dec) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Invalid decoder or v4l2_fd not opened.");
        return false;
    }

    // 1) 抓取一帧摄像头的 MJPEG 压缩数据
    v4l2_buffer vbuf{};
    void * cam_data = nullptr;
    size_t cam_len = 0;
    if (!impl_->grab_camera_frame(vbuf, cam_data, cam_len)) {
        // helper 内部已做必要日志与回队（当需要时）。
        return false;
    }

    if (capture_time) {
        *capture_time = vbuf.timestamp;
    }
    if (timestamp_monotonic) {
        *timestamp_monotonic = (vbuf.flags & V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) != 0;
    }

    // 2) 将压缩数据喂给 NVDEC 并从 capture 平面取一帧解码输出
    v4l2_buffer cbuf{};
    v4l2_plane cplanes[VIDEO_MAX_PLANES]{};
    cbuf.m.planes = cplanes;
    NvBuffer * cap_nvbuf = nullptr;
    if (!impl_->feed_decoder_and_dequeue_capture(vbuf, cam_data, cam_len, cap_nvbuf, cbuf)) {
        return false;
    }

    // 3) 将 capture NvBuffer 转成 GPU 上的 RGB 图像
    if (!impl_->convert_capture_to_rgb(cap_nvbuf, cbuf, out_rgb)) {
        return false;
    }

    if (out_rgb.empty()) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Output RGB frame is empty after decoding.");
        return false;
    }

    return true;
}

void NvdecMjpegDecoder::close_decoder()
{
    impl_->reset();
}

bool NvdecMjpegDecoder::is_open() const { return impl_->opened; }

bool NvdecMjpegDecoder::is_supported() { return true; }

}  // namespace gpu_cam_minimal
