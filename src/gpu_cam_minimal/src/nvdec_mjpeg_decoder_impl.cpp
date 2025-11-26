#include "gpu_cam_minimal/nvdec_mjpeg_decoder_impl.hpp"
#include "gpu_cam_minimal/yuv2rgb.cuh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#include <drm/drm_fourcc.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <unistd.h>
#define EGL_EGLEXT_PROTOTYPES
#include "NvBufSurface.h"
#include "NvUtils.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <NvBuffer.h>
#include <NvVideoDecoder.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <libv4l2.h>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <rcutils/logging_macros.h>
#include <stdarg.h>

namespace gpu_cam_minimal {

namespace {

bool ensure_cuda_initialized() {
    static std::once_flag init_flag;
    static bool initialized = false;
    static CUresult init_result = CUDA_ERROR_NOT_INITIALIZED;
    std::call_once(init_flag, []() {
        init_result = cuInit(0);
        initialized = (init_result == CUDA_SUCCESS);
    });
    if (!initialized) {
        const char *err_name = nullptr;
        const char *err_str = nullptr;
        (void)cuGetErrorName(init_result, &err_name);
        (void)cuGetErrorString(init_result, &err_str);
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "cuInit failed: %s (%s)",
                                err_name ? err_name : "UNKNOWN",
                                err_str ? err_str : "no description");
    }
    return initialized;
}

CUcontext get_shared_cuda_context() {
    if (!ensure_cuda_initialized()) {
        return nullptr;
    }

    static std::once_flag ctx_flag;
    static CUcontext shared_ctx = nullptr;
    static CUdevice shared_device{};
    static CUresult ctx_result = CUDA_ERROR_NOT_INITIALIZED;
    std::call_once(ctx_flag, []() {
        CUdevice dev{};
        ctx_result = cuDeviceGet(&dev, 0);
        if (ctx_result == CUDA_SUCCESS) {
            shared_device = dev;
            CUresult flag_result =
                cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_AUTO);
            if (flag_result != CUDA_SUCCESS &&
                flag_result != CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE) {
                ctx_result = flag_result;
            }
        }
        if (ctx_result == CUDA_SUCCESS) {
            ctx_result = cuDevicePrimaryCtxRetain(&shared_ctx, shared_device);
        }
        if (ctx_result != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            const char *err_str = nullptr;
            (void)cuGetErrorName(ctx_result, &err_name);
            (void)cuGetErrorString(ctx_result, &err_str);
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder",
                "Failed to access CUDA primary context: %s (%s)",
                err_name ? err_name : "UNKNOWN",
                err_str ? err_str : "no description");
        }
    });
    if (ctx_result != CUDA_SUCCESS || shared_ctx == nullptr) {
        return nullptr;
    }
    return shared_ctx;
}

class ScopedCudaContext {
  public:
    ScopedCudaContext() {
        CUcontext ctx = get_shared_cuda_context();
        if (!ctx) {
            return;
        }
        CUresult res = cuCtxPushCurrent(ctx);
        if (res == CUDA_SUCCESS) {
            active_ = true;
        } else {
            const char *err_name = nullptr;
            const char *err_str = nullptr;
            (void)cuGetErrorName(res, &err_name);
            (void)cuGetErrorString(res, &err_str);
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "cuCtxPushCurrent failed: %s (%s)",
                                    err_name ? err_name : "UNKNOWN",
                                    err_str ? err_str : "no description");
        }
    }

    ~ScopedCudaContext() {
        if (!active_) {
            return;
        }
        CUcontext popped{};
        CUresult res = cuCtxPopCurrent(&popped);
        if (res != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            const char *err_str = nullptr;
            (void)cuGetErrorName(res, &err_name);
            (void)cuGetErrorString(res, &err_str);
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "cuCtxPopCurrent failed: %s (%s)",
                                    err_name ? err_name : "UNKNOWN",
                                    err_str ? err_str : "no description");
        }
    }

    bool valid() const { return active_; }

  private:
    bool active_{false};
};

} // namespace

// 简单的 JPEG 帧边界检测（FFD8 = SOI, FFD9 = EOI）
static inline bool is_jpeg_soi(const unsigned char *p) {
    return p[0] == 0xFF && p[1] == 0xD8;
}
static inline bool is_jpeg_eoi(const unsigned char *p) {
    return p[0] == 0xFF && p[1] == 0xD9;
}

bool NvdecMjpegDecoderImpl::initEglExtensions() {
    eglCreateImageKHR =
        (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
    if (!eglCreateImageKHR) {
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                "Failed to get eglCreateImageKHR");
        return false;
    }
    eglDestroyImageKHR =
        (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    if (!eglDestroyImageKHR) {
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                "Failed to get eglDestroyImageKHR");
        return false;
    }
    return true;
}

bool NvdecMjpegDecoderImpl::set_v4l2_mjpeg(int fd, int w, int h, double f) {
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = w;
    fmt.fmt.pix.height = h;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;
    if (v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                "Failed to set V4L2 MJPEG format: %s",
                                strerror(errno));
        return false;
    }
    if (f <= 0.0) {
        f = 30.0; // 默认帧率
    }
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<unsigned int>(f);
    int ret = v4l2_ioctl(fd, VIDIOC_S_PARM, &parm);
    if (ret < 0) {
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                "Failed to set V4L2 frame rate: %s",
                                strerror(errno));
        return false;
    }
    RCUTILS_LOG_INFO_NAMED("nvdec_mjpeg_decoder",
                           "V4L2 MJPEG format set: %dx%d @ %.2f FPS", w, h, f);
    return true;
}

bool NvdecMjpegDecoderImpl::grab_camera_frame(v4l2_buffer &out_vbuf,
                                              void *&out_data,
                                              size_t &out_len) {
    out_data = nullptr;
    out_len = 0;
    out_vbuf = {};
    out_vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    out_vbuf.memory = V4L2_MEMORY_MMAP;
    if (v4l2_ioctl(v4l2_fd, VIDIOC_DQBUF, &out_vbuf) < 0) {
        if (errno == EAGAIN) {
            return false; // no data right now
        }
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "VIDIOC_DQBUF failed: %s",
                               strerror(errno));
        return false;
    }
    out_len = static_cast<size_t>(out_vbuf.bytesused);
    if (out_vbuf.index < v4l2_bufs.size()) {
        out_data = v4l2_bufs[out_vbuf.index].start;
    }
    if (!out_data || out_len == 0) {
        RCUTILS_LOG_WARN_NAMED(
            "nvdec_mjpeg_decoder",
            "Empty or invalid V4L2 buffer (index=%u, len=%zu)",
            static_cast<unsigned int>(out_vbuf.index), out_len);
        (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &out_vbuf);
        return false;
    }
    return true;
}

NvBufSurfaceColorFormat
NvdecMjpegDecoderImpl::resolve_capture_color_format(uint32_t pixfmt) const {
    switch (pixfmt) {
    case V4L2_PIX_FMT_YUV420:
    case V4L2_PIX_FMT_YUV420M:
        return NVBUF_COLOR_FORMAT_YUV420;
    case V4L2_PIX_FMT_YUV422M:
    case V4L2_PIX_FMT_YUYV:
        return NVBUF_COLOR_FORMAT_YUV422;
    case V4L2_PIX_FMT_NV12:
    case V4L2_PIX_FMT_NV12M:
        return NVBUF_COLOR_FORMAT_NV12;
    case V4L2_PIX_FMT_NV24:
    case V4L2_PIX_FMT_NV24M:
        return NVBUF_COLOR_FORMAT_NV24;
    default:
        return NVBUF_COLOR_FORMAT_NV12;
    }
}

bool NvdecMjpegDecoderImpl::prepare_capture_dmabuf_buffer(v4l2_buffer &cbuf) {
    if (capture_dmabuf_fds.empty()) {
        return false;
    }
    if (cbuf.index >= capture_dmabuf_fds.size()) {
        return false;
    }

    cbuf.length = static_cast<uint32_t>(capture_num_planes);
    for (int p = 0; p < capture_num_planes && p < VIDEO_MAX_PLANES; ++p) {
        cbuf.m.planes[p].m.fd = capture_dmabuf_fds[cbuf.index];
        cbuf.m.planes[p].bytesused = 0;
        cbuf.m.planes[p].data_offset = 0;
        cbuf.m.planes[p].length = capture_plane_fmts[p].sizeimage;
    }
    return true;
}

bool NvdecMjpegDecoderImpl::requeue_capture_buffer(v4l2_buffer &cbuf) {
    if (!prepare_capture_dmabuf_buffer(cbuf)) {
        RCUTILS_LOG_WARN_NAMED(
            "nvdec_mjpeg_decoder",
            "Failed to prepare capture buffer %u for requeue", cbuf.index);
        return false;
    }

    if (dec->capture_plane.qBuffer(cbuf, nullptr) < 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                               "Failed to requeue capture buffer index %u",
                               cbuf.index);
        return false;
    }
    return true;
}

int NvdecMjpegDecoderImpl::get_capture_dmabuf_fd(uint32_t index) const {
    if (capture_dmabuf_fds.empty() || index >= capture_dmabuf_fds.size()) {
        return -1;
    }
    return capture_dmabuf_fds[index];
}

void NvdecMjpegDecoderImpl::start_output_reclaim_thread() {
    if (output_reclaim_future.valid() || !dec) {
        return;
    }
    output_reclaim_stop.store(false);
    output_reclaim_future = std::async(std::launch::async, [this]() {
        while (!output_reclaim_stop.load()) {
            v4l2_buffer obuf{};
            v4l2_plane oplanes[VIDEO_MAX_PLANES]{};
            obuf.m.planes = oplanes;
            if (!dec) {
                break;
            }
            // 使用 100ms 超时实现非阻塞检查
            int ret = dec->output_plane.dqBuffer(obuf, nullptr, nullptr, 100);
            if (ret < 0) {
                if (output_reclaim_stop.load()) {
                    break;
                }
                if (!dec->output_plane.getStreamStatus()) {
                    break;
                }
                if (errno == EAGAIN) {
                    continue;
                }
                RCUTILS_LOG_WARN_NAMED(
                    "nvdec_mjpeg_decoder",
                    "Output reclaim async dqBuffer failed (errno=%d)", errno);
                continue;
            }
            if (static_cast<size_t>(obuf.index) < 2) {
                std::lock_guard<std::mutex> lock(output_plane_mutex);
                out_in_use[obuf.index] = false;
            }
            output_plane_cv.notify_all();
        }
    });
}

void NvdecMjpegDecoderImpl::stop_output_reclaim_thread() {
    if (!output_reclaim_future.valid()) {
        output_reclaim_stop.store(false);
        return;
    }
    output_reclaim_stop.store(true);
    output_plane_cv.notify_all();

    // 等待最多 2 秒任务完成
    auto status = output_reclaim_future.wait_for(std::chrono::seconds(2));
    if (status == std::future_status::timeout) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                               "Output reclaim task did not finish within 2s, "
                               "abandoning (future will be destroyed)");
        // future 析构时会自动 detach，无需手动处理
    } else {
        // 任务已完成，获取结果以清理 future
        output_reclaim_future.get();
    }
    output_reclaim_stop.store(false);
}

bool NvdecMjpegDecoderImpl::feed_decoder_and_dequeue_capture(
    v4l2_buffer &cam_vbuf, const void *data, size_t len,
    NvBuffer *&out_cap_nvbuf, v4l2_buffer &out_cbuf) {
    out_cap_nvbuf = nullptr;

    auto submit_bitstream_dmabuf = [this](v4l2_buffer &cam_buf) -> bool {
        if (cam_buf.index >= v4l2_buf_fds.size()) {
            return false;
        }
        int fd = v4l2_buf_fds[cam_buf.index];
        if (fd < 0 || cam_buf.bytesused == 0) {
            return false;
        }

        bool requeued = false;
        auto requeue_camera = [&]() {
            if (!requeued) {
                if (v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_buf) < 0) {
                    RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                           "Failed to requeue camera buffer "
                                           "after DMABUF attempt: %s",
                                           strerror(errno));
                }
                requeued = true;
            }
        };

        v4l2_buffer obuf{};
        v4l2_plane oplanes[VIDEO_MAX_PLANES]{};
        obuf.m.planes = oplanes;
        obuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        obuf.memory = V4L2_MEMORY_DMABUF;
        obuf.length = 1;
        obuf.index = cam_buf.index;
        oplanes[0].m.fd = fd;
        oplanes[0].bytesused = cam_buf.bytesused;
        oplanes[0].length = cam_buf.length;
        if (dec->output_plane.qBuffer(obuf, nullptr) < 0) {
            RCUTILS_LOG_WARN_NAMED(
                "nvdec_mjpeg_decoder",
                "Failed to queue DMABUF to decoder output plane: %s",
                strerror(errno));
            requeue_camera();
            return false;
        }

        v4l2_buffer dqbuf{};
        v4l2_plane dqplanes[VIDEO_MAX_PLANES]{};
        dqbuf.m.planes = dqplanes;
        dqbuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        dqbuf.memory = V4L2_MEMORY_DMABUF;
        if (dec->output_plane.dqBuffer(dqbuf, nullptr, nullptr, -1) < 0) {
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                   "dqBuffer failed for DMABUF submission: %s",
                                   strerror(errno));
            requeue_camera();
            return false;
        }
        requeue_camera();
        return true;
    };

    bool used_dmabuf = false;
    if (camera_dmabuf_enabled && output_plane_memory == V4L2_MEMORY_DMABUF) {
        if (submit_bitstream_dmabuf(cam_vbuf)) {
            used_dmabuf = true;
        } else {
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                   "Falling back to memcpy path; DMABUF "
                                   "submission failed for buffer %u",
                                   static_cast<unsigned int>(cam_vbuf.index));
        }
    }

    int active_output_slot = -1;
    bool output_slot_reserved = false;

    auto release_output_slot = [this](int slot) {
        if (slot < 0 || slot >= 2) {
            return;
        }
        std::lock_guard<std::mutex> lock(output_plane_mutex);
        out_in_use[slot] = false;
        output_plane_cv.notify_all();
    };

    if (!used_dmabuf) {
        int idx = out_next_idx;
        if (idx < 0 || idx >= 2) {
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                   "Invalid out_next_idx: %d", idx);
            (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
            errno = EINVAL;
            return false;
        }

        {
            std::unique_lock<std::mutex> lock(output_plane_mutex);
            if (out_in_use[idx]) {
                constexpr auto wait_duration = std::chrono::milliseconds(10);
                if (!output_plane_cv.wait_for(
                        lock, wait_duration,
                        [this, idx]() { return !out_in_use[idx]; })) {
                    (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
                    RCUTILS_LOG_WARN_NAMED(
                        "nvdec_mjpeg_decoder",
                        "Timeout waiting for output slot %d to become "
                        "available (in_use[0]=%d, in_use[1]=%d).",
                        idx, static_cast<int>(out_in_use[0]),
                        static_cast<int>(out_in_use[1]));
                    errno = EAGAIN;
                    return false;
                }
            }
            out_in_use[idx] = true;
        }
        active_output_slot = idx;
        output_slot_reserved = true;

        NvBuffer *out_nvbuf = dec->output_plane.getNthBuffer(idx);
        if (!out_nvbuf) {
            RCUTILS_LOG_WARN_NAMED(
                "nvdec_mjpeg_decoder",
                "Failed to get NvBuffer from output plane (idx=%d).", idx);
            release_output_slot(idx);
            (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
            errno = EIO;
            return false;
        }

        const unsigned char *src = static_cast<const unsigned char *>(data);
        size_t src_len = len;
        ssize_t soi = -1, eoi = -1;
        for (size_t i = 0; i + 1 < src_len; ++i) {
            if (is_jpeg_soi(src + i)) {
                soi = static_cast<ssize_t>(i);
                break;
            }
        }
        if (soi >= 0) {
            for (size_t i = static_cast<size_t>(soi + 2); i + 1 < src_len;
                 ++i) {
                if (is_jpeg_eoi(src + i)) {
                    eoi = static_cast<ssize_t>(i + 2);
                    break;
                }
            }
        }
        size_t copy_len = 0;
        const unsigned char *copy_src = nullptr;
        if (soi >= 0 && eoi > soi) {
            copy_src = src + soi;
            copy_len = static_cast<size_t>(eoi - soi);
        } else {
            RCUTILS_LOG_WARN_NAMED(
                "nvdec_mjpeg_decoder",
                "MJPEG: SOI/EOI not found; dropping V4L2 buffer (%zu bytes)",
                src_len);
            release_output_slot(idx);
            (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
            errno = EPROTO;
            return false;
        }

        if (copy_len > static_cast<size_t>(out_nvbuf->planes[0].length)) {
            RCUTILS_LOG_WARN_NAMED(
                "nvdec_mjpeg_decoder",
                "Encoded JPEG too large for output buffer: %zu > %u", copy_len,
                out_nvbuf->planes[0].length);
            release_output_slot(idx);
            (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
            errno = EOVERFLOW;
            return false;
        }
        std::memcpy(out_nvbuf->planes[0].data, copy_src, copy_len);
        out_nvbuf->planes[0].bytesused = static_cast<uint32_t>(copy_len);

        v4l2_buffer obuf{};
        v4l2_plane oplanes[VIDEO_MAX_PLANES]{};
        obuf.m.planes = oplanes;
        obuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        obuf.memory = V4L2_MEMORY_MMAP;
        obuf.index = idx;
        obuf.m.planes[0].bytesused = static_cast<uint32_t>(copy_len);
        if (dec->output_plane.qBuffer(obuf, nullptr) < 0) {
            const int q_err = errno;
            RCUTILS_LOG_WARN_NAMED(
                "nvdec_mjpeg_decoder",
                "Failed to queue buffer to decoder output plane.");
            release_output_slot(idx);
            (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
            errno = q_err != 0 ? q_err : EIO;
            return false;
        }
        out_next_idx = 1 - out_next_idx;

        if (v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf) < 0) {
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                   "Failed to requeue V4L2 buffer: %s",
                                   strerror(errno));
        }
    }

    auto release_on_failure = [&]() {
        if (output_slot_reserved && active_output_slot >= 0) {
            release_output_slot(active_output_slot);
            output_slot_reserved = false;
        }
    };

    if (!capture_configured) {
        // 步骤5：当 capture 平面尚未完成配置时，监听事件并初始化 capture 端缓冲
        v4l2_event ev{};
        int evret = dec->dqEvent(ev, 0);
        if (evret == 0 && ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
            RCUTILS_LOG_INFO_NAMED(
                "nvdec_mjpeg_decoder",
                "Received V4L2_EVENT_RESOLUTION_CHANGE from decoder");
        }

        v4l2_format format{};
        int fmt_ret = dec->capture_plane.getFormat(format);
        if (fmt_ret < 0) {
            // capture 端尚未就绪，本帧释放 OUTPUT 槽位，稍后重试
            release_on_failure();
            frames_fed++;
            errno = EINVAL;
            return false;
        }
        dec_w = static_cast<int>(format.fmt.pix_mp.width);
        dec_h = static_cast<int>(format.fmt.pix_mp.height);
        capture_pixfmt = format.fmt.pix_mp.pixelformat;
        capture_num_planes = static_cast<int>(format.fmt.pix_mp.num_planes);
        RCUTILS_LOG_INFO_NAMED(
            "nvdec_mjpeg_decoder",
            "Decoder resolution finalized: %dx%d, pixfmt=0x%08x", dec_w, dec_h,
            capture_pixfmt);

        int32_t min_bufs = 0;
        if (dec->getMinimumCapturePlaneBuffers(min_bufs) < 0) {
            const int min_err = errno;
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "getMinimumCapturePlaneBuffers failed");
            release_on_failure();
            errno = min_err != 0 ? min_err : EIO;
            return false;
        }

        capture_num_buffers = static_cast<uint32_t>(min_bufs + 2);
        capture_dmabuf_fds.assign(capture_num_buffers, -1);
        release_capture_cuda_resources();

        NvBufSurf::NvCommonAllocateParams cap_params{};
        cap_params.memType = NVBUF_MEM_SURFACE_ARRAY;
        cap_params.width = static_cast<uint32_t>(dec_w);
        cap_params.height = static_cast<uint32_t>(dec_h);
        cap_params.layout = NVBUF_LAYOUT_PITCH;
        cap_params.colorFormat = resolve_capture_color_format(capture_pixfmt);
        cap_params.memtag = NvBufSurfaceTag_VIDEO_DEC;

        if (NvBufSurf::NvAllocate(&cap_params, capture_num_buffers,
                                  capture_dmabuf_fds.data()) < 0) {
            const int alloc_err = errno;
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder",
                "NvBufSurf::NvAllocate failed for capture plane");
            release_on_failure();
            errno = alloc_err != 0 ? alloc_err : ENOMEM;
            return false;
        }

        if (dec->capture_plane.reqbufs(V4L2_MEMORY_DMABUF,
                                       capture_num_buffers) < 0) {
            const int req_err = errno;
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "capture_plane.reqbufs failed");
            release_on_failure();
            errno = req_err != 0 ? req_err : EIO;
            return false;
        }

        if (dec->capture_plane.setStreamStatus(true) < 0) {
            const int stream_err = errno;
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "capture_plane.streamon failed");
            release_on_failure();
            errno = stream_err != 0 ? stream_err : EIO;
            return false;
        }

        for (auto &fmt : capture_plane_fmts) {
            std::memset(&fmt, 0, sizeof(fmt));
        }

        for (int p = 0; p < capture_num_planes && p < VIDEO_MAX_PLANES; ++p) {
            capture_plane_fmts[p] = format.fmt.pix_mp.plane_fmt[p];
        }

        for (uint32_t i = 0; i < capture_num_buffers; ++i) {
            v4l2_buffer cbuf{};
            v4l2_plane cplanes[VIDEO_MAX_PLANES]{};
            cbuf.m.planes = cplanes;
            cbuf.index = i;
            cbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            cbuf.memory = V4L2_MEMORY_DMABUF;

            if (!prepare_capture_dmabuf_buffer(cbuf)) {
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                        "Failed to prepare capture buffer %u",
                                        i);
                errno = EFAULT;
                return false;
            }

            if (dec->capture_plane.qBuffer(cbuf, nullptr) < 0) {
                const int cap_q_err = errno;
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                        "capture_plane.qBuffer failed");
                errno = cap_q_err != 0 ? cap_q_err : EIO;
                return false;
            }
        }
        capture_configured = true;
        RCUTILS_LOG_INFO_NAMED("nvdec_mjpeg_decoder",
                               "Capture plane configured with %u buffers.",
                               dec->capture_plane.getNumBuffers());
    }

    // 步骤6：从 capture 平面取出一帧解码后的输出并返回
    if (dec->capture_plane.dqBuffer(out_cbuf, &out_cap_nvbuf, nullptr, -1) <
        0) {
        const int dq_err = errno;
        RCUTILS_LOG_WARN_NAMED(
            "nvdec_mjpeg_decoder",
            "Timeout or failure dequeuing from capture plane.");
        errno = dq_err != 0 ? dq_err : EIO;
        return false;
    }
    int queued_fd = get_capture_dmabuf_fd(out_cbuf.index);

    (void)queued_fd;
    return true;
}

bool NvdecMjpegDecoderImpl::convert_capture_to_rgb(NvBuffer *cap_nvbuf,
                                                   v4l2_buffer &cbuf,
                                                   cv::cuda::GpuMat &out_rgb) {
    (void)cap_nvbuf;
    if (!ensure_capture_cuda_resource(cbuf.index)) {
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    auto &resource = capture_cuda_resources[cbuf.index];
    if (!resource.surface || resource.egl_image == EGL_NO_IMAGE_KHR) {
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    ScopedCudaContext ctx_guard;
    if (!ctx_guard.valid()) {
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    if (!resource.cuda_resource) {
        CUresult reg_result = cuGraphicsEGLRegisterImage(
            &resource.cuda_resource, resource.egl_image,
            CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
        if (reg_result != CUDA_SUCCESS) {
            const char *err_name = nullptr;
            const char *err_desc = nullptr;
            (void)cuGetErrorName(reg_result, &err_name);
            (void)cuGetErrorString(reg_result, &err_desc);
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                   "Failed to register cached EGLImage to CUDA "
                                   "(index=%u, CUresult=%d: %s - %s)",
                                   static_cast<unsigned int>(cbuf.index),
                                   static_cast<int>(reg_result),
                                   err_name ? err_name : "UNKNOWN",
                                   err_desc ? err_desc : "no description");
            (void)requeue_capture_buffer(cbuf);
            return false;
        }
    }

    CUeglFrame eglFrame{};
    if (cuGraphicsResourceGetMappedEglFrame(&eglFrame, resource.cuda_resource,
                                            0, 0) != CUDA_SUCCESS) {
        RCUTILS_LOG_WARN_NAMED(
            "nvdec_mjpeg_decoder",
            "Failed to map CUeglFrame from cached CUDA resource.");
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    int W = dec_w > 0 ? dec_w : width;
    int H = dec_h > 0 ? dec_h : height;

    bool converted = false;
    do {
        if (capture_pixfmt != V4L2_PIX_FMT_YUV422M) {
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder",
                "Unsupported capture pixelformat 0x%08x for RGB conversion",
                capture_pixfmt);
            break;
        }

        capture_cuda_resources.resize(capture_num_buffers);
        if (eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
            RCUTILS_LOG_ERROR_NAMED(
                "nvdec_mjpeg_decoder",
                "Unexpected CUeglFrame type %d (expected PITCH)",
                static_cast<int>(eglFrame.frameType));
            break;
        }

        const NvBufSurfaceParams &surf_params =
            resource.surface->surfaceList[0];
        const NvBufSurfacePlaneParams &plane_params = surf_params.planeParams;
        if (plane_params.num_planes < 3) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "NvBufSurface plane count %u < 3",
                                    plane_params.num_planes);
            break;
        }

        size_t y_pitch = plane_params.pitch[0];
        size_t u_pitch = plane_params.pitch[1];
        size_t v_pitch = plane_params.pitch[2];
        const size_t fallback_pitch = static_cast<size_t>(eglFrame.pitch);
        if (y_pitch == 0) {
            y_pitch = fallback_pitch;
        }
        if (u_pitch == 0) {
            u_pitch = fallback_pitch;
        }
        if (v_pitch == 0) {
            v_pitch = fallback_pitch;
        }

        const unsigned char *y_plane =
            static_cast<const unsigned char *>(eglFrame.frame.pPitch[0]);
        const unsigned char *u_plane =
            static_cast<const unsigned char *>(eglFrame.frame.pPitch[1]);
        const unsigned char *v_plane =
            static_cast<const unsigned char *>(eglFrame.frame.pPitch[2]);

        if (out_rgb.empty() || out_rgb.rows != H || out_rgb.cols != W ||
            out_rgb.type() != CV_8UC3) {
            out_rgb.create(H, W, CV_8UC3);
        }

        cudaStream_t stream = 0;
        cudaError_t err = gpuConvertYUV422MToRGB(
            y_plane, u_plane, v_plane, out_rgb.ptr<unsigned char>(), y_pitch,
            u_pitch, v_pitch, static_cast<size_t>(out_rgb.step),
            static_cast<unsigned int>(W), static_cast<unsigned int>(H), stream);
        if (err != cudaSuccess) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "gpuConvertYUV422MToRGB failed: %s",
                                    cudaGetErrorString(err));
            break;
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder",
                                    "cudaStreamSynchronize failed: %s",
                                    cudaGetErrorString(err));
            break;
        }

        converted = true;
    } while (false);

    if (!requeue_capture_buffer(cbuf)) {
        return false;
    }

    return converted;
}

int NvdecMjpegDecoderImpl::get_camera_dmabuf_fd(uint32_t index) const {
    if (!camera_dmabuf_enabled || index >= v4l2_buf_fds.size()) {
        return -1;
    }
    return v4l2_buf_fds[index];
}

bool NvdecMjpegDecoderImpl::ensure_capture_cuda_resource(uint32_t index) {
    if (index >= capture_dmabuf_fds.size() ||
        index >= capture_cuda_resources.size()) {
        return false;
    }
    int fd = capture_dmabuf_fds[index];
    if (fd < 0) {
        return false;
    }

    auto &resource = capture_cuda_resources[index];
    if (!resource.surface) {
        if (NvBufSurfaceFromFd(
                fd, reinterpret_cast<void **>(&resource.surface)) != 0 ||
            resource.surface == nullptr) {
            resource.surface = nullptr;
            RCUTILS_LOG_WARN_NAMED(
                "nvdec_mjpeg_decoder",
                "NvBufSurfaceFromFd failed for capture buffer %u",
                static_cast<unsigned int>(index));
            return false;
        }
        if (!resource.surface->surfaceList ||
            resource.surface->batchSize == 0) {
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                   "NvBufSurface invalid for capture buffer %u",
                                   static_cast<unsigned int>(index));
            resource.surface = nullptr;
            return false;
        }
    }

    if (!resource.egl_mapped) {
        if (NvBufSurfaceMapEglImage(resource.surface, 0) != 0) {
            RCUTILS_LOG_WARN_NAMED(
                "nvdec_mjpeg_decoder",
                "NvBufSurfaceMapEglImage failed for capture buffer %u",
                static_cast<unsigned int>(index));
            return false;
        }
        resource.egl_mapped = true;
        resource.egl_image = static_cast<EGLImageKHR>(
            resource.surface->surfaceList[0].mappedAddr.eglImage);
        if (resource.egl_image == EGL_NO_IMAGE_KHR) {
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                                   "Invalid EGL image for capture buffer %u",
                                   static_cast<unsigned int>(index));
            return false;
        }
    }

    return true;
}

void NvdecMjpegDecoderImpl::release_capture_cuda_resources() {
    ScopedCudaContext ctx_guard;
    const bool cuda_ctx_ready = ctx_guard.valid();
    for (auto &resource : capture_cuda_resources) {
        if (resource.cuda_resource) {
            if (cuda_ctx_ready) {
                cuGraphicsUnregisterResource(resource.cuda_resource);
            }
            resource.cuda_resource = nullptr;
        }
        if (resource.surface && resource.egl_mapped) {
            NvBufSurfaceUnMapEglImage(resource.surface, 0);
            resource.egl_mapped = false;
        }
        resource.surface = nullptr;
        resource.egl_image = EGL_NO_IMAGE_KHR;
    }
    capture_cuda_resources.clear();
}

} // namespace gpu_cam_minimal
