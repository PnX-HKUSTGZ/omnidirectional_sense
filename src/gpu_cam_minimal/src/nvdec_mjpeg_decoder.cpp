#include "gpu_cam_minimal/nvdec_mjpeg_decoder.hpp"

#include <vector>
#include <thread>
#include <mutex>
#include <cstring>
#include <unistd.h>


#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <drm/drm_fourcc.h>
#define EGL_EGLEXT_PROTOTYPES   
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <NvVideoDecoder.h>
#include <cuda.h>
#include <cudaEGL.h>
#include "NvUtils.h"
#include <stdarg.h>
#include <libv4l2.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <rcutils/logging_macros.h>
#include <NvBuffer.h>
#include "NvBufSurface.h"
#include "gpu_cam_minimal/yuv2rgb.cuh"

namespace gpu_cam_minimal {

namespace {

bool ensure_cuda_initialized()
{
    static std::once_flag init_flag;
    static bool initialized = false;
    static CUresult init_result = CUDA_ERROR_NOT_INITIALIZED;
    std::call_once(init_flag, []() {
        init_result = cuInit(0);
        initialized = (init_result == CUDA_SUCCESS);
    });
    if (!initialized) {
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        (void)cuGetErrorName(init_result, &err_name);
        (void)cuGetErrorString(init_result, &err_str);
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "cuInit failed: %s (%s)",
                                err_name ? err_name : "UNKNOWN",
                                err_str ? err_str : "no description");
    }
    return initialized;
}

bool ensure_cuda_context()
{
    if (!ensure_cuda_initialized()) {
        return false;
    }

    CUcontext current = nullptr;
    CUresult get_result = cuCtxGetCurrent(&current);
    if (get_result == CUDA_SUCCESS && current != nullptr) {
        return true;
    }

    static std::once_flag ctx_flag;
    static CUcontext shared_ctx = nullptr;
    static CUresult ctx_result = CUDA_ERROR_NOT_INITIALIZED;
    std::call_once(ctx_flag, []() {
        CUdevice dev{};
        ctx_result = cuDeviceGet(&dev, 0);
        if (ctx_result == CUDA_SUCCESS) {
            ctx_result = cuCtxCreate(&shared_ctx, CU_CTX_SCHED_AUTO, dev);
        }
    });
    if (ctx_result != CUDA_SUCCESS || shared_ctx == nullptr) {
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        (void)cuGetErrorName(ctx_result, &err_name);
        (void)cuGetErrorString(ctx_result, &err_str);
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to create CUDA context: %s (%s)",
                                err_name ? err_name : "UNKNOWN",
                                err_str ? err_str : "no description");
        return false;
    }

    CUresult set_result = cuCtxSetCurrent(shared_ctx);
    if (set_result != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        (void)cuGetErrorName(set_result, &err_name);
        (void)cuGetErrorString(set_result, &err_str);
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "cuCtxSetCurrent failed: %s (%s)",
                                err_name ? err_name : "UNKNOWN",
                                err_str ? err_str : "no description");
        return false;
    }

    return true;
}

} // namespace


// 简单的 JPEG 帧边界检测（FFD8 = SOI, FFD9 = EOI）
static inline bool is_jpeg_soi(const unsigned char* p) {
    return p[0] == 0xFF && p[1] == 0xD8;
}
static inline bool is_jpeg_eoi(const unsigned char* p) {
    return p[0] == 0xFF && p[1] == 0xD9;
}

struct NvdecMjpegDecoder::Impl {
    // 通用成员（即使在不支持时也存在）
    std::string device;
    int width{0};
    int height{0};
    double fps{0.0};
    bool opened{false};
    int v4l2_fd{-1};
    bool v4l2_streaming{false};
    struct V4L2Buffer { void* start{nullptr}; size_t length{0}; };
    std::vector<V4L2Buffer> v4l2_bufs; // MMAP buffers for compressed MJPEG frames
    NvVideoDecoder* dec{nullptr};
    EGLDisplay egl_display{EGL_NO_DISPLAY};
    std::vector<unsigned char> enc_buf;
    bool capture_configured{false};
    int dec_w{0};
    int dec_h{0};
    int capture_num_planes{0};
    uint32_t capture_num_buffers{0};
    v4l2_plane_pix_format capture_plane_fmts[VIDEO_MAX_PLANES]{};
    std::vector<int> capture_dmabuf_fds;
    int out_next_idx{0};
    bool out_in_use[2]{false, false};
    int frames_fed{0}; // 已喂给 NVDEC 的输出帧数量，用于无事件回退策略
    uint32_t capture_pixfmt{0};

    // EGL / CUDA helper
    PFNEGLCREATEIMAGEKHRPROC  eglCreateImageKHR{nullptr};
    PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR{nullptr};

    // 初始化 EGL 函数指针
    bool initEglExtensions() {
        eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC) eglGetProcAddress("eglCreateImageKHR");
        if (!eglCreateImageKHR) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to get eglCreateImageKHR");
            return false;
        }
        eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC) eglGetProcAddress("eglDestroyImageKHR");
        if (!eglDestroyImageKHR) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to get eglDestroyImageKHR");
            return false;
        }
        return true;
    }

    // helper member functions (implemented below)
    bool grab_camera_frame(v4l2_buffer &out_vbuf, void*& out_data, size_t& out_len);
    bool feed_decoder_and_dequeue_capture(v4l2_buffer &cam_vbuf, const void* data, size_t len,
                                         NvBuffer*& out_cap_nvbuf, v4l2_buffer &out_cbuf);
    bool convert_capture_to_rgb(NvBuffer* cap_nvbuf, v4l2_buffer &cbuf, cv::cuda::GpuMat &out_rgb);
    bool prepare_capture_dmabuf_buffer(v4l2_buffer &cbuf);
    NvBufSurfaceColorFormat resolve_capture_color_format(uint32_t pixfmt) const;
    bool requeue_capture_buffer(v4l2_buffer &cbuf);
    int get_capture_dmabuf_fd(uint32_t index) const;

    static bool set_v4l2_mjpeg(int fd, int w, int h, double f)
    {
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = w;
        fmt.fmt.pix.height = h;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;
        if (v4l2_ioctl(fd, VIDIOC_S_FMT, &fmt) < 0) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to set V4L2 MJPEG format: %s", strerror(errno));
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
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to set V4L2 frame rate: %s", strerror(errno));
            return false;
        }
        RCUTILS_LOG_INFO_NAMED("nvdec_mjpeg_decoder", "V4L2 MJPEG format set: %dx%d @ %.2f FPS", w, h, f);
        return true;
    }
};

// Helper: grab a compressed MJPEG frame from the V4L2 camera (DQBUF).
bool NvdecMjpegDecoder::Impl::grab_camera_frame(v4l2_buffer &out_vbuf, void*& out_data, size_t& out_len)
{
    out_data = nullptr;
    out_len = 0;
    out_vbuf = {};
    out_vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    out_vbuf.memory = V4L2_MEMORY_MMAP;
    if (v4l2_ioctl(v4l2_fd, VIDIOC_DQBUF, &out_vbuf) < 0) {
        if (errno == EAGAIN) {
            return false; // no data right now
        }
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "VIDIOC_DQBUF failed: %s", strerror(errno));
        return false;
    }
    out_len = static_cast<size_t>(out_vbuf.bytesused);
    if (out_vbuf.index < v4l2_bufs.size()) {
        out_data = v4l2_bufs[out_vbuf.index].start;
    }
    if (!out_data || out_len == 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Empty or invalid V4L2 buffer (index=%u, len=%zu)",
                               static_cast<unsigned int>(out_vbuf.index), out_len);
        // requeue to keep camera working
        (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &out_vbuf);
        return false;
    }
    return true;
}

NvBufSurfaceColorFormat NvdecMjpegDecoder::Impl::resolve_capture_color_format(uint32_t pixfmt) const
{
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

bool NvdecMjpegDecoder::Impl::prepare_capture_dmabuf_buffer(v4l2_buffer &cbuf)
{
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

bool NvdecMjpegDecoder::Impl::requeue_capture_buffer(v4l2_buffer &cbuf)
{
    if (!prepare_capture_dmabuf_buffer(cbuf)) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Failed to prepare capture buffer %u for requeue", cbuf.index);
        return false;
    }

    if (dec->capture_plane.qBuffer(cbuf, nullptr) < 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Failed to requeue capture buffer index %u", cbuf.index);
        return false;
    }
    return true;
}

int NvdecMjpegDecoder::Impl::get_capture_dmabuf_fd(uint32_t index) const
{
    if (capture_dmabuf_fds.empty() || index >= capture_dmabuf_fds.size()) {
        return -1;
    }
    return capture_dmabuf_fds[index];
}

// Helper: feed compressed JPEG data into NVDEC output plane, configure capture plane if needed,
// and dequeue one decoded buffer from capture plane. Returns true and fills cap_nvbuf/cbuf on success.
bool NvdecMjpegDecoder::Impl::feed_decoder_and_dequeue_capture(v4l2_buffer &cam_vbuf,
                                                               const void* data, size_t len,
                                                               NvBuffer*& out_cap_nvbuf, v4l2_buffer &out_cbuf)
{
    out_cap_nvbuf = nullptr;
    // choose output index (ping-pong)
    int idx = out_next_idx;
    if (idx < 0 || idx >= 2) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Invalid out_next_idx: %d", idx);
        // requeue camera buffer
        (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
        return false;
    }

    if (out_in_use[idx]) {
        v4l2_buffer obuf{}; v4l2_plane oplanes[VIDEO_MAX_PLANES]{}; obuf.m.planes = oplanes;
        if (dec->output_plane.dqBuffer(obuf, nullptr, nullptr, -1) < 0) {
            RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Failed to dq from output plane.");
            (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
            return false;
        }
        if (static_cast<size_t>(obuf.index) < 2) {
            out_in_use[obuf.index] = false;
        }
    }

    // get NvBuffer for output plane and copy compressed JPEG into it
    NvBuffer* out_nvbuf = dec->output_plane.getNthBuffer(idx);
    if (!out_nvbuf) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Failed to get NvBuffer from output plane (idx=%d).", idx);
        (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
        return false;
    }

    // find SOI/EOI within the camera buffer and copy that region (fallback to full buffer)
    const unsigned char* src = static_cast<const unsigned char*>(data);
    size_t src_len = len;
    ssize_t soi = -1, eoi = -1;
    for (size_t i = 0; i + 1 < src_len; ++i) {
        if (is_jpeg_soi(src + i)) { soi = static_cast<ssize_t>(i); break; }
    }
    if (soi >= 0) {
        for (size_t i = static_cast<size_t>(soi + 2); i + 1 < src_len; ++i) {
            if (is_jpeg_eoi(src + i)) { eoi = static_cast<ssize_t>(i + 2); break; }
        }
    }
    size_t copy_len = 0;
    const unsigned char* copy_src = nullptr;
    if (soi >= 0 && eoi > soi) {
        copy_src = src + soi;
        copy_len = static_cast<size_t>(eoi - soi);
    } else {
        copy_src = src;
        copy_len = src_len;
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "MJPEG: SOI/EOI not found; using full V4L2 buffer (%zu bytes)", copy_len);
    }

    if (copy_len > static_cast<size_t>(out_nvbuf->planes[0].length)) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Encoded JPEG too large for output buffer: %zu > %u",
                               copy_len, out_nvbuf->planes[0].length);
        (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
        return false;
    }
    std::memcpy(out_nvbuf->planes[0].data, copy_src, copy_len);
    out_nvbuf->planes[0].bytesused = static_cast<uint32_t>(copy_len);

    // queue to decoder output plane
    v4l2_buffer obuf{}; v4l2_plane oplanes[VIDEO_MAX_PLANES]{}; obuf.m.planes = oplanes;
    obuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    obuf.memory = V4L2_MEMORY_MMAP;
    obuf.index = idx;
    obuf.m.planes[0].bytesused = static_cast<uint32_t>(copy_len);
    if (dec->output_plane.qBuffer(obuf, nullptr) < 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Failed to queue buffer to decoder output plane.");
        (void)v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf);
        return false;
    }
    out_in_use[idx] = true;
    out_next_idx = 1 - out_next_idx;

    // requeue camera buffer immediately
    if (v4l2_ioctl(v4l2_fd, VIDIOC_QBUF, &cam_vbuf) < 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Failed to requeue V4L2 buffer: %s", strerror(errno));
    }

    // configure capture plane on first successful parse (prefer event; fallback to getFormat; caller handles retry/backoff)
    if (!capture_configured) {
        // Try to dequeue a resolution-change event first (non-blocking)
        v4l2_event ev{};
        int evret = dec->dqEvent(ev, 0);
        if (evret == 0 && ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
            RCUTILS_LOG_INFO_NAMED("nvdec_mjpeg_decoder", "Received V4L2_EVENT_RESOLUTION_CHANGE from decoder");
        }

        v4l2_format format{};
        int fmt_ret = dec->capture_plane.getFormat(format);
        if (fmt_ret < 0) {
            // Decoder may still be parsing headers; let caller retry/backoff
            frames_fed++;
            return false;
        }
        dec_w = static_cast<int>(format.fmt.pix_mp.width);
        dec_h = static_cast<int>(format.fmt.pix_mp.height);
        capture_pixfmt = format.fmt.pix_mp.pixelformat;
        capture_num_planes = static_cast<int>(format.fmt.pix_mp.num_planes);
        RCUTILS_LOG_INFO_NAMED("nvdec_mjpeg_decoder", "Decoder resolution finalized: %dx%d, pixfmt=0x%08x", dec_w, dec_h, capture_pixfmt);

        int32_t min_bufs = 0;
        if (dec->getMinimumCapturePlaneBuffers(min_bufs) < 0) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "getMinimumCapturePlaneBuffers failed");
            return false;
        }

        capture_num_buffers = static_cast<uint32_t>(min_bufs + 2);
        capture_dmabuf_fds.assign(capture_num_buffers, -1);

        NvBufSurf::NvCommonAllocateParams cap_params{};
        cap_params.memType = NVBUF_MEM_SURFACE_ARRAY;
        cap_params.width = static_cast<uint32_t>(dec_w);
        cap_params.height = static_cast<uint32_t>(dec_h);
        cap_params.layout = NVBUF_LAYOUT_PITCH;
        cap_params.colorFormat = resolve_capture_color_format(capture_pixfmt);
        cap_params.memtag = NvBufSurfaceTag_VIDEO_DEC;

        if (NvBufSurf::NvAllocate(&cap_params, capture_num_buffers, capture_dmabuf_fds.data()) < 0) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "NvBufSurf::NvAllocate failed for capture plane");
            return false;
        }

        if (dec->capture_plane.reqbufs(V4L2_MEMORY_DMABUF, capture_num_buffers) < 0) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "capture_plane.reqbufs failed");
            return false;
        }

        if (dec->capture_plane.setStreamStatus(true) < 0) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "capture_plane.streamon failed");
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
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to prepare capture buffer %u", i);
                return false;
            }

            if (dec->capture_plane.qBuffer(cbuf, nullptr) < 0) {
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "capture_plane.qBuffer failed");
                return false;
            }
        }
        capture_configured = true;
        RCUTILS_LOG_INFO_NAMED("nvdec_mjpeg_decoder", "Capture plane configured with %u buffers.", dec->capture_plane.getNumBuffers());
    }

    // dequeue a decoded buffer from capture plane
    if (dec->capture_plane.dqBuffer(out_cbuf, &out_cap_nvbuf, nullptr, -1) < 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Timeout or failure dequeuing from capture plane.");
        return false;
    }
    int queued_fd = get_capture_dmabuf_fd(out_cbuf.index);

    return true;
}

// Helper: convert a captured NvBuffer to CUDA GpuMat RGB frame. Handles EGL/CUDA registration and color conversion.
bool NvdecMjpegDecoder::Impl::convert_capture_to_rgb(NvBuffer* cap_nvbuf, v4l2_buffer &cbuf, cv::cuda::GpuMat &out_rgb)
{
    (void)cap_nvbuf;
    int dmabuf_fd = get_capture_dmabuf_fd(cbuf.index);
    if (dmabuf_fd < 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Invalid DMABUF fd for capture buffer %u", cbuf.index);
        return false;
    }

    NvBufSurface* nvbuf_surf = nullptr;
    if (NvBufSurfaceFromFd(dmabuf_fd, reinterpret_cast<void**>(&nvbuf_surf)) != 0 || nvbuf_surf == nullptr) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "NvBufSurfaceFromFd failed for fd=%d", dmabuf_fd);
        return false;
    }
    if (!nvbuf_surf->surfaceList || nvbuf_surf->batchSize == 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "NvBufSurfaceFromFd returned empty surface for fd=%d", dmabuf_fd);
        return false;
    }

    if (NvBufSurfaceMapEglImage(nvbuf_surf, 0) != 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "NvBufSurfaceMapEglImage failed for fd=%d", dmabuf_fd);
        return false;
    }

    const auto unmap_egl_image = [nvbuf_surf]() {
        if (nvbuf_surf) {
            NvBufSurfaceUnMapEglImage(nvbuf_surf, 0);
        }
    };

    EGLImageKHR egl_image = static_cast<EGLImageKHR>(nvbuf_surf->surfaceList[0].mappedAddr.eglImage);

    if (egl_image == EGL_NO_IMAGE_KHR) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                               "eglCreateImageKHR(NV12) failed for dmabuf_fd=%d", dmabuf_fd);
        unmap_egl_image();
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    if (!ensure_cuda_context()) {
        eglDestroyImageKHR(egl_display, egl_image);
        unmap_egl_image();
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    CUgraphicsResource cuda_resource{};
    CUresult reg_result = cuGraphicsEGLRegisterImage(&cuda_resource, egl_image, CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE);
    if (reg_result != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_desc = nullptr;
        (void)cuGetErrorName(reg_result, &err_name);
        (void)cuGetErrorString(reg_result, &err_desc);
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder",
                               "Failed to register EGLImage to CUDA (dmabuf_fd=%d, EGLImage=%p, CUresult=%d: %s - %s)",
                               dmabuf_fd,
                               static_cast<void*>(egl_image),
                               static_cast<int>(reg_result),
                               err_name ? err_name : "UNKNOWN",
                               err_desc ? err_desc : "no description");
        eglDestroyImageKHR(egl_display, egl_image);
        unmap_egl_image();
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    CUeglFrame eglFrame{};
    if (cuGraphicsResourceGetMappedEglFrame(&eglFrame, cuda_resource, 0, 0) != CUDA_SUCCESS) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Failed to map CUeglFrame from EGLImage.");
        cuGraphicsUnregisterResource(cuda_resource);
        eglDestroyImageKHR(egl_display, egl_image);
        unmap_egl_image();
        (void)requeue_capture_buffer(cbuf);
        return false;
    }

    int W = dec_w > 0 ? dec_w : width;
    int H = dec_h > 0 ? dec_h : height;

    bool converted = false;
    do {
        if (capture_pixfmt != V4L2_PIX_FMT_YUV422M) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Unsupported capture pixelformat 0x%08x for RGB conversion", capture_pixfmt);
            break;
        }

        if (eglFrame.frameType != CU_EGL_FRAME_TYPE_PITCH) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Unexpected CUeglFrame type %d (expected PITCH)", static_cast<int>(eglFrame.frameType));
            break;
        }

        const NvBufSurfaceParams &surf_params = nvbuf_surf->surfaceList[0];
        const NvBufSurfacePlaneParams &plane_params = surf_params.planeParams;
        if (plane_params.num_planes < 3) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "NvBufSurface plane count %u < 3", plane_params.num_planes);
            break;
        }

        size_t y_pitch = plane_params.pitch[0];
        size_t u_pitch = plane_params.pitch[1];
        size_t v_pitch = plane_params.pitch[2];
        const size_t fallback_pitch = static_cast<size_t>(eglFrame.pitch);
        if (y_pitch == 0) { y_pitch = fallback_pitch; }
        if (u_pitch == 0) { u_pitch = fallback_pitch; }
        if (v_pitch == 0) { v_pitch = fallback_pitch; }

        const unsigned char *y_plane = static_cast<const unsigned char*>(eglFrame.frame.pPitch[0]);
        const unsigned char *u_plane = static_cast<const unsigned char*>(eglFrame.frame.pPitch[1]);
        const unsigned char *v_plane = static_cast<const unsigned char*>(eglFrame.frame.pPitch[2]);

        if (out_rgb.empty() || out_rgb.rows != H || out_rgb.cols != W || out_rgb.type() != CV_8UC3) {
            out_rgb.create(H, W, CV_8UC3);
        }

        cudaStream_t stream = 0;
        cudaError_t err = gpuConvertYUV422MToRGB(
                y_plane,
                u_plane,
                v_plane,
                out_rgb.ptr<unsigned char>(),
                y_pitch,
                u_pitch,
                v_pitch,
                static_cast<size_t>(out_rgb.step),
                static_cast<unsigned int>(W),
                static_cast<unsigned int>(H),
                stream);
        if (err != cudaSuccess) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "gpuConvertYUV422MToRGB failed: %s", cudaGetErrorString(err));
            break;
        }

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "cudaStreamSynchronize failed: %s", cudaGetErrorString(err));
            break;
        }

        converted = true;
    } while (false);

    // cleanup
    cuGraphicsUnregisterResource(cuda_resource);
    eglDestroyImageKHR(egl_display, egl_image);
    unmap_egl_image();

    // requeue capture buffer
    if (!requeue_capture_buffer(cbuf)) {
        return false;
    }

    return converted;
}


NvdecMjpegDecoder::NvdecMjpegDecoder() : impl_(new Impl) {}
NvdecMjpegDecoder::~NvdecMjpegDecoder() { close_decoder(); }

bool NvdecMjpegDecoder::open(const std::string& video_device, int width, int height, double fps)
{
    impl_->device = video_device;
    impl_->width  = width;
    impl_->height = height;
    impl_->fps    = fps;

    // 打开 V4L2 camera（MJPEG bitstream）
    impl_->v4l2_fd = ::open(video_device.c_str(), O_RDWR | O_NONBLOCK);
    if (impl_->v4l2_fd < 0) {
        impl_->opened = false;
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to open V4L2 device %s: %s",
                                video_device.c_str(), strerror(errno));
        return false;
    }
    
    if (!impl_->initEglExtensions()) {
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Failed to initialize EGL KHR extensions");
        return false;
    }


    // 尽力设置分辨率
    (void)Impl::set_v4l2_mjpeg(impl_->v4l2_fd, width, height, fps);

    // 确认支持 STREAMING 能力
    v4l2_capability cap{};
    if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_QUERYCAP, &cap) == 0) {
        if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "V4L2 device does not support STREAMING API");
            close_decoder();
            return false;
        }
    }

    // 初始化 V4L2 MMAP 缓冲并开启 STREAMON
    {
        v4l2_requestbuffers req{};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_REQBUFS, &req) < 0) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "VIDIOC_REQBUFS failed: %s", strerror(errno));
            close_decoder();
            return false;
        }
        if (req.count < 2) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "Insufficient V4L2 buffers allocated: %u", req.count);
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
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "VIDIOC_QUERYBUF failed: %s", strerror(errno));
                close_decoder();
                return false;
            }
            void* start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, impl_->v4l2_fd, buf.m.offset);
            if (start == MAP_FAILED) {
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "mmap failed: %s", strerror(errno));
                close_decoder();
                return false;
            }
            impl_->v4l2_bufs[i].start = start;
            impl_->v4l2_bufs[i].length = buf.length;

            if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_QBUF, &buf) < 0) {
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "VIDIOC_QBUF failed: %s", strerror(errno));
                close_decoder();
                return false;
            }
        }
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (v4l2_ioctl(impl_->v4l2_fd, VIDIOC_STREAMON, &type) < 0) {
            RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "VIDIOC_STREAMON failed: %s", strerror(errno));
            close_decoder();
            return false;
        }
        impl_->v4l2_streaming = true;
    }

    // ---- 创建 Jetson NVDEC MJPEG 解码器 ----
    impl_->dec = NvVideoDecoder::createVideoDecoder("dec0");
    if (!impl_->dec) {
        close_decoder();
        return false;
    }

    // 订阅分辨率变化事件
    if (impl_->dec->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0) < 0) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE) failed; will use fallback without event.");
    }

    // 设置 OUTPUT 平面格式（输入单帧 JPEG 码流）
    // 使用 V4L2_PIX_FMT_MJPEG 能在部分 Jetson NVDEC 版本上更稳定触发内部解析，避免后续 capture_plane.getFormat EINVAL。
    if (impl_->dec->setOutputPlaneFormat(V4L2_PIX_FMT_MJPEG, 2 * 1024 * 1024) < 0) {
        RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "setOutputPlaneFormat(V4L2_PIX_FMT_MJPEG) failed");
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



bool NvdecMjpegDecoder::read_rgb(cv::cuda::GpuMat& out_rgb)
{
    if (impl_->v4l2_fd < 0 || !impl_->dec) {
        RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "Invalid decoder or v4l2_fd not opened.");
        return false;
    }

    // 1) 抓取一帧摄像头的 MJPEG 压缩数据
    v4l2_buffer vbuf{};
    void* cam_data = nullptr;
    size_t cam_len = 0;
    if (!impl_->grab_camera_frame(vbuf, cam_data, cam_len)) {
        // helper 内部已做必要日志与回队（当需要时）。
        return false;
    }
    

    // 2) 将压缩数据喂给 NVDEC 并从 capture 平面取一帧解码输出
    v4l2_buffer cbuf{}; 
    v4l2_plane cplanes[VIDEO_MAX_PLANES]{}; cbuf.m.planes = cplanes;
    NvBuffer* cap_nvbuf = nullptr;
    if (!impl_->feed_decoder_and_dequeue_capture(vbuf, cam_data, cam_len, cap_nvbuf, cbuf)) {
        // 与旧实现保持一致：若还未完成 capture 配置，前若干次（<=10）在常见 errno 下短暂等待并返回 true 让上层继续循环
        if (!impl_->capture_configured) {
            const int max_try = 10;
            if ((errno == EINVAL || errno == EIO || errno == EAGAIN) && impl_->frames_fed <= max_try) {
                RCUTILS_LOG_WARN_NAMED("nvdec_mjpeg_decoder", "JPEG: waiting decoder to finalize format (try %d/%d, errno=%d)", impl_->frames_fed, max_try, errno);
                usleep(1500);
                return true; // 按旧逻辑：不算失败，驱动/解码器还在就绪中
            }
            if (impl_->frames_fed > max_try) {
                RCUTILS_LOG_ERROR_NAMED("nvdec_mjpeg_decoder", "JPEG: capture_plane.getFormat failed after %d tries: %s", impl_->frames_fed, strerror(errno));
            }
        }
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
    if (!impl_->opened) {
        return;
    }

    if (impl_->dec) {
        impl_->dec->abort();

        impl_->dec->capture_plane.setStreamStatus(false);
        impl_->dec->capture_plane.deinitPlane();

        impl_->dec->output_plane.setStreamStatus(false);
        impl_->dec->output_plane.deinitPlane();

        delete impl_->dec;
        impl_->dec = nullptr;
    }

    for (int fd : impl_->capture_dmabuf_fds) {
        if (fd >= 0) {
            NvBufSurf::NvDestroy(fd);
        }
    }
    impl_->capture_dmabuf_fds.clear();
    impl_->capture_configured = false;
    impl_->capture_num_buffers = 0;
    impl_->capture_num_planes = 0;
    impl_->capture_pixfmt = 0;
    impl_->dec_w = 0;
    impl_->dec_h = 0;
    impl_->frames_fed = 0;
    impl_->out_in_use[0] = impl_->out_in_use[1] = false;

    if (impl_->egl_display != EGL_NO_DISPLAY) {
        eglTerminate(impl_->egl_display);
        impl_->egl_display = EGL_NO_DISPLAY;
    }

    if (impl_->v4l2_streaming && impl_->v4l2_fd >= 0) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        (void)v4l2_ioctl(impl_->v4l2_fd, VIDIOC_STREAMOFF, &type);
        impl_->v4l2_streaming = false;
    }

    for (auto &buf : impl_->v4l2_bufs) {
        if (buf.start && buf.length) {
            munmap(buf.start, buf.length);
        }
    }
    impl_->v4l2_bufs.clear();

    if (impl_->v4l2_fd >= 0) {
        ::close(impl_->v4l2_fd);
        impl_->v4l2_fd = -1;
    }

    impl_->opened = false;
}

bool NvdecMjpegDecoder::is_open() const { return impl_->opened; }

bool NvdecMjpegDecoder::is_supported()
{
  return true;
}

} // namespace gpu_cam_minimal
