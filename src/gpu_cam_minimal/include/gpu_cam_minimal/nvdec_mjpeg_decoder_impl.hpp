#ifndef GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_IMPL_HPP
#define GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_IMPL_HPP

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <NvBufSurface.h>
#include <NvBuffer.h>
#include <NvVideoDecoder.h>
#include <linux/videodev2.h>
#include <nvbufsurface.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <memory>
#include <mutex>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <thread>
#include <vector>

namespace gpu_cam_minimal
{

class NvdecMjpegDecoderImpl
{
public:
    NvdecMjpegDecoderImpl() = default;
    ~NvdecMjpegDecoderImpl();

    void reset();

    struct NvDecoderDeleter
    {
        void operator()(NvVideoDecoder * ptr) const { delete ptr; }
    };

    struct CaptureResult
    {
        int ret{-1};
        int err{0};
        NvBuffer * nvbuf{nullptr};
        v4l2_buffer buffer{};
        std::array<v4l2_plane, VIDEO_MAX_PLANES> planes{};
    };

    bool grab_camera_frame(v4l2_buffer & out_vbuf, void *& out_data, size_t & out_len);
    bool feed_decoder_and_dequeue_capture(
        v4l2_buffer & cam_vbuf, const void * data, size_t len, NvBuffer *& out_cap_nvbuf,
        v4l2_buffer & out_cbuf);
    bool convert_capture_to_rgb(
        NvBuffer * cap_nvbuf, v4l2_buffer & cbuf, cv::cuda::GpuMat & out_rgb);
    bool prepare_capture_dmabuf_buffer(v4l2_buffer & cbuf);
    NvBufSurfaceColorFormat resolve_capture_color_format(uint32_t pixfmt) const;
    bool requeue_capture_buffer(v4l2_buffer & cbuf);
    int get_capture_dmabuf_fd(uint32_t index) const;
    void start_output_reclaim_thread();
    void stop_output_reclaim_thread();
    static bool set_v4l2_mjpeg(int fd, int w, int h, double f);
    bool initEglExtensions();

    std::string device;
    int width{0};
    int height{0};
    double fps{0.0};
    bool opened{false};
    int v4l2_fd{-1};
    bool v4l2_streaming{false};
    struct V4L2Buffer
    {
        void * start{nullptr};
        size_t length{0};
    };
    std::vector<V4L2Buffer> v4l2_bufs;
    std::unique_ptr<NvVideoDecoder, NvDecoderDeleter> dec;
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
    int frames_fed{0};
    uint32_t capture_pixfmt{0};
    std::future<void> output_reclaim_future;
    std::atomic<bool> output_reclaim_stop{false};
    std::mutex output_plane_mutex;
    std::condition_variable output_plane_cv;
    PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR{nullptr};
    PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR{nullptr};
    std::chrono::steady_clock::time_point last_capture_init_log{
        std::chrono::steady_clock::time_point::min()};
    uint32_t requested_v4l2_buffers{4};
    uint32_t capture_buffer_padding{2};
    bool drop_late_frames{false};
    std::future<CaptureResult> capture_future_;
    bool capture_future_valid_{false};

    void ensure_capture_future();
    bool fetch_capture_result(CaptureResult & result, int timeout_ms);
    void fill_v4l2_buffer_from_capture(const CaptureResult & src, v4l2_buffer & dst);
};

}  // namespace gpu_cam_minimal

#endif  // GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_IMPL_HPP
