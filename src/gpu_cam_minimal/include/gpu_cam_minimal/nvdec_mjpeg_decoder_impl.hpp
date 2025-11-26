#ifndef GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_IMPL_HPP
#define GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_IMPL_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <NvBufSurface.h>
#include <NvBuffer.h>
#include <NvVideoDecoder.h>
#include <linux/videodev2.h>
#include <nvbufsurface.h>
#include <opencv2/core/cuda.hpp>

namespace gpu_cam_minimal {

class NvdecMjpegDecoderImpl {
  public:
    NvdecMjpegDecoderImpl() = default;
    ~NvdecMjpegDecoderImpl() = default;

    bool grab_camera_frame(v4l2_buffer &out_vbuf, void *&out_data,
                           size_t &out_len);
    bool feed_decoder_and_dequeue_capture(v4l2_buffer &cam_vbuf,
                                          const void *data, size_t len,
                                          NvBuffer *&out_cap_nvbuf,
                                          v4l2_buffer &out_cbuf);
    bool convert_capture_to_rgb(NvBuffer *cap_nvbuf, v4l2_buffer &cbuf,
                                cv::cuda::GpuMat &out_rgb);
    bool prepare_capture_dmabuf_buffer(v4l2_buffer &cbuf);
    NvBufSurfaceColorFormat resolve_capture_color_format(uint32_t pixfmt) const;
    bool requeue_capture_buffer(v4l2_buffer &cbuf);
    int get_capture_dmabuf_fd(uint32_t index) const;
    int get_camera_dmabuf_fd(uint32_t index) const;
    bool ensure_capture_cuda_resource(uint32_t index);
    void release_capture_cuda_resources();
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
    struct V4L2Buffer {
        void *start{nullptr};
        size_t length{0};
    };
    std::vector<V4L2Buffer> v4l2_bufs;
    std::vector<int> v4l2_buf_fds;
    bool camera_dmabuf_enabled{false};
    NvVideoDecoder *dec{nullptr};
    EGLDisplay egl_display{EGL_NO_DISPLAY};
    std::vector<unsigned char> enc_buf;
    bool capture_configured{false};
    int dec_w{0};
    int dec_h{0};
    int capture_num_planes{0};
    uint32_t capture_num_buffers{0};
    v4l2_plane_pix_format capture_plane_fmts[VIDEO_MAX_PLANES]{};
    std::vector<int> capture_dmabuf_fds;
    struct CaptureCudaResource {
        NvBufSurface *surface{nullptr};
        EGLImageKHR egl_image{EGL_NO_IMAGE_KHR};
        CUgraphicsResource cuda_resource{nullptr};
        bool egl_mapped{false};
    };
    std::vector<CaptureCudaResource> capture_cuda_resources;
    int out_next_idx{0};
    bool out_in_use[2]{false, false};
    int frames_fed{0};
    uint32_t capture_pixfmt{0};
    std::future<void> output_reclaim_future;
    std::atomic<bool> output_reclaim_stop{false};
    std::mutex output_plane_mutex;
    std::condition_variable output_plane_cv;
    bool use_output_reclaim_thread{false};
    uint32_t output_plane_memory{V4L2_MEMORY_MMAP};
    PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR{nullptr};
    PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR{nullptr};
    std::chrono::steady_clock::time_point last_capture_init_log{
        std::chrono::steady_clock::time_point::min()};
};

} // namespace gpu_cam_minimal

#endif // GPU_CAM_MINIMAL_NVDEC_MJPEG_DECODER_IMPL_HPP
