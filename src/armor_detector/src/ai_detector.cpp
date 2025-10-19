// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <rclcpp/rclcpp.hpp>

#include "armor_detector/ai_detector.hpp"
#include "armor_detector/ai_kernels.hpp"

namespace rm_auto_aim
{

namespace
{

inline void checkCuda(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

inline void checkFile(const std::ifstream& f, const std::string& path)
{
    if (!f.good()) throw std::runtime_error("Failed to open engine file: " + path);
}

inline size_t getElementSize(nvinfer1::DataType t)
{
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return sizeof(float);
        case nvinfer1::DataType::kHALF: return sizeof(__half);
        case nvinfer1::DataType::kINT32: return sizeof(int32_t);
        case nvinfer1::DataType::kINT8:  return sizeof(int8_t);
        case nvinfer1::DataType::kBOOL:  return sizeof(bool);
        default: throw std::runtime_error("Unsupported TensorRT data type");
    }
}

template <typename T>
inline size_t computeSize(const nvinfer1::Dims& d)
{
    size_t vol = 1;
    for (int i = 0; i < d.nbDims; ++i)
        vol *= static_cast<size_t>(d.d[i] > 0 ? d.d[i] : 1);
    return vol;
}

}  // namespace

AIDetector::AIDetector(const std::string& model_path, const std::string&, float conf_th, float nms_th)
    : conf_threshold_(conf_th), nms_threshold_(nms_th)
{
    input_shape = {1, IMAGE_HEIGHT, IMAGE_WIDTH, 3};

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (!device_count) throw std::runtime_error("No CUDA-enabled device found");

    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    // === Load engine ===
    std::ifstream file(model_path, std::ios::binary);
    checkFile(file, model_path);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT engine");

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create execution context");

    // === Tensor info ===
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) input_tensor_name_ = name;
        else output_tensor_name_ = name;
    }
    if (input_tensor_name_.empty() || output_tensor_name_.empty())
        throw std::runtime_error("Missing input/output tensor names");

    input_dims_ = engine_->getTensorShape(input_tensor_name_.c_str());
    output_dims_ = engine_->getTensorShape(output_tensor_name_.c_str());
    input_data_type_ = engine_->getTensorDataType(input_tensor_name_.c_str());
    output_data_type_ = engine_->getTensorDataType(output_tensor_name_.c_str());

    // 固定支持：输入 FP16，输出 FP32（当前模型约定）。
    if (input_data_type_ != nvinfer1::DataType::kHALF || output_data_type_ != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("AIDetector expects engine with FP16 input and FP32 output. Got input="
                                 + std::to_string(static_cast<int>(input_data_type_)) +
                                 " output=" + std::to_string(static_cast<int>(output_data_type_)));
    }

    const int fallback[4] = {1, 3, IMAGE_HEIGHT, IMAGE_WIDTH};
    for (int i = 0; i < input_dims_.nbDims; ++i)
        if (input_dims_.d[i] == -1) input_dims_.d[i] = fallback[i];

    context_->setInputShape(input_tensor_name_.c_str(), input_dims_);
    output_dims_ = context_->getTensorShape(output_tensor_name_.c_str());

    input_size_ = computeSize<size_t>(input_dims_);
    output_size_ = computeSize<size_t>(output_dims_);

    checkCuda(cudaMalloc(&input_device_buffer_, input_size_ * getElementSize(input_data_type_)), "cudaMalloc input");
    checkCuda(cudaMalloc(&output_device_buffer_, output_size_ * getElementSize(output_data_type_)), "cudaMalloc output");

    // Allocate BGR8 pinned/device buffers for GPU preprocess (H x W x 3)
    bgr8_bytes_ = static_cast<size_t>(IMAGE_WIDTH) * IMAGE_HEIGHT * 3;
    checkCuda(cudaMalloc(&input_device_bgr8_, bgr8_bytes_), "cudaMalloc bgr8");

    // Allocate GPU postprocess buffers
    checkCuda(cudaMalloc(&device_post_dets_, static_cast<size_t>(max_post_out_) * sizeof(PostDet)), "cudaMalloc post_dets");
    checkCuda(cudaMalloc(&device_post_count_, sizeof(int)), "cudaMalloc post_count");

    // Pre-create reusable buffers
    host_post_dets_.reserve(max_post_out_);
    boxes_buf_.reserve(max_post_out_);
    scores_buf_.reserve(max_post_out_);
    idx_buf_.reserve(max_post_out_);

    std::cout << "[AIDetector] Engine loaded. Input: " << input_tensor_name_
              << " Output: " << output_tensor_name_ << std::endl;
}

AIDetector::~AIDetector()
{
    if (input_device_buffer_) cudaFree(input_device_buffer_);
    if (output_device_buffer_) cudaFree(output_device_buffer_);
    if (input_device_bgr8_) cudaFree(input_device_bgr8_);
    
    if (device_post_dets_) cudaFree(device_post_dets_);
    if (device_post_count_) cudaFree(device_post_count_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Armor> AIDetector::detect(const cv::cuda::GpuMat& gpu_img, int color)
{
    // 清理本帧状态
    armors_.clear();
    objects_.clear();
    tmp_objects_.clear();

    infer(gpu_img, color);

    armors_.reserve(tmp_objects_.size());
    for (const auto& o : tmp_objects_) {
        Armor a = objectToArmor(o);
        if (a.type != ArmorType::INVALID) armors_.push_back(a);
    }
    return armors_;
}

void AIDetector::infer(const cv::cuda::GpuMat& gpu_bgr8, int detect_color)
{
    // 清理结果
    objects_.clear();
    tmp_objects_.clear();

    // 1) 输入检查
    if (gpu_bgr8.type() != CV_8UC3) {
        RCLCPP_ERROR(rclcpp::get_logger("AIDetector"),
                     "[AIDetector] Input GpuMat must be CV_8UC3, aborting inference.");
        return;
    }
    // 2) 融合：从原图直接双线性缩放 + BGR->RGB + NCHW + 归一化到 FP16 输入
    launch_resize_bgr8_to_rgb_nchw_fp16(
        static_cast<const unsigned char*>(gpu_bgr8.ptr<unsigned char>()),
        static_cast<size_t>(gpu_bgr8.step),
        gpu_bgr8.cols, gpu_bgr8.rows,
        static_cast<__half*>(input_device_buffer_), IMAGE_WIDTH, IMAGE_HEIGHT,
        stream_);

    context_->setInputShape(input_tensor_name_.c_str(), input_dims_);
    context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
    context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueue failed");

    // GPU 后处理：在设备端完成 sigmoid/阈值/argmax/过滤/压缩
    const int kAttr = 22;
    int num_det = static_cast<int>(output_size_ / kAttr);
    float sx = static_cast<float>(gpu_bgr8.cols) / IMAGE_WIDTH;
    float sy = static_cast<float>(gpu_bgr8.rows) / IMAGE_HEIGHT;

    // 清零计数器
    checkCuda(cudaMemsetAsync(device_post_count_, 0, sizeof(int), stream_), "Memset post_count");

    // 后处理固定走 FP32
    launch_postprocess_fp32(static_cast<const float*>(output_device_buffer_), num_det, conf_threshold_,
                            detect_color, sx, sy,
                            static_cast<PostDet*>(device_post_dets_), max_post_out_, device_post_count_,
                            stream_);

    // 拷回数量
    int host_count = 0;
    checkCuda(cudaMemcpyAsync(&host_count, device_post_count_, sizeof(int), cudaMemcpyDeviceToHost, stream_), "Memcpy count D2H");
    cudaStreamSynchronize(stream_);
    host_count = std::max(0, std::min(host_count, max_post_out_));

    // 拷回候选
    host_post_dets_.resize(host_count);
    if (host_count > 0) {
        checkCuda(cudaMemcpyAsync(host_post_dets_.data(), device_post_dets_, static_cast<size_t>(host_count) * sizeof(PostDet),
                                  cudaMemcpyDeviceToHost, stream_), "Memcpy dets D2H");
        cudaStreamSynchronize(stream_);
    }

    // CPU 侧只做 NMS
    boxes_buf_.clear();
    scores_buf_.clear();
    boxes_buf_.reserve(host_count);
    scores_buf_.reserve(host_count);
    for (int i = 0; i < host_count; ++i) {
        const auto &d = host_post_dets_[i];
        Object obj;
        obj.label = d.label;
        obj.color = d.color;
        obj.prob  = d.prob;
        for (int j = 0; j < 8; ++j) obj.landmarks[j] = d.landmarks[j];
        obj.rect = cv::Rect(static_cast<int>(d.x), static_cast<int>(d.y),
                            static_cast<int>(d.w), static_cast<int>(d.h));
        objects_.push_back(obj);
        boxes_buf_.push_back(obj.rect);
        scores_buf_.push_back(d.score_num);
    }

    idx_buf_.clear();
    cv::dnn::NMSBoxes(boxes_buf_, scores_buf_, conf_threshold_, 0, idx_buf_);
    for (int i : idx_buf_) tmp_objects_.push_back(objects_[i]);
}

Armor AIDetector::objectToArmor(const Object& o)
{
    Armor a;
    cv::Point2f lt(o.landmarks[0], o.landmarks[1]);
    cv::Point2f lb(o.landmarks[2], o.landmarks[3]);
    cv::Point2f rt(o.landmarks[6], o.landmarks[7]);
    cv::Point2f rb(o.landmarks[4], o.landmarks[5]);

    Light L(o.color, lt, lb), R(o.color, rt, rb);
    if (L.boundingRect().area() == 0 || R.boundingRect().area() == 0){
        Armor invalid;
        invalid.type = ArmorType::INVALID;
        return invalid;
    }

    ArmorType type = (o.label == 1 || o.label == 7) ? ArmorType::LARGE : ArmorType::SMALL;
    std::vector<std::string> cls = {"outpost","1","2","3","4","5","guard","base","base"};

    a = Armor(L, R);
    a.number = cls[o.label];
    a.type = type;
    a.confidence = o.prob;
    a.classfication_result = a.number + ": " + std::to_string(a.confidence * 100.0f).substr(0, 4) + "%";
    return a;
}

void AIDetector::drawResults(cv::Mat& img)
{
    for (const auto& a : armors_) {
        cv::line(img, a.left_light.top, a.right_light.bottom, {0,255,0}, 2);
        cv::line(img, a.right_light.top, a.left_light.bottom, {0,255,0}, 2);
        cv::putText(img, a.classfication_result, a.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0,255,255}, 2);
    }
}

}  // namespace rm_auto_aim
