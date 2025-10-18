// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#include "armor_detector/ai_detector.hpp"
#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <fstream>
#include <cstring>
#include <stdexcept>

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

inline void convertFloatToHalf(const std::vector<float>& src, std::vector<__half>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = __float2half(src[i]);
}

inline void convertHalfToFloat(const std::vector<__half>& src, std::vector<float>& dst)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i)
        dst[i] = __half2float(src[i]);
}

// moved to ai_kernels.cu

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
    checkCuda(cudaMallocHost(&host_pinned_bgr8_, bgr8_bytes_), "cudaMallocHost bgr8");
    checkCuda(cudaMalloc(&input_device_bgr8_, bgr8_bytes_), "cudaMalloc bgr8");

    std::cout << "[AIDetector] Engine loaded. Input: " << input_tensor_name_
              << " Output: " << output_tensor_name_ << std::endl;
}

AIDetector::~AIDetector()
{
    if (input_device_buffer_) cudaFree(input_device_buffer_);
    if (output_device_buffer_) cudaFree(output_device_buffer_);
    if (input_device_bgr8_) cudaFree(input_device_bgr8_);
    if (host_pinned_bgr8_) cudaFreeHost(host_pinned_bgr8_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Armor> AIDetector::detect(const cv::Mat& img, int color)
{
    // 清理本帧状态，避免累积
    armors_.clear();
    objects_.clear();
    tmp_objects_.clear();

    infer(img, color);

    // 使用 NMS 之后的结果构建装甲板，避免重复
    armors_.reserve(tmp_objects_.size());
    for (const auto& o : tmp_objects_) {
        Armor a = objectToArmor(o);
        if (a.type != ArmorType::INVALID) armors_.push_back(a);
    }
    return armors_;
}

void AIDetector::infer(const cv::Mat& img, int detect_color)
{
    // 清理本次推理的结果容器
    objects_.clear();
    tmp_objects_.clear();

    // 1) CPU 仅做 resize 到目标网络输入尺寸（保持 BGR8）
    cv::Mat resized;
    cv::resize(img, resized, {IMAGE_WIDTH, IMAGE_HEIGHT});
    if (resized.type() != CV_8UC3) {
        cv::Mat tmp;
        resized.convertTo(tmp, CV_8UC3);
        resized = tmp;
    }

    // 2) 复制 BGR8 到固定页主机缓冲
    std::memcpy(host_pinned_bgr8_, resized.data, bgr8_bytes_);

    // 3) 异步拷贝到设备端 BGR8 缓冲
    checkCuda(cudaMemcpyAsync(input_device_bgr8_, host_pinned_bgr8_, bgr8_bytes_, cudaMemcpyHostToDevice, stream_), "Memcpy H2D BGR8");

    // 4) 在 GPU 上将 BGR8(HWC) 转换为 RGB(NCHW, 归一化) 直接写到 TensorRT 输入缓冲
    if (input_data_type_ == nvinfer1::DataType::kHALF) {
        launch_bgr8_to_rgb_nchw_fp16(static_cast<const unsigned char*>(input_device_bgr8_),
                                     static_cast<__half*>(input_device_buffer_),
                                     IMAGE_WIDTH, IMAGE_HEIGHT, stream_);
    } else if (input_data_type_ == nvinfer1::DataType::kFLOAT) {
        launch_bgr8_to_rgb_nchw_fp32(static_cast<const unsigned char*>(input_device_bgr8_),
                                     static_cast<float*>(input_device_buffer_),
                                     IMAGE_WIDTH, IMAGE_HEIGHT, stream_);
    } else {
        throw std::runtime_error("Unsupported input data type for GPU preprocess");
    }

    context_->setInputShape(input_tensor_name_.c_str(), input_dims_);
    context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
    context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueue failed");

    // Copy back result
    std::vector<float> output_fp32(output_size_);
    size_t out_bytes = output_size_ * getElementSize(output_data_type_);
    if (output_data_type_ == nvinfer1::DataType::kHALF) {
        std::vector<__half> output_fp16(output_size_);
        checkCuda(cudaMemcpyAsync(output_fp16.data(), output_device_buffer_, out_bytes, cudaMemcpyDeviceToHost, stream_), "Memcpy D2H");
        cudaStreamSynchronize(stream_);
        convertHalfToFloat(output_fp16, output_fp32);
    } else {
        checkCuda(cudaMemcpyAsync(output_fp32.data(), output_device_buffer_, out_bytes, cudaMemcpyDeviceToHost, stream_), "Memcpy D2H");
        cudaStreamSynchronize(stream_);
    }

    // === Postprocess ===
    const int kAttr = 22;
    int num_det = output_size_ / kAttr;
    cv::Mat out(num_det, kAttr, CV_32F, output_fp32.data());
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (int i = 0; i < out.rows; ++i) {
        float conf = sigmoid(out.at<float>(i, 8));
        if (conf < conf_threshold_) continue;
        cv::Point cid, colorid;
        double sc_num, sc_color;
        cv::minMaxLoc(out.row(i).colRange(13, 22), nullptr, &sc_num, nullptr, &cid);
        cv::minMaxLoc(out.row(i).colRange(9, 13), nullptr, &sc_color, nullptr, &colorid);

        if (colorid.x >= 2) continue;
        if ((detect_color == 0 && colorid.x == 1) || (detect_color == 1 && colorid.x == 0)) continue;

        Object obj;
        obj.label = cid.x;
        obj.color = colorid.x;
        obj.prob = conf;

        float sx = static_cast<float>(img.cols) / IMAGE_WIDTH;
        float sy = static_cast<float>(img.rows) / IMAGE_HEIGHT;
        for (int j = 0; j < 8; ++j)
            obj.landmarks[j] = out.at<float>(i, j) * (j % 2 ? sy : sx);

        float minx = 1e9, maxx = 0, miny = 1e9, maxy = 0;
        for (int k = 0; k < 8; k += 2) {
            minx = std::min(minx, obj.landmarks[k]);
            maxx = std::max(maxx, obj.landmarks[k]);
            miny = std::min(miny, obj.landmarks[k + 1]);
            maxy = std::max(maxy, obj.landmarks[k + 1]);
        }

        obj.rect = cv::Rect(minx, miny, maxx - minx, maxy - miny);
        objects_.push_back(obj);
        boxes.push_back(obj.rect);
        scores.push_back(sc_num);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, idx);
    for (int i : idx) tmp_objects_.push_back(objects_[i]);
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
