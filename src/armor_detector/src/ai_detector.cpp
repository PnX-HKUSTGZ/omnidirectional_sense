// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#include "armor_detector/ai_detector.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <rclcpp/rclcpp.hpp>
#include <stdexcept>

#include "armor_detector/ai_kernels.hpp"

namespace rm_auto_aim
{

namespace
{

inline void checkCuda(cudaError_t err, const char * msg)
{
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

inline void checkFile(const std::ifstream & f, const std::string & path)
{
    if (!f.good()) throw std::runtime_error("Failed to open engine file: " + path);
}

inline size_t getElementSize(nvinfer1::DataType t)
{
    switch (t) {
        case nvinfer1::DataType::kFLOAT:
            return sizeof(float);
        case nvinfer1::DataType::kHALF:
            return sizeof(__half);
        case nvinfer1::DataType::kINT32:
            return sizeof(int32_t);
        case nvinfer1::DataType::kINT8:
            return sizeof(int8_t);
        case nvinfer1::DataType::kBOOL:
            return sizeof(bool);
        default:
            throw std::runtime_error("Unsupported TensorRT data type");
    }
}

inline const char * trtTypeName(nvinfer1::DataType t)
{
    switch (t) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
        default:
            return "UNKNOWN";
    }
}

template <typename T>
inline size_t computeSize(const nvinfer1::Dims & d)
{
    size_t vol = 1;
    for (int i = 0; i < d.nbDims; ++i) vol *= static_cast<size_t>(d.d[i] > 0 ? d.d[i] : 1);
    return vol;
}

struct CarDet
{
    cv::Rect rect;
    float score;
    int cls;
};

inline cv::Rect makeSquareRoi(const cv::Rect & r, int img_w, int img_h)
{
    if (r.width <= 0 || r.height <= 0) return cv::Rect();

    float cx = r.x + r.width * 0.5f;
    float cy = r.y + r.height * 0.5f;
    float side = static_cast<float>(std::max(r.width, r.height));

    float x1 = cx - side * 0.5f;
    float y1 = cy - side * 0.5f;
    float x2 = cx + side * 0.5f;
    float y2 = cy + side * 0.5f;

    int ix1 = static_cast<int>(std::floor(x1));
    int iy1 = static_cast<int>(std::floor(y1));
    int ix2 = static_cast<int>(std::ceil(x2));
    int iy2 = static_cast<int>(std::ceil(y2));

    ix1 = std::max(0, std::min(ix1, img_w - 1));
    iy1 = std::max(0, std::min(iy1, img_h - 1));
    ix2 = std::max(0, std::min(ix2, img_w));
    iy2 = std::max(0, std::min(iy2, img_h));

    int w = ix2 - ix1;
    int h = iy2 - iy1;
    if (w <= 1 || h <= 1) return cv::Rect();
    return cv::Rect(ix1, iy1, w, h);
}

inline int parseCudaDeviceId(const std::string & device)
{
    if (device.empty() || device == "GPU") {
        return 0;
    }
    if (device.rfind("GPU", 0) == 0) {
        auto pos = device.find(':');
        if (pos != std::string::npos && pos + 1 < device.size()) {
            try {
                return std::stoi(device.substr(pos + 1));
            } catch (...) {
                return 0;
            }
        }
        return 0;
    }
    try {
        return std::stoi(device);
    } catch (...) {
        return 0;
    }
}

}  // namespace

AIDetector::AIDetector(
    const std::string & model_path, const std::string & device, float conf_th, float nms_th)
: conf_threshold_(conf_th), nms_threshold_(nms_th)
{
    cuda_device_id_ = parseCudaDeviceId(device);
    checkCuda(cudaSetDevice(cuda_device_id_), "cudaSetDevice");
    input_shape = {1, IMAGE_HEIGHT, IMAGE_WIDTH, 3};

    armor_engine_path_ = std::filesystem::path(model_path);
    car_engine_path_ = armor_engine_path_.parent_path() / "car.engine";

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (!device_count) throw std::runtime_error("No CUDA-enabled device found");

    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    // OpenCV stream wrapper (shares the same cudaStream_t)
    // NOTE: cv::cuda APIs below will use this when possible.

    // === Load car.engine ===
    {
        std::ifstream f(car_engine_path_, std::ios::binary);
        checkFile(f, car_engine_path_.string());
        f.seekg(0, std::ios::end);
        size_t sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> data(sz);
        f.read(data.data(), sz);
        f.close();

        car_runtime_.reset(nvinfer1::createInferRuntime(logger_));
        car_engine_.reset(car_runtime_->deserializeCudaEngine(data.data(), sz));
        if (!car_engine_) throw std::runtime_error("Failed to deserialize car TensorRT engine");

        car_context_.reset(car_engine_->createExecutionContext());
        if (!car_context_) throw std::runtime_error("Failed to create car execution context");

        for (int i = 0; i < car_engine_->getNbIOTensors(); ++i) {
            const char * name = car_engine_->getIOTensorName(i);
            auto mode = car_engine_->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
                car_input_tensor_name_ = name;
            else
                car_output_tensor_name_ = name;
        }
        if (car_input_tensor_name_.empty() || car_output_tensor_name_.empty())
            throw std::runtime_error("Missing car engine input/output tensor names");

        car_input_dims_ = car_engine_->getTensorShape(car_input_tensor_name_.c_str());
        car_output_dims_ = car_engine_->getTensorShape(car_output_tensor_name_.c_str());
        car_input_data_type_ = car_engine_->getTensorDataType(car_input_tensor_name_.c_str());
        car_output_data_type_ = car_engine_->getTensorDataType(car_output_tensor_name_.c_str());

        // car 模型约定：输入 FP16；输出允许 FP16 或 FP32（不同构建/精度模式可能不同）
        if (car_input_data_type_ != nvinfer1::DataType::kHALF ||
            (car_output_data_type_ != nvinfer1::DataType::kFLOAT &&
             car_output_data_type_ != nvinfer1::DataType::kHALF)) {
            throw std::runtime_error(
                std::string("car.engine expects FP16 input and FP16/FP32 output. Got input=") +
                trtTypeName(car_input_data_type_) + " output=" + trtTypeName(car_output_data_type_));
        }

        const int fallback[4] = {1, 3, IMAGE_HEIGHT, IMAGE_WIDTH};
        for (int i = 0; i < car_input_dims_.nbDims; ++i)
            if (car_input_dims_.d[i] == -1) car_input_dims_.d[i] = fallback[i];

        car_context_->setInputShape(car_input_tensor_name_.c_str(), car_input_dims_);
        car_output_dims_ = car_context_->getTensorShape(car_output_tensor_name_.c_str());

        car_input_size_ = computeSize<size_t>(car_input_dims_);
        car_output_size_ = computeSize<size_t>(car_output_dims_);

        checkCuda(
            cudaMalloc(
                &car_input_device_buffer_,
                car_input_size_ * getElementSize(car_input_data_type_)),
            "cudaMalloc car input");
        checkCuda(
            cudaMalloc(
                &car_output_device_buffer_,
                car_output_size_ * getElementSize(car_output_data_type_)),
            "cudaMalloc car output");

        // Host output buffer: keep a float view for downstream parsing.
        car_host_output_.resize(car_output_size_);
        if (car_output_data_type_ == nvinfer1::DataType::kHALF) {
            car_host_output_fp16_.resize(car_output_size_);
        }

        std::cout << "[AIDetector] Car engine loaded: " << car_engine_path_.string() << std::endl;
    }

    // === Load armor engine ===
    std::ifstream file(armor_engine_path_, std::ios::binary);
    checkFile(file, armor_engine_path_.string());
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
        const char * name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT)
            input_tensor_name_ = name;
        else
            output_tensor_name_ = name;
    }
    if (input_tensor_name_.empty() || output_tensor_name_.empty())
        throw std::runtime_error("Missing input/output tensor names");

    input_dims_ = engine_->getTensorShape(input_tensor_name_.c_str());
    output_dims_ = engine_->getTensorShape(output_tensor_name_.c_str());
    input_data_type_ = engine_->getTensorDataType(input_tensor_name_.c_str());
    output_data_type_ = engine_->getTensorDataType(output_tensor_name_.c_str());

    // 固定支持：输入 FP16，输出 FP32（当前模型约定）。
    if (input_data_type_ != nvinfer1::DataType::kHALF ||
        output_data_type_ != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error(
            "AIDetector expects engine with FP16 input and FP32 output. Got input=" +
            std::to_string(static_cast<int>(input_data_type_)) +
            " output=" + std::to_string(static_cast<int>(output_data_type_)));
    }

    const int fallback[4] = {1, 3, IMAGE_HEIGHT, IMAGE_WIDTH};
    for (int i = 0; i < input_dims_.nbDims; ++i)
        if (input_dims_.d[i] == -1) input_dims_.d[i] = fallback[i];

    context_->setInputShape(input_tensor_name_.c_str(), input_dims_);
    output_dims_ = context_->getTensorShape(output_tensor_name_.c_str());

    input_size_ = computeSize<size_t>(input_dims_);
    output_size_ = computeSize<size_t>(output_dims_);

    checkCuda(
        cudaMalloc(&input_device_buffer_, input_size_ * getElementSize(input_data_type_)),
        "cudaMalloc input");
    checkCuda(
        cudaMalloc(&output_device_buffer_, output_size_ * getElementSize(output_data_type_)),
        "cudaMalloc output");

    // Allocate GPU postprocess buffers
    checkCuda(
        cudaMalloc(&device_post_dets_, static_cast<size_t>(max_post_out_) * sizeof(PostDet)),
        "cudaMalloc post_dets");
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
    if (cudaSetDevice(cuda_device_id_) != cudaSuccess) {
        std::cerr << "[AIDetector] cudaSetDevice(" << cuda_device_id_
                  << ") failed during destruction" << std::endl;
        return;
    }
    if (input_device_buffer_) cudaFree(input_device_buffer_);
    if (output_device_buffer_) cudaFree(output_device_buffer_);

    if (car_input_device_buffer_) cudaFree(car_input_device_buffer_);
    if (car_output_device_buffer_) cudaFree(car_output_device_buffer_);

    if (device_post_dets_) cudaFree(device_post_dets_);
    if (device_post_count_) cudaFree(device_post_count_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Armor> AIDetector::detect(const cv::cuda::GpuMat & gpu_img, int color)
{
    // 清理本帧状态
    armors_.clear();
    objects_.clear();
    tmp_objects_.clear();

    infer(gpu_img, color);

    armors_.reserve(tmp_objects_.size());
    for (const auto & o : tmp_objects_) {
        Armor a = objectToArmor(o);
        if (a.type != ArmorType::INVALID) armors_.push_back(a);
    }
    return armors_;
}

void AIDetector::infer(const cv::cuda::GpuMat & gpu_rgb8, int detect_color)
{
    checkCuda(cudaSetDevice(cuda_device_id_), "cudaSetDevice");
    // 清理结果
    objects_.clear();
    tmp_objects_.clear();
    last_car_boxes_.clear();

    const auto logger = rclcpp::get_logger("AIDetector");

    // 1) 输入检查
    if (gpu_rgb8.type() != CV_8UC3) {
        RCLCPP_ERROR(
            logger,
            "[AIDetector] Input GpuMat must be CV_8UC3, aborting inference.");
        return;
    }

    // ============ Stage 1: car.engine (letterbox + NMS) ============
    // Letterbox parameters (match the provided minimal Python: fixed 640x640)
    int orig_w = gpu_rgb8.cols;
    int orig_h = gpu_rgb8.rows;
    float r = std::min(static_cast<float>(IMAGE_WIDTH) / orig_w, static_cast<float>(IMAGE_HEIGHT) / orig_h);
    int resized_w = static_cast<int>(std::round(orig_w * r));
    int resized_h = static_cast<int>(std::round(orig_h * r));
    int pad_x = (IMAGE_WIDTH - resized_w) / 2;
    int pad_y = (IMAGE_HEIGHT - resized_h) / 2;

    cv::cuda::Stream cv_stream = cv::cuda::StreamAccessor::wrapStream(stream_);
    cv::cuda::GpuMat car_resized;
    cv::cuda::resize(gpu_rgb8, car_resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_LINEAR, cv_stream);
    cv::cuda::GpuMat car_padded(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3);
    car_padded.setTo(cv::Scalar(114, 114, 114), cv_stream);
    cv::cuda::GpuMat car_roi = car_padded(cv::Rect(pad_x, pad_y, resized_w, resized_h));
    car_resized.copyTo(car_roi, cv_stream);

    // Convert to FP16 NCHW
    launch_resize_rgb8_to_rgb_nchw_fp16(
        static_cast<const unsigned char *>(car_padded.ptr<unsigned char>()),
        static_cast<size_t>(car_padded.step), car_padded.cols, car_padded.rows,
        static_cast<__half *>(car_input_device_buffer_), IMAGE_WIDTH, IMAGE_HEIGHT, stream_);

    car_context_->setInputShape(car_input_tensor_name_.c_str(), car_input_dims_);
    car_context_->setTensorAddress(car_input_tensor_name_.c_str(), car_input_device_buffer_);
    car_context_->setTensorAddress(car_output_tensor_name_.c_str(), car_output_device_buffer_);
    if (!car_context_->enqueueV3(stream_)) throw std::runtime_error("car TensorRT enqueue failed");

    // Copy car output to host
    if (car_output_data_type_ == nvinfer1::DataType::kFLOAT) {
        checkCuda(
            cudaMemcpyAsync(
                car_host_output_.data(), car_output_device_buffer_,
                car_output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream_),
            "Memcpy car output D2H");
        cudaStreamSynchronize(stream_);
    } else if (car_output_data_type_ == nvinfer1::DataType::kHALF) {
        checkCuda(
            cudaMemcpyAsync(
                car_host_output_fp16_.data(), car_output_device_buffer_,
                car_output_size_ * sizeof(__half), cudaMemcpyDeviceToHost, stream_),
            "Memcpy car output D2H");
        cudaStreamSynchronize(stream_);
        for (size_t i = 0; i < car_output_size_; ++i) {
            car_host_output_[i] = __half2float(car_host_output_fp16_[i]);
        }
    } else {
        throw std::runtime_error("Unsupported car.engine output dtype");
    }

    // Parse YOLOv5 export output: [cx,cy,w,h,obj,cls...]
    int kAttrCar = 0;
    if (car_output_dims_.nbDims >= 1) {
        kAttrCar = car_output_dims_.d[car_output_dims_.nbDims - 1];
    }
    if (kAttrCar < 6) {
        RCLCPP_ERROR(logger, "[AIDetector] car.engine output attr invalid: %d", kAttrCar);
        return;
    }
    int num_det_car = static_cast<int>(car_output_size_ / static_cast<size_t>(kAttrCar));

    std::vector<cv::Rect> car_boxes;
    std::vector<float> car_scores;
    car_boxes.reserve(256);
    car_scores.reserve(256);

    for (int i = 0; i < num_det_car; ++i) {
        const float * row = car_host_output_.data() + static_cast<size_t>(i) * kAttrCar;
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        float obj = row[4];

        int best_cls = 0;
        float best_cls_score = row[5];
        for (int c = 6; c < kAttrCar; ++c) {
            if (row[c] > best_cls_score) {
                best_cls_score = row[c];
                best_cls = c - 5;
            }
        }
        float conf = obj * best_cls_score;
        if (conf < conf_threshold_) continue;

        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // Map letterboxed coords (640x640) back to original image
        x1 = (x1 - static_cast<float>(pad_x)) / r;
        y1 = (y1 - static_cast<float>(pad_y)) / r;
        x2 = (x2 - static_cast<float>(pad_x)) / r;
        y2 = (y2 - static_cast<float>(pad_y)) / r;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h - 1)));

        int ix1 = static_cast<int>(std::floor(x1));
        int iy1 = static_cast<int>(std::floor(y1));
        int iw = static_cast<int>(std::ceil(x2 - x1));
        int ih = static_cast<int>(std::ceil(y2 - y1));
        if (iw <= 1 || ih <= 1) continue;

        car_boxes.emplace_back(ix1, iy1, iw, ih);
        car_scores.emplace_back(conf);
        (void)best_cls;
    }

    std::vector<int> car_keep;
    if (!car_boxes.empty()) {
        cv::dnn::NMSBoxes(car_boxes, car_scores, conf_threshold_, nms_threshold_, car_keep);
    }

    // 保存 car.engine 的检测框，供外部可视化（已是原图坐标）
    last_car_boxes_.reserve(car_keep.size());
    for (int ki : car_keep) {
        if (ki >= 0 && ki < static_cast<int>(car_boxes.size())) {
            last_car_boxes_.push_back(car_boxes[ki]);
        }
    }

    // ============ Stage 2: 0526.engine (armor) on each square ROI ============
    boxes_buf_.clear();
    scores_buf_.clear();
    objects_.clear();

    // 由于模型训练时的问题，红蓝反了。
    if (detect_color == 0)
        detect_color = 1;
    else if (detect_color == 1)
        detect_color = 0;

    // 始终对整幅图像做一次装甲板检测，再追加车辆 ROI 的双层检测结果，由后续 NMS 融合。
    std::vector<cv::Rect> armor_rois;
    armor_rois.reserve(car_keep.size() + 1);
    armor_rois.emplace_back(0, 0, orig_w, orig_h);
    for (int ki : car_keep) {
        if (ki < 0 || ki >= static_cast<int>(car_boxes.size())) continue;
        const cv::Rect & car_box = car_boxes[ki];
        cv::Rect roi_rect = makeSquareRoi(car_box, orig_w, orig_h);
        if (roi_rect.area() <= 0) continue;
        armor_rois.push_back(roi_rect);
    }

    for (const auto & roi_rect : armor_rois) {
        if (roi_rect.area() <= 0) continue;

        cv::cuda::GpuMat roi_gpu(gpu_rgb8, roi_rect);

        // Preprocess for armor model: direct resize ROI -> 640x640, NCHW FP16
        launch_resize_rgb8_to_rgb_nchw_fp16(
            static_cast<const unsigned char *>(roi_gpu.ptr<unsigned char>()),
            static_cast<size_t>(roi_gpu.step), roi_gpu.cols, roi_gpu.rows,
            static_cast<__half *>(input_device_buffer_), IMAGE_WIDTH, IMAGE_HEIGHT, stream_);

        context_->setInputShape(input_tensor_name_.c_str(), input_dims_);
        context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
        context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);
        if (!context_->enqueueV3(stream_)) throw std::runtime_error("armor TensorRT enqueue failed");

        // GPU postprocess for armor
        const int kAttrArmor = 22;
        int num_det_armor = static_cast<int>(output_size_ / kAttrArmor);
        float sx = static_cast<float>(roi_gpu.cols) / IMAGE_WIDTH;
        float sy = static_cast<float>(roi_gpu.rows) / IMAGE_HEIGHT;

        checkCuda(cudaMemsetAsync(device_post_count_, 0, sizeof(int), stream_), "Memset post_count");
        launch_postprocess_fp32(
            static_cast<const float *>(output_device_buffer_), num_det_armor, conf_threshold_,
            detect_color, sx, sy, static_cast<PostDet *>(device_post_dets_), max_post_out_,
            device_post_count_, stream_);

        int host_count = 0;
        checkCuda(
            cudaMemcpyAsync(
                &host_count, device_post_count_, sizeof(int), cudaMemcpyDeviceToHost, stream_),
            "Memcpy count D2H");
        cudaStreamSynchronize(stream_);
        host_count = std::max(0, std::min(host_count, max_post_out_));

        host_post_dets_.resize(host_count);
        if (host_count > 0) {
            checkCuda(
                cudaMemcpyAsync(
                    host_post_dets_.data(), device_post_dets_,
                    static_cast<size_t>(host_count) * sizeof(PostDet), cudaMemcpyDeviceToHost,
                    stream_),
                "Memcpy dets D2H");
            cudaStreamSynchronize(stream_);
        }

        // Append candidates (offset back to original image)
        for (int i = 0; i < host_count; ++i) {
            const auto & d = host_post_dets_[i];
            Object obj;
            obj.label = d.label;
            obj.color = d.color;
            obj.prob = d.prob;
            for (int j = 0; j < 8; ++j) {
                obj.landmarks[j] = d.landmarks[j] + ((j & 1) ? roi_rect.y : roi_rect.x);
            }
            obj.rect = cv::Rect(
                static_cast<int>(d.x) + roi_rect.x, static_cast<int>(d.y) + roi_rect.y,
                static_cast<int>(d.w), static_cast<int>(d.h));

            objects_.push_back(obj);
            boxes_buf_.push_back(obj.rect);
            scores_buf_.push_back(d.score_num);
        }
    }

    idx_buf_.clear();
    if (!boxes_buf_.empty()) {
        cv::dnn::NMSBoxes(boxes_buf_, scores_buf_, conf_threshold_, nms_threshold_, idx_buf_);
        for (int i : idx_buf_) tmp_objects_.push_back(objects_[i]);
    }
}

Armor AIDetector::objectToArmor(const Object & o)
{
    Armor a;
    cv::Point2f lt(o.landmarks[0], o.landmarks[1]);
    cv::Point2f lb(o.landmarks[2], o.landmarks[3]);
    cv::Point2f rt(o.landmarks[6], o.landmarks[7]);
    cv::Point2f rb(o.landmarks[4], o.landmarks[5]);

    Light L(o.color, lt, lb), R(o.color, rt, rb);
    if (L.boundingRect().area() == 0 || R.boundingRect().area() == 0) {
        Armor invalid;
        invalid.type = ArmorType::INVALID;
        return invalid;
    }

    ArmorType type = (o.label == 1 || o.label == 7) ? ArmorType::LARGE : ArmorType::SMALL;
    std::vector<std::string> cls = {"outpost", "1", "2", "3", "4", "5", "guard", "base", "base"};

    a = Armor(L, R);
    a.number = cls[o.label];
    a.type = type;
    a.confidence = o.prob;
    a.classfication_result =
        a.number + ": " + std::to_string(a.confidence * 100.0f).substr(0, 4) + "%";
    return a;
}

void AIDetector::drawResults(cv::Mat & img)
{
    for (const auto & a : armors_) {
        cv::line(img, a.left_light.top, a.right_light.bottom, {0, 255, 0}, 2);
        cv::line(img, a.right_light.top, a.left_light.bottom, {0, 255, 0}, 2);
        cv::putText(
            img, a.classfication_result, a.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.7,
            {0, 255, 255}, 2);
    }
}

}  // namespace rm_auto_aim
