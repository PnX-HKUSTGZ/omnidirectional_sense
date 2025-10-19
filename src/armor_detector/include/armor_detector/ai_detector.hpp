// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__AI_DETECTOR_HPP_
#define ARMOR_DETECTOR__AI_DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

// CUDA
#include <cuda_runtime.h>

// TensorRT
#include <NvInfer.h>
#include <NvOnnxParser.h>

// STD
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "armor_detector/types.hpp"
#include "armor_detector/ai_kernels.hpp"  // for PostDet

namespace rm_auto_aim
{

/**
 * @brief 基于 CUDA/cuDNN 的装甲板 AI 检测器
 * 
 * 使用深度学习模型进行装甲板检测和数字识别
 */
class AIDetector
{
public:
    /**
     * @brief 构造 AI 检测器
     * 
     * @param model_path ONNX 模型文件路径
     * @param device 推理设备 (仅支持 "GPU")
     * @param conf_threshold 置信度阈值
     * @param nms_threshold NMS 阈值
     */
    AIDetector(
        const std::string & model_path, const std::string & device = "GPU",
        float conf_threshold = 0.65f, float nms_threshold = 0.45f);

    /**
     * @brief 析构函数
     */
    ~AIDetector();

    /**
     * @brief 直接使用 GPU 图像进行检测（建议传入 BGR8，任意尺寸）
     * 
     * @param input_gpu 输入 GPU 图像（CV_8UC3）
     * @param detect_color 检测颜色 (0: red, 1: blue)
     * @return std::vector<Armor> 检测到的装甲板数组
     */
    std::vector<Armor> detect(const cv::cuda::GpuMat & input_gpu, int detect_color);

    /**
     * @brief 在图像上绘制检测结果
     * 
     * @param img 要绘制结果的图像
     */
    void drawResults(cv::Mat & img);

    /**
     * @brief 获取检测器类型名称
     * 
     * @return std::string 检测器类型名称
     */
    std::string getDetectorType() const { return "AIDetector"; }

private:
    /**
     * @brief 执行模型推理
     * 
     * @param gpu_bgr8 输入GPU图像
     * @param detect_color 检测颜色
     */
    void infer(const cv::cuda::GpuMat & gpu_bgr8, int detect_color);

    /**
     * @brief Sigmoid 激活函数
     * 
     * @param x 输入值
     * @return float Sigmoid 输出
     */
    inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

    /**
     * @brief 将检测对象转换为装甲板
     * 
     * @param obj 检测对象
     * @return Armor 装甲板
     */
    Armor objectToArmor(const Object & obj);

    /**
     * @brief TensorRT Logger 类
     */
    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char* msg) noexcept override
        {
            // 只输出警告和错误信息
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    };

    // TensorRT 相关
    Logger logger_;                                                    ///< TensorRT Logger
    std::unique_ptr<nvinfer1::IRuntime> runtime_;                     ///< TensorRT Runtime
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;                   ///< TensorRT Engine
    std::unique_ptr<nvinfer1::IExecutionContext> context_;            ///< TensorRT Execution Context
    
    // CUDA 相关
    cudaStream_t stream_;                                              ///< CUDA Stream
    void * input_device_buffer_{nullptr};                              ///< 输入缓冲区 (GPU)
    void * output_device_buffer_{nullptr};                             ///< 输出缓冲区 (GPU)
    void * input_device_bgr8_{nullptr};                                ///< 设备端 BGR8 中间缓冲
    size_t bgr8_bytes_{0};                                             ///< BGR8 缓冲区大小（字节）
    // GPU 后处理缓冲
    void * device_post_dets_{nullptr};                                  ///< 设备端候选输出数组
    int  * device_post_count_{nullptr};                                 ///< 设备端候选数量计数器
    int    max_post_out_{1024};                                         ///< 候选最大数量

    // 复用的缓冲区，避免在 infer 中分配大内存
    cv::cuda::GpuMat resized_gpu_;                                      ///< 预分配的缩放输出 GPU 图
    std::vector<PostDet> host_post_dets_;                               ///< 预分配的主机候选缓冲
    std::vector<cv::Rect> boxes_buf_;                                   ///< 预分配的 NMS 框缓冲
    std::vector<float> scores_buf_;                                     ///< 预分配的 NMS 分数缓冲
    std::vector<int> idx_buf_;                                          ///< 预分配的 NMS 索引缓冲
    
    // 输入输出信息
    std::string input_tensor_name_;                                    ///< 输入张量名称
    std::string output_tensor_name_;                                   ///< 输出张量名称
    nvinfer1::DataType input_data_type_{nvinfer1::DataType::kFLOAT};   ///< 输入数据类型
    nvinfer1::DataType output_data_type_{nvinfer1::DataType::kFLOAT};  ///< 输出数据类型
    nvinfer1::Dims input_dims_{};                                      ///< 模型输入原始维度
    nvinfer1::Dims output_dims_{};                                     ///< 模型输出原始维度
    size_t input_size_{0};                                             ///< 输入大小 (float 数量)
    size_t output_size_{0};                                            ///< 输出大小 (float 数量)
    
    // 参数
    float conf_threshold_;            ///< 置信度阈值
    float nms_threshold_;             ///< NMS 阈值
    std::vector<size_t> input_shape;  ///< 输入形状

    // 原始图像尺寸 (用于坐标缩放)
    int original_width_;   ///< 原始图像宽度
    int original_height_;  ///< 原始图像高度

    // 检测结果
    std::vector<Object> objects_;      ///< 原始检测对象
    std::vector<Object> tmp_objects_;  ///< NMS 后的检测对象
    std::vector<Armor> armors_;        ///< 本帧装甲板结果
    
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__AI_DETECTOR_HPP_