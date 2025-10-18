// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__AI_DETECTOR_HPP_
#define ARMOR_DETECTOR__AI_DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

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

#include "armor_detector/base_detector.hpp"
#include "armor_detector/types.hpp"

namespace rm_auto_aim
{

/**
 * @brief 基于 CUDA/cuDNN 的装甲板 AI 检测器
 * 
 * 使用深度学习模型进行装甲板检测和数字识别
 */
class AIDetector : public BaseDetector
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
    ~AIDetector() override;

    /**
     * @brief 检测装甲板
     * 
     * @param input 输入图像
     * @param detect_color 检测颜色 (0: red, 1: blue)
     * @return std::vector<Armor> 检测到的装甲板数组
     */
    std::vector<Armor> detect(const cv::Mat & input, int detect_color) override;

    /**
     * @brief 在图像上绘制检测结果
     * 
     * @param img 要绘制结果的图像
     */
    void drawResults(cv::Mat & img) override;

    /**
     * @brief 获取检测器类型名称
     * 
     * @return std::string 检测器类型名称
     */
    std::string getDetectorType() const override { return "AIDetector"; }

private:
    /**
     * @brief 执行模型推理
     * 
     * @param img 输入图像
     * @param detect_color 检测颜色
     */
    void infer(const cv::Mat & img, int detect_color);

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
    void * host_pinned_bgr8_{nullptr};                                  ///< 主机端 BGR8 固定页缓冲
    size_t bgr8_bytes_{0};                                             ///< BGR8 缓冲区大小（字节）
    
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
    
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__AI_DETECTOR_HPP_