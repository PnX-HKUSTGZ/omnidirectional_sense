// Copyright 2024 PnX-HKUSTGZ
// Licensed under the MIT License.

#include "armor_detector/ai_detector.hpp"

#include <algorithm>
#include <vector>
#include <opencv2/cudaarithm.hpp>

namespace rm_auto_aim
{

AIDetector::AIDetector(
    const std::string & model_path, const std::string & device, float conf_threshold,
    float nms_threshold)
: conf_threshold_(conf_threshold), nms_threshold_(nms_threshold)
{
    // 设置输入形状
    input_shape = {1, static_cast<size_t>(IMAGE_HEIGHT), static_cast<size_t>(IMAGE_WIDTH), 3};

    // 检查 CUDA 是否可用
    int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_device_count == 0) {
        throw std::runtime_error("No CUDA-enabled devices found!");
    }
    
    std::cout << "Found " << cuda_device_count << " CUDA device(s)" << std::endl;
    
    // 获取 CUDA 设备信息
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    
    // 加载 ONNX 模型
    std::cout << "Loading model from: " << model_path << std::endl;
    net_ = cv::dnn::readNetFromONNX(model_path);
    
    // 设置 CUDA 后端
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    
    std::cout << "AI Detector initialized with CUDA backend" << std::endl;
}

AIDetector::~AIDetector() = default;

std::vector<Armor> AIDetector::detect(const cv::Mat & input, int detect_color)
{
    // 清空之前的结果
    objects_.clear();
    tmp_objects_.clear();
    ious_.clear();
    armors_.clear();

    cv::cvtColor(input, gray_img, cv::COLOR_RGB2GRAY);
    
    // 记录原始图像尺寸用于坐标缩放
    original_width_ = input.cols;
    original_height_ = input.rows;

    // 执行推理
    infer(input, detect_color);

    // 将检测对象转换为装甲板
    armors_.reserve(tmp_objects_.size());

    for (const auto & obj : tmp_objects_) {
        Armor armor = objectToArmor(obj);
        if (armor.type == ArmorType::INVALID) continue;
        armors_.push_back(armor);
    }

    return armors_;
}

void AIDetector::infer(const cv::Mat & img, int detect_color)
{
    // 清空之前的结果
    objects_.clear();
    tmp_objects_.clear();
    ious_.clear();

    // Resize图像到模型输入尺寸
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT));

    // 创建 blob (NCHW 格式，归一化到 [0, 1])
    cv::Mat blob = cv::dnn::blobFromImage(
        resized_img, 1.0 / 255.0, cv::Size(IMAGE_WIDTH, IMAGE_HEIGHT), 
        cv::Scalar(), true, false);

    // 设置输入
    net_.setInput(blob);

    // 执行推理
    std::vector<cv::Mat> outputs;
    net_.forward(outputs);

    // 假设输出是 [1, 25200, 22] 的格式
    cv::Mat output_buffer = outputs[0].reshape(1, outputs[0].size[1]);

    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<float> confidences;

    // 解析输出结果
    for (int i = 0; i < output_buffer.rows; i++) {
        // 获取置信度 (第8列)
        float confidence = output_buffer.at<float>(i, 8);
        confidence = sigmoid(confidence);

        if (confidence < conf_threshold_) {
            continue;
        }

        // 获取颜色和类别得分
        cv::Mat color_scores = output_buffer.row(i).colRange(9, 13);     // 颜色 (4列)
        cv::Mat classes_scores = output_buffer.row(i).colRange(13, 22);  // 数字类别 (9列)

        cv::Point class_id, color_id;
        double score_color, score_num;
        cv::minMaxLoc(classes_scores, nullptr, &score_num, nullptr, &class_id);
        cv::minMaxLoc(color_scores, nullptr, &score_color, nullptr, &color_id);

        // 过滤不需要的颜色 (修正颜色判断逻辑)
        // color_id.x: 0=red, 1=blue, 2=none, 3=purple
        if (color_id.x == 2 || color_id.x == 3) {  // None 或 Purple
            continue;
        }

        if (detect_color == 0 && color_id.x == 1) {  // detect blue but found red
            continue;
        }

        if (detect_color == 1 && color_id.x == 0) {  // detect red but found blue
            continue;
        }

        // 创建检测对象
        Object obj;
        obj.prob = confidence;
        obj.color = color_id.x;
        obj.label = class_id.x;

        // 获取关键点坐标 (前8列) 并缩放到原始图像尺寸
        float scale_x = static_cast<float>(original_width_) / IMAGE_WIDTH;
        float scale_y = static_cast<float>(original_height_) / IMAGE_HEIGHT;

        for (int j = 0; j < 8; j++) {
            if (j % 2 == 0) {
                // x 坐标 (偶数索引)
                obj.landmarks[j] = output_buffer.at<float>(i, j) * scale_x;
            } else {
                // y 坐标 (奇数索引)
                obj.landmarks[j] = output_buffer.at<float>(i, j) * scale_y;
            }
        }

        // 计算长度和宽度 (按照示范代码修正)
        obj.length = cv::norm(
            cv::Point2f(obj.landmarks[0] - obj.landmarks[6]) -
            cv::Point2f(obj.landmarks[1] - obj.landmarks[7]));
        obj.width = cv::norm(
            cv::Point2f(obj.landmarks[0] - obj.landmarks[2]) -
            cv::Point2f(obj.landmarks[1] - obj.landmarks[3]));
        obj.ratio = obj.length / obj.width;

        // 构建四个角点 (左上逆时针 -> 左上顺时针)
        std::vector<cv::Point2f> points;
        points.reserve(4);
        points.push_back(cv::Point2f(obj.landmarks[0], obj.landmarks[1]));  // 左上
        points.push_back(cv::Point2f(obj.landmarks[6], obj.landmarks[7]));  // 右上
        points.push_back(cv::Point2f(obj.landmarks[4], obj.landmarks[5]));  // 右下
        points.push_back(cv::Point2f(obj.landmarks[2], obj.landmarks[3]));  // 左下

        // 计算边界框
        float min_x = points[0].x;
        float max_x = points[0].x;
        float min_y = points[0].y;
        float max_y = points[0].y;

        for (size_t k = 1; k < points.size(); k++) {
            min_x = std::min(min_x, points[k].x);
            max_x = std::max(max_x, points[k].x);
            min_y = std::min(min_y, points[k].y);
            max_y = std::max(max_y, points[k].y);
        }

        obj.rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);

        objects_.push_back(obj);
        boxes.push_back(obj.rect);
        confidences.push_back(score_num);
    }

    // 非极大值抑制 (NMS)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, nms_threshold_, indices);

    for (int valid_index : indices) {
        if (valid_index < static_cast<int>(objects_.size())) {
            tmp_objects_.push_back(objects_[valid_index]);
        }
    }
}

Armor AIDetector::objectToArmor(const Object & obj)
{
    // 使用关键点信息构建灯条
    cv::Point2f left_top(obj.landmarks[0], obj.landmarks[1]);
    cv::Point2f left_bottom(obj.landmarks[2], obj.landmarks[3]);
    cv::Point2f right_top(obj.landmarks[6], obj.landmarks[7]);
    cv::Point2f right_bottom(obj.landmarks[4], obj.landmarks[5]);

    // 创建虚拟的左右灯条
    Light left_light(obj.color, left_top, left_bottom);
    Light right_light(obj.color, right_top, right_bottom);
    Armor invalid_armor;
    invalid_armor.type = ArmorType::INVALID;

    // 检验灯条矩形是否越界或者为空
    if (left_light.boundingRect().area() == 0 || right_light.boundingRect().area() == 0) {
        return invalid_armor;  // 返回一个空的装甲板对象
    }
    if (left_light.boundingRect().x < 0 || left_light.boundingRect().y < 0 ||
        right_light.boundingRect().x < 0 || right_light.boundingRect().y < 0) {
        return invalid_armor;  // 返回一个空的装甲板对象
    }
    if (left_light.boundingRect().x + left_light.boundingRect().width > original_width_ ||
        left_light.boundingRect().y + left_light.boundingRect().height > original_height_ ||
        right_light.boundingRect().x + right_light.boundingRect().width > original_width_ ||
        right_light.boundingRect().y + right_light.boundingRect().height > original_height_) {
        return invalid_armor;  // 返回一个空的装甲板对象
    }

    // 创建装甲板
    Armor armor(left_light, right_light);

    // 设置数字识别结果
    std::vector<std::string> classes = {"outpost", "1",     "2",    "3",   "4",
                                        "5",       "guard", "base", "base"};
    armor.number = classes[obj.label];
    armor.confidence = obj.prob;
    std::stringstream result_ss;
    result_ss << armor.number << ": " << std::fixed << std::setprecision(1)
              << armor.confidence * 100.0 << "%";
    armor.classfication_result = result_ss.str();

    // 设置装甲板类型 (根据数字判断)
    if (armor.number == "1" || armor.number == "base")
        armor.type = ArmorType::LARGE;
    else
        armor.type = ArmorType::SMALL;

    return armor;
}

void AIDetector::drawResults(cv::Mat & img)
{
    // Draw armors
    for (const auto & armor : armors_) {
        cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
        cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
    }

    // Show numbers and confidence
    for (const auto & armor : armors_) {
        cv::putText(
            img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);
    }
}

}  // namespace rm_auto_aim