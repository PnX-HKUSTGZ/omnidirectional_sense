// Created by Chengfu Zou
// Copyright (C) FYT Vision Group. All rights reserved.

// std
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <optional>
#include <rclcpp/executors.hpp>
#include <thread>
// ros2
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Eigen>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/logging.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
// project

namespace rm_serial_driver
{
class VirtualSerialNode : public rclcpp::Node
{
public:
    explicit VirtualSerialNode(const rclcpp::NodeOptions & options) : Node("serial_driver", options)
    {
        RCLCPP_INFO(this->get_logger(), "Start VirtualSerialNode!");

        // Detect parameter clients (multiple armor_detector instances)
        detector_scan_timer_ = this->create_wall_timer(std::chrono::seconds(1), [this]() {
            refreshDetectorClients();
        });

        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        this->declare_parameter("color", static_cast<int>(0));
        this->declare_parameter("roll", 0.0);
        this->declare_parameter("pitch", 0.0);
        this->declare_parameter("yaw", 0.0);

        transform_stamped_.header.frame_id = "odom";
        transform_stamped_.child_frame_id = "gimbal_link";

        timer_ = this->create_wall_timer(std::chrono::milliseconds(5), [this]() {
            int color = this->get_parameter("color").as_int();
            double roll = this->get_parameter("roll").as_double();
            double pitch = this->get_parameter("pitch").as_double();
            double yaw = this->get_parameter("yaw").as_double();

            if (!initial_set_param_ || color != previous_receive_color_) {
                setParam(rclcpp::Parameter("detect_color", int(color)));
                previous_receive_color_ = color;
            }
            tf2::Quaternion q;
            q.setRPY(roll * M_PI / 180.0, -pitch * M_PI / 180.0, yaw * M_PI / 180.0);
            transform_stamped_.transform.rotation = tf2::toMsg(q);
            transform_stamped_.header.frame_id = "odom";
            transform_stamped_.child_frame_id = "gimbal_link";
            // serial_receive_data_msg.mode = mode;
            transform_stamped_.header.stamp = this->now();
            tf_broadcaster_->sendTransform(transform_stamped_);
            Eigen::Quaterniond q_eigen(q.w(), q.x(), q.y(), q.z());
            Eigen::Vector3d rpy = getRPY(q_eigen.toRotationMatrix());
            q.setRPY(rpy[0], 0, 0);
            transform_stamped_.transform.rotation = tf2::toMsg(q);
            transform_stamped_.header.frame_id = "odom";
            transform_stamped_.child_frame_id = "odom_rectify";
            tf_broadcaster_->sendTransform(transform_stamped_);
        });
    }

    void setParam(const rclcpp::Parameter & param)
    {
        last_param_ = param;
        std::lock_guard<std::mutex> lk(detectors_mutex_);
        if (detector_param_clients_.empty()) {
            RCLCPP_WARN(get_logger(), "No detector param clients found yet, will retry automatically");
            return;
        }
        for (auto & kv : detector_param_clients_) {
            const auto & name = kv.first;
            auto & client = kv.second;
            if (!client->service_is_ready()) {
                RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "[%s] param service not ready", name.c_str());
                continue;
            }
            auto & fut = set_param_futures_[name];
            if (!fut.valid() || fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                RCLCPP_INFO(get_logger(), "Setting %s.detect_color to %ld...", name.c_str(), param.as_int());
                fut = client->set_parameters({param}, [this, name, param](const ResultFuturePtr & results) {
                    for (const auto & result : results.get()) {
                        if (!result.successful) {
                            RCLCPP_ERROR(get_logger(), "[%s] Failed to set parameter: %s", name.c_str(), result.reason.c_str());
                            return;
                        }
                    }
                    RCLCPP_INFO(get_logger(), "[%s] Successfully set detect_color to %ld!", name.c_str(), param.as_int());
                    initial_set_param_ = true;
                });
            }
        }
    }



private:
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    rclcpp::TimerBase::SharedPtr timer_;
    geometry_msgs::msg::TransformStamped transform_stamped_;

    // Param clients to set detect_color for multiple detectors
    using ResultFuturePtr =
        std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>;
    bool initial_set_param_ = false;
    uint8_t previous_receive_color_ = 0;
    std::unordered_map<std::string, rclcpp::AsyncParametersClient::SharedPtr> detector_param_clients_;
    std::unordered_map<std::string, ResultFuturePtr> set_param_futures_;
    std::string detector_name_prefix_ = "armor_detector";
    rclcpp::TimerBase::SharedPtr detector_scan_timer_;
    std::mutex detectors_mutex_;
    std::optional<rclcpp::Parameter> last_param_;
    void refreshDetectorClients()
    {
        auto graph = this->get_node_graph_interface();
        auto names_and_ns = graph->get_node_names_and_namespaces();
        std::unordered_map<std::string, rclcpp::AsyncParametersClient::SharedPtr> new_map;
        for (const auto & pair : names_and_ns) {
            const auto & n = pair.first;
            const auto & ns = pair.second;
            if (n.rfind(detector_name_prefix_, 0) == 0) {
                std::string full_name = ns;
                if (full_name.empty() || full_name == "/") {
                    full_name = n;
                } else {
                    if (full_name.back() != '/') full_name += '/';
                    full_name += n;
                }
                try {
                    auto it = detector_param_clients_.find(full_name);
                    if (it != detector_param_clients_.end()) {
                        new_map.emplace(full_name, it->second);
                    } else {
                        auto client = std::make_shared<rclcpp::AsyncParametersClient>(this, full_name);
                        new_map.emplace(full_name, client);
                        RCLCPP_INFO(get_logger(), "Discovered detector node: %s", full_name.c_str());
                        if (last_param_.has_value()) {
                            if (client->service_is_ready()) {
                                RCLCPP_INFO(get_logger(), "Applying cached %s=%ld to %s", last_param_->get_name().c_str(), last_param_->as_int(), full_name.c_str());
                                client->set_parameters({*last_param_}, [this, full_name](const ResultFuturePtr & results) {
                                    for (const auto & result : results.get()) {
                                        if (!result.successful) {
                                            RCLCPP_ERROR(get_logger(), "[%s] Failed to set cached parameter: %s", full_name.c_str(), result.reason.c_str());
                                            return;
                                        }
                                    }
                                    RCLCPP_INFO(get_logger(), "[%s] Cached parameter applied", full_name.c_str());
                                });
                            }
                        }
                    }
                } catch (const std::exception & e) {
                    RCLCPP_WARN(get_logger(), "Failed to create param client for %s: %s", n.c_str(), e.what());
                }
            }
        }
        std::lock_guard<std::mutex> lk(detectors_mutex_);
        detector_param_clients_.swap(new_map);
    }
    inline Eigen::Vector3d getRPY(const Eigen::Matrix3d & rotation_matrix)
    {
        return rotation_matrix.eulerAngles(2, 1, 0).reverse();
    }
};
}  // namespace rm_serial_driver

#include "rclcpp_components/register_node_macro.hpp"

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable when its library
// is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rm_serial_driver::VirtualSerialNode)
