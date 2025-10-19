// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
#define RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_

#include <tf2_ros/transform_broadcaster.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/parameter_client.hpp>
#include <rclcpp/publisher.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription.hpp>
#include <serial_driver/serial_driver.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker.hpp>

// C++ system
#include <auto_aim_interfaces/msg/firecontrol.hpp>
#include <auto_aim_interfaces/srv/set_mode.hpp>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <optional>

#include "auto_aim_interfaces/msg/target.hpp"

namespace rm_serial_driver
{
class RMSerialDriver : public rclcpp::Node
{
public:
    explicit RMSerialDriver(const rclcpp::NodeOptions & options);

    ~RMSerialDriver() override;

private:
    void getParams();

    void receiveData();

    void sendData(const auto_aim_interfaces::msg::Firecontrol::SharedPtr msg);

    void reopenPort();

    void setParam(const rclcpp::Parameter & param);
    void resetTracker();
    void refreshDetectorClients();

    // Serial port
    std::unique_ptr<IoContext> owned_ctx_;
    std::string device_name_;
    std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

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

    // Service client to reset tracker
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_tracker_client_;


    // Aimimg point receiving from serial port for visualization
    visualization_msgs::msg::Marker aiming_point_;

    // Broadcast tf from odom to gimbal_link
    double timestamp_offset_ = 0;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::Subscription<auto_aim_interfaces::msg::Firecontrol>::SharedPtr target_sub_;

    // For debug usage
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr latency_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

    std::thread receive_thread_;

    // mode
    uint8_t mode_ = -1;
};
}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__RM_SERIAL_DRIVER_HPP_
