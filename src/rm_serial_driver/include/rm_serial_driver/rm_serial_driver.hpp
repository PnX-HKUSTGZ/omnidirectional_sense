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
    void setRuneParam(const rclcpp::Parameter & param);

    void resetTracker();

    bool setRuneMode(uint8_t mode);
    bool setCarMode(uint8_t mode);

    // Serial port
    std::unique_ptr<IoContext> owned_ctx_;
    std::string device_name_;
    std::unique_ptr<drivers::serial_driver::SerialPortConfig> device_config_;
    std::unique_ptr<drivers::serial_driver::SerialDriver> serial_driver_;

    // Param client to set detect_colr
    using ResultFuturePtr =
        std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>;
    bool initial_set_param_ = false;
    bool initial_set_rune_param_ = false;
    uint8_t previous_receive_color_ = 0;
    rclcpp::AsyncParametersClient::SharedPtr detector_param_client_;
    ResultFuturePtr set_param_future_;
    rclcpp::AsyncParametersClient::SharedPtr rune_detector_param_client_;
    ResultFuturePtr set_rune_param_future_;

    // Service client to reset tracker
    rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr reset_tracker_client_;

    // Service client to set mode
    rclcpp::Client<auto_aim_interfaces::srv::SetMode>::SharedPtr set_rune_detector_mode_client_,
        set_rune_solver_mode_client_, set_car_detector_mode_client_, set_car_tracker_mode_client_;

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
