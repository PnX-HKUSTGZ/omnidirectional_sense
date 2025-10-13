// Copyright (c) 2022 ChenJun
// Licensed under the Apache-2.0 License.

#ifndef RM_SERIAL_DRIVER__PACKET_HPP_
#define RM_SERIAL_DRIVER__PACKET_HPP_

#include <algorithm>
#include <cstdint>
#include <vector>

namespace rm_serial_driver
{
struct ReceivePacket
{
    uint8_t header = 0x5A;
    uint8_t detect_color : 1;  // 0-red 1-blue
    bool reset_tracker : 1;
    uint8_t set_mode : 4;  // 0-outpost 6-guard 7-base 8-rune
    uint8_t reserved : 2;
    float q[4];  // x y z w
    uint16_t checksum = 0;
} __attribute__((packed));

struct SendPacket
{
    uint8_t header = 0xA5;
    uint8_t tracking : 1;
    uint8_t iffire : 1;
    uint8_t id : 4;  // 0-outpost 6-guard 7-base  8-rune
    uint8_t reserved : 2;

    float pitch;
    float yaw;
    float x;
    float y;

    uint16_t checksum = 0;
} __attribute__((packed));

inline ReceivePacket fromVector(const std::vector<uint8_t> & data)
{
    ReceivePacket packet;
    std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t *>(&packet));
    return packet;
}

inline std::vector<uint8_t> toVector(const SendPacket & data)
{
    std::vector<uint8_t> packet(sizeof(SendPacket));
    std::copy(
        reinterpret_cast<const uint8_t *>(&data),
        reinterpret_cast<const uint8_t *>(&data) + sizeof(SendPacket), packet.begin());
    return packet;
}

}  // namespace rm_serial_driver

#endif  // RM_SERIAL_DRIVER__PACKET_HPP_
