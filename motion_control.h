#ifndef RICK_MOTION_CONTROL_H
#define RICK_MOTION_CONTROL_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace rick {

struct MotionCommand { float forward = 0.0f, yaw = 0.0f; };

// Control characteristic opcode 6: signed little-endian mm/s and mrad/s.
// Positive yaw is left (world +Z with model -Y forward).
inline bool decode_motion_command(const uint8_t *data, size_t size,
                                  MotionCommand minimum, MotionCommand maximum,
                                  MotionCommand &out) {
    if (size != 5 || data[0] != 6) return false;
    const auto signed16 = [](const uint8_t *p) {
        const uint16_t raw = uint16_t(p[0]) | (uint16_t(p[1]) << 8);
        return raw < 32768 ? int(raw) : int(raw) - 65536;
    };
    const MotionCommand value = {signed16(data + 1) / 1000.0f,
                                 signed16(data + 3) / 1000.0f};
    if (value.forward < minimum.forward || value.forward > maximum.forward ||
        value.yaw < minimum.yaw || value.yaw > maximum.yaw) return false;
    out = value;
    return true;
}

inline float slew_motion_value(float value, float target, float acceleration, float dt) {
    const float limit = acceleration * dt;
    return value + std::max(-limit, std::min(limit, target - value));
}

inline void step_motion_command(MotionCommand &value, MotionCommand target,
                                MotionCommand acceleration, float dt,
                                float &heading_reference) {
    value.forward = slew_motion_value(value.forward, target.forward, acceleration.forward, dt);
    value.yaw = slew_motion_value(value.yaw, target.yaw, acceleration.yaw, dt);
    constexpr float pi = 3.14159265358979323846f;
    heading_reference = std::remainder(heading_reference + value.yaw * dt, 2.0f * pi);
}

}  // namespace rick
#endif
