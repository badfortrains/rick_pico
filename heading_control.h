#ifndef RICK_HEADING_CONTROL_H
#define RICK_HEADING_CONTROL_H

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace rick {

// Matches Joystick._heading_from_quat: IMU -Y is forward after axis mapping.
inline float heading_from_quat(float w, float x, float y, float z) {
    return std::atan2(2.0f * (w * z - x * y),
                      1.0f - 2.0f * (x * x + z * z));
}

inline void relative_heading_observation(float heading, float reference, float *out) {
    out[0] = std::sin(heading - reference);
    out[1] = std::cos(heading - reference);
}

// Average only while stopped and held still. Bias is frozen while walking.
// Slow constant rotation cannot be distinguished from gyro bias: the robot
// must actually be stationary during calibration.
class GyroBiasCalibration {
public:
    explicit GyroBiasCalibration(uint32_t samples) : samples_(samples) {}

    void reset_window() {
        count_ = 0;
        for (int i = 0; i < 6; ++i) sum_[i] = sum_squared_[i] = 0.0f;
    }

    bool observe(const float *gyro, const float *accel, bool stopped) {
        float g2 = 0.0f, a2 = 0.0f;
        for (int i = 0; i < 3; ++i) {
            g2 += gyro[i] * gyro[i];
            a2 += accel[i] * accel[i];
        }
        if (!stopped || !std::isfinite(g2) || !std::isfinite(a2) ||
            g2 > 0.04f || std::abs(std::sqrt(a2) - 1.0f) > 0.1f) {
            reset_window();
            return false;
        }
        for (int i = 0; i < 6; ++i) {
            const float value = i < 3 ? gyro[i] : accel[i - 3];
            sum_[i] += value;
            sum_squared_[i] += value * value;
        }
        if (++count_ < samples_) return false;
        bool stable = true;
        for (int i = 0; i < 6; ++i) {
            const float mean = sum_[i] / count_;
            const float variance = std::max(0.0f, sum_squared_[i] / count_ - mean * mean);
            stable &= variance <= (i < 3 ? 0.0025f : 0.0009f);
        }
        if (stable) {
            for (int i = 0; i < 3; ++i) bias_[i] = sum_[i] / count_;
            ready_ = true;
        }
        reset_window();
        return stable;
    }

    bool ready() const { return ready_; }
    float bias(int axis) const { return bias_[axis]; }

private:
    uint32_t samples_, count_ = 0;
    float sum_[6] = {}, sum_squared_[6] = {}, bias_[3] = {};
    bool ready_ = false;
};

}  // namespace rick
#endif
