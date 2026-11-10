#ifndef RICK_IMU_CALIBRATION_H
#define RICK_IMU_CALIBRATION_H

#include "imu_axes.h"

namespace rick {

// Calibration for this assembled Rick's Tokmas IMU, measured 2026-11-09.
// Sphere fit to 18 stationary samples from six poses at +/-2, +/-4, +/-8 g.
// Converted to raw counts at the production +/-4 g setting (register 0x21=0x13).
// A separate 12-sample repeat validates the fit; these are not vendor defaults.
constexpr float IMU_CALIBRATION_RANGE_G = 4.0f;
constexpr ImuVector3 IMU_ACCEL_ZERO_COUNTS = {-7368.697f, -6731.319f, -7104.671f};
constexpr float IMU_ACCEL_COUNTS_PER_G = 4092.576f;

constexpr ImuVector3 calibrated_sensor_accel(float raw_x, float raw_y, float raw_z) {
    return {
        (raw_x - IMU_ACCEL_ZERO_COUNTS.x) / IMU_ACCEL_COUNTS_PER_G,
        (raw_y - IMU_ACCEL_ZERO_COUNTS.y) / IMU_ACCEL_COUNTS_PER_G,
        (raw_z - IMU_ACCEL_ZERO_COUNTS.z) / IMU_ACCEL_COUNTS_PER_G,
    };
}

}  // namespace rick
#endif
