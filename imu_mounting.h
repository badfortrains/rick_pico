#ifndef RICK_IMU_MOUNTING_H
#define RICK_IMU_MOUNTING_H

#include "imu_axes.h"

namespace rick {

// Mounting calibration 1: chassis verified level, 2026-11-12.
// +/-4 g raw sensor mean: [-11691.3, -6792.2, -6800.7] (50 samples).
// After imu_calibration.h and nominal axis mapping, acceleration is
// [-0.074273758, -1.056205920, -0.014875961] g in right/down/forward axes.
// Shortest rotation aligning that direction with [0, -1, 0]. Gravity cannot
// establish yaw alignment. +/-2 and +/-8 g captures independently agree.
// Apply to BOTH acceleration and angular velocity, preserving magnitudes.
constexpr unsigned IMU_MOUNTING_CALIBRATION_ID = 1;
constexpr ImuVector3 correct_imu_mounting(const ImuVector3 &v) {
    return {
        0.9975369557f*v.x - 0.0701411332f*v.y - 0.0004933122f*v.z,
        0.0701411332f*v.x + 0.9974381524f*v.y + 0.0140482557f*v.z,
        -0.0004933122f*v.x - 0.0140482557f*v.y + 0.9999011967f*v.z,
    };
}

}  // namespace rick
#endif
