#include "imu_calibration.h"
#include "imu_mounting.h"
#include "heading_control.h"
#include <cassert>
#include <cmath>
#include <cstdio>

float dot(rick::ImuVector3 a, rick::ImuVector3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

int main() {
    // Verified-level capture. The +/-2 and +/-8 g rows were not used to fit
    // the rotation, and validate its direction against independent readings.
    const float rows[][4] = {
        {2, -23378.9f, -13583.7f, -13600.6f},
        {4, -11691.3f, -6792.2f, -6800.7f},
        {8, -5844.9f, -3397.0f, -3402.6f},
    };
    for (const auto &row : rows) {
        const float scale = row[0]/4;
        const auto sensor = rick::calibrated_sensor_accel(
            row[1]*scale, row[2]*scale, row[3]*scale);
        const auto nominal = rick::sensor_to_robot_axes(sensor.x, sensor.y, sensor.z);
        const auto corrected = rick::correct_imu_mounting(nominal);
        assert(std::abs(corrected.x) < .0012f);
        assert(std::abs(corrected.z) < .0012f);
        assert(corrected.y < -1.05f && corrected.y > -1.07f);
        assert(std::abs(dot(corrected, corrected) - dot(nominal, nominal)) < 1e-6f);
        const auto policy = rick::robot_to_policy_axes(corrected);
        float q[4];
        assert(rick::attitude_from_accel(policy.x, policy.y, policy.z, q));
        assert(std::abs(q[1]) < .0006f && std::abs(q[2]) < .0006f);
    }
    // Rotation must preserve the lengths, angles and handedness of gyro axes.
    const auto x = rick::correct_imu_mounting({1,0,0});
    const auto y = rick::correct_imu_mounting({0,1,0});
    const auto z = rick::correct_imu_mounting({0,0,1});
    assert(std::abs(dot(x,x)-1) < 1e-6f);
    assert(std::abs(dot(y,y)-1) < 1e-6f);
    assert(std::abs(dot(z,z)-1) < 1e-6f);
    assert(std::abs(dot(x,y)) < 1e-6f && std::abs(dot(x,z)) < 1e-6f && std::abs(dot(y,z)) < 1e-6f);
    const rick::ImuVector3 cross = {x.y*y.z-x.z*y.y, x.z*y.x-x.x*y.z, x.x*y.y-x.y*y.x};
    assert(std::abs(dot(cross,z)-1) < 1e-6f);
    std::puts("IMU mounting correction passed level captures and rotation checks");
}
