#include "imu_calibration.h"
#include "heading_control.h"
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    // Independent repeat captured after the six-pose data used for fitting.
    // Left/right here are the observer's sides when facing Rick.
    struct Reading { int pose; float range, x, y, z; };
    const Reading repeat[] = {
        {0,2,-22926.2f,-13985.1f,-14157.6f},
        {0,4,-11462.7f,-6992.8f,-7077.2f},
        {0,8,-5732.1f,-3497.4f,-3540.3f},
        {1,2,-6480.5f,-12801.0f,-14508.7f},
        {1,4,-3241.9f,-6391.5f,-7256.7f},
        {1,8,-1621.2f,-3197.0f,-3639.4f},
        {2,2,-15512.6f,-13972.4f,-6059.0f},
        {2,4,-7758.7f,-6975.1f,-3030.3f},
        {2,8,-3855.9f,-3491.9f,-1513.2f},
        {3,2,-13569.8f,-13837.2f,-22419.4f},
        {3,4,-6755.6f,-6926.6f,-11198.8f},
        {3,8,-3391.7f,-3469.0f,-5601.3f},
    };
    for (const auto &sample : repeat) {
        // Normalize raw-test ranges to the production +/-4 g count units.
        const float scale = sample.range / 4.0f;
        const auto sensor = rick::calibrated_sensor_accel(
            sample.x * scale, sample.y * scale, sample.z * scale);
        const auto robot = rick::sensor_to_robot_axes(sensor.x, sensor.y, sensor.z);
        const float norm = std::sqrt(robot.x*robot.x + robot.y*robot.y + robot.z*robot.z);
        assert(norm > 0.98f && norm < 1.025f);
        if (sample.pose == 0) {
            assert(std::abs(robot.x) < 0.015f && std::abs(robot.z) < 0.09f);
            assert(robot.y < -0.98f && robot.y > -1.02f);
            const auto policy = rick::robot_to_policy_axes(robot);
            assert(policy.z > 0.98f && policy.z < 1.02f);
            // The corrected stationary magnitude must pass the startup gyro
            // calibration's gravity gate, which rejected the biased readings.
            rick::GyroBiasCalibration gyro_calibration(10);
            const float gyro[3] = {0,0,0};
            const float accel[3] = {policy.x,policy.y,policy.z};
            for (int i=0;i<10;++i) gyro_calibration.observe(gyro,accel,true);
            assert(gyro_calibration.ready());
        } else if (sample.pose == 1) assert(robot.y > 0.98f);
        else if (sample.pose == 2) assert(robot.x < -0.98f);
        else assert(robot.x > 0.98f);
    }
    std::puts("IMU calibration passed all 12 independent repeat measurements");
}
