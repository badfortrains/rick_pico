#ifndef RICK_IMU_AXES_H
#define RICK_IMU_AXES_H

namespace rick {

struct ImuVector3 {
    float x;
    float y;
    float z;
};

// U2 is mounted on the outward-facing (left) side of the robot, with the
// servo bank at the front and S1 above S8. In that installation the Tokmas
// package axes point down, forward, and left respectively. Rick Control
// presents the right/down/forward robot frame.
constexpr ImuVector3 sensor_to_robot_axes(float sensor_x, float sensor_y,
                                          float sensor_z) {
    return {-sensor_z, sensor_x, sensor_y};
}

// The policy/MuJoCo frame is +X left, -Y forward, +Z up. Convert the
// telemetry frame before updating attitude or observations.
constexpr ImuVector3 robot_to_policy_axes(const ImuVector3 &robot) {
    return {robot.x, -robot.z, -robot.y};
}

}  // namespace rick

#endif
