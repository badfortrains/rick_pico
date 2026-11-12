#ifndef RICK_IMU_WAKE_H
#define RICK_IMU_WAKE_H

#include "heading_control.h"

namespace rick {

// Also sent as byte 2 of the BLE control status. Keep these values stable.
enum class ImuWakeError : uint8_t {
    NONE = 0, POWER_DOWN = 1, BUS = 2, RESUME = 3,
    CHIP_ID = 4, CONFIG = 5, READ = 6, ACCEL = 7,
};

// IO supplies begin(), write(reg, value), configure(), read_accel(x,y,z),
// and wait_ms(ms). Keep the vendor resume sequence separate from cold boot:
// Tokmas datasheet figure 9: 0x30=0, wait 100 ms, 0xd1=0x6b then 0x68.
template<class IO>
ImuWakeError resume_imu(IO &io, float *attitude) {
    if (!io.begin()) return ImuWakeError::BUS;
    if (!io.write(0x30, 0x00)) return ImuWakeError::RESUME;
    io.wait_ms(100);
    if (!io.write(0xd1, 0x6b) || !io.write(0xd1, 0x68)) return ImuWakeError::RESUME;
    // Give restarted ADC/filter output time to settle before configuring it.
    io.wait_ms(100);
    const auto configured = io.configure();
    if (configured != ImuWakeError::NONE) return configured;

    // Startup data may be invalid or stale. Require three consecutive plausible
    // readings, separated by more than the sensor's 110 Hz sample period.
    unsigned count = 0;
    float sum[3] = {};
    bool read_any = false;
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        io.wait_ms(20);
        float ax, ay, az, candidate[4];
        const bool read = io.read_accel(ax, ay, az);
        read_any |= read;
        if (!read || !attitude_from_accel(ax, ay, az, candidate)) {
            count = 0;
            sum[0] = sum[1] = sum[2] = 0;
            continue;
        }
        sum[0] += ax; sum[1] += ay; sum[2] += az;
        if (++count == 3) {
            if (attitude_from_accel(sum[0]/3, sum[1]/3, sum[2]/3, candidate)) {
                for (unsigned i = 0; i < 4; ++i) attitude[i] = candidate[i];
                return ImuWakeError::NONE;
            }
            count = 0;
            sum[0] = sum[1] = sum[2] = 0;
        }
    }
    return read_any ? ImuWakeError::ACCEL : ImuWakeError::READ;
}

}  // namespace rick
#endif
