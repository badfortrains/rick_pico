#ifndef RICK_IMU_RAW_DIAGNOSTIC_H
#define RICK_IMU_RAW_DIAGNOSTIC_H

// Included by main.cpp only for the imu_raw_test build. Reuses the production
// I2C helpers and initialization, but never starts PWM, Bluetooth, or the policy.
bool raw_test_ready = false;

void raw_test_help() {
    printf("\nIMU RAW TEST: servo signals held low; Bluetooth/policy disabled.\n");
    printf("Hold the whole robot still, then send a pose letter:\n");
    printf("u=upright, d=upside-down, l=left side down, r=right side down,\n");
    printf("f=front down, b=back down. Each capture takes about 4 seconds.\n");
    printf("? repeats this help. Counts are uncalibrated, with no g scaling.\n");
}

void raw_test_capture(char pose) {
    const uint8_t range_bits[] = {0x00, 0x10, 0x20};
    const unsigned ranges_g[] = {2, 4, 8};
    printf("\nBEGIN pose=%c: keep still until END\n", pose);
    printf("pose,range_g,reg21,sensor_x,sensor_y,sensor_z,robot_x,robot_y,robot_z,span_x,span_y,span_z,clipped\n");
    for (unsigned range = 0; range < 3; ++range) {
        const uint8_t expected = range_bits[range] | IMU_ODR_110_HZ;
        uint8_t actual = 0;
        if (!write_imu_register(IMU_REG_ACC_CONFIG_1, expected)) return;
        delay(250);  // Discard filter transients following the range change.
        if (!read_imu_register(IMU_REG_ACC_CONFIG_1, actual)) return;
        if (actual != expected) {
            printf("ABORT: reg21 expected=0x%02x actual=0x%02x\n", expected, actual);
            return;
        }

        int32_t sums[3] = {};
        int32_t minima[3] = {32767, 32767, 32767};
        int32_t maxima[3] = {-32768, -32768, -32768};
        bool clipped = false;
        constexpr int samples = 50;
        for (int sample = 0; sample < samples; ++sample) {
            uint8_t bytes[6];
            if (!read_imu_registers(IMU_REG_ACC_DATA_XL, bytes, sizeof(bytes))) {
                printf("ABORT: sample read failed; repeat this pose.\n");
                return;
            }
            for (int axis = 0; axis < 3; ++axis) {
                const int32_t value = static_cast<int16_t>(
                    (uint16_t(bytes[2 * axis + 1]) << 8) | bytes[2 * axis]);
                sums[axis] += value;
                minima[axis] = std::min(minima[axis], value);
                maxima[axis] = std::max(maxima[axis], value);
                clipped |= value <= -32760 || value >= 32760;
            }
            delay(20);
        }
        const float sx = sums[0] / float(samples);
        const float sy = sums[1] / float(samples);
        const float sz = sums[2] / float(samples);
        const auto robot = rick::sensor_to_robot_axes(sx, sy, sz);
        printf("%c,%u,0x%02x,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%ld,%ld,%ld,%u\n",
               pose, ranges_g[range], actual, sx, sy, sz,
               robot.x, robot.y, robot.z,
               long(maxima[0] - minima[0]), long(maxima[1] - minima[1]),
               long(maxima[2] - minima[2]), unsigned(clipped));
    }
    printf("END pose=%c. You can move to the next pose.\n", pose);
}

void setup() {
    for (int pin : SERVO_PINS) {
        digitalWrite(pin, LOW);
        pinMode(pin, OUTPUT);
    }
    Serial.begin(115200);
    delay(1000);
    raw_test_ready = init_imu();
    raw_test_help();
    if (!raw_test_ready) printf("IMU initialization failed. Power cycle to retry.\n");
}

void loop() {
    if (Serial.available()) {
        const char command = char(Serial.read());
        if (command == '?') raw_test_help();
        else if (command != '\r' && command != '\n') {
            if (!raw_test_ready) printf("IMU unavailable. Power cycle to retry.\n");
            else if (command == 'u' || command == 'd' || command == 'l' ||
                     command == 'r' || command == 'f' || command == 'b') {
                raw_test_capture(command);
                // Ignore queued keystrokes so no capture starts while moving.
                while (Serial.available()) Serial.read();
            } else raw_test_help();
        }
    }
    delay(5);
}

#endif
