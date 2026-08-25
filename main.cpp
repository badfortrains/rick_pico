#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdio.h>

#include "hardware/pwm.h"
#include "hardware/spi.h"
#include "hardware/timer.h"
#include "pico/stdlib.h"

#include "policy_weights.h"

// The generated header is the source of truth for the policy interface.
#define OBS_DIM POLICY_OBS_DIM
#define ACTION_DIM POLICY_ACTION_DIM
#define HISTORY_LEN POLICY_COMMAND_HISTORY_LENGTH
#define HISTORY_DIM (HISTORY_LEN * ACTION_DIM)

constexpr int GRAVITY_OFFSET = HISTORY_DIM;
constexpr int GYRO_OFFSET = GRAVITY_OFFSET + 3;
constexpr int CLOCK_OFFSET = GYRO_OFFSET + 3;
constexpr int TARGET_VELOCITY_OFFSET = CLOCK_OFFSET + 2;
constexpr float PI_F = 3.14159265358979323846f;

static_assert(ACTION_DIM == 8, "Rick v2 firmware expects eight servos");
static_assert(
    TARGET_VELOCITY_OFFSET + 1 == OBS_DIM,
    "Firmware observation layout does not match the exported policy");
static_assert(
    POLICY_OUTPUT_DIM == 2 * ACTION_DIM,
    "Expected tanh-normal mean and scale outputs");

// SPI1 and LSM6DSO pins.
#define SPI_PORT spi1
#define PIN_MISO 8
#define PIN_CS 9
#define PIN_SCK 10
#define PIN_MOSI 11

// Policy actuator order is the MuJoCo XML actuator order:
//   left roll, left upper, left lower, left foot,
//   right roll, right upper, right lower, right foot.
// GPIO 12/13 are defaults for the two roll servos newly present in this model.
// The other six assignments preserve the previous firmware's wiring.
const uint SERVO_PINS[ACTION_DIM] = {12, 4, 3, 2, 13, 7, 5, 6};

// Verify these signs and centers with the robot supported off the ground.
const float SERVO_DIRS[ACTION_DIM] = {
    -1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
};

// Existing lower-leg calibration values are retained at their shifted indices.
float SERVO_CENTERS_US[ACTION_DIM] = {
    1500.0f, 1500.0f, 1500.0f, 1460.0f,
    1500.0f, 1500.0f, 1500.0f, 1440.0f,
};

// A conventional 500--2500 us servo spans approximately pi radians. The
// training environment commands default_pose + 0.35 * policy_action radians;
// POLICY_ACTION_SCALE_RAD is generated from that environment config.
constexpr float SERVO_US_PER_RAD = 1000.0f / (PI_F / 2.0f);
constexpr float SERVO_MIN_US = 800.0f;
constexpr float SERVO_MAX_US = 2200.0f;

// LSM6DSO registers and scale factors: +/-4 g and +/-2000 degrees/second.
#define CTRL1_XL 0x10
#define CTRL2_G 0x11
#define OUTX_L_G 0x22
#define GYRO_SCALE_RAD_S (0.070f * (PI_F / 180.0f))
#define ACCEL_SCALE_G (0.122f / 1000.0f)

float current_obs[OBS_DIM] = {0.0f};
float target_actions[ACTION_DIM] = {0.0f};

float normalized_obs[OBS_DIM];
float network_buffer_a[POLICY_MAX_LAYER_DIM];
float network_buffer_b[POLICY_MAX_LAYER_DIM];

// Madgwick filter state. Identity gives projected gravity [0, 0, -1].
float q0 = 1.0f;
float q1 = 0.0f;
float q2 = 0.0f;
float q3 = 0.0f;
constexpr float MADGWICK_BETA = 0.1f;

volatile bool run_control_step = false;
uint32_t step_counter = 0;
float gait_phase = 0.0f;

inline float swish(float x) {
    // Algebraically identical to x * sigmoid(x), without exp overflow.
    if (x >= 0.0f) {
        return x / (1.0f + std::exp(-x));
    }
    const float exp_x = std::exp(x);
    return x * exp_x / (1.0f + exp_x);
}

void normalize_observation(const float *raw_obs, float *output) {
    for (int index = 0; index < OBS_DIM; ++index) {
        output[index] =
            (raw_obs[index] - POLICY_OBS_MEAN[index]) / POLICY_OBS_STD[index];
    }
}

void dense_layer(
    const float *input,
    const float *weights,
    const float *biases,
    float *output,
    int in_features,
    int out_features,
    bool apply_swish) {
    // Flax Dense kernels are [input, output] in row-major order.
    for (int output_index = 0; output_index < out_features; ++output_index) {
        float sum = biases[output_index];
        for (int input_index = 0; input_index < in_features; ++input_index) {
            sum += input[input_index] *
                   weights[input_index * out_features + output_index];
        }
        output[output_index] = apply_swish ? swish(sum) : sum;
    }
}

void infer_action(const float *raw_obs, float *action) {
    normalize_observation(raw_obs, normalized_obs);

    const float *layer_input = normalized_obs;
    float *layer_output = network_buffer_a;
    for (int layer = 0; layer < POLICY_DENSE_LAYER_COUNT; ++layer) {
        const bool is_output_layer = layer + 1 == POLICY_DENSE_LAYER_COUNT;
        dense_layer(
            layer_input,
            POLICY_LAYER_KERNELS[layer],
            POLICY_LAYER_BIASES[layer],
            layer_output,
            POLICY_LAYER_IN_DIMS[layer],
            POLICY_LAYER_OUT_DIMS[layer],
            !is_output_layer);
        layer_input = layer_output;
        layer_output =
            layer_output == network_buffer_a ? network_buffer_b : network_buffer_a;
    }

    // Brax deterministic tanh-normal mode is tanh(first half of the logits).
    for (int index = 0; index < ACTION_DIM; ++index) {
        action[index] = std::tanh(layer_input[index]);
    }
}

void madgwick_update_6dof(
    float gx, float gy, float gz, float ax, float ay, float az, float dt) {
    float q_dot_1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    float q_dot_2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    float q_dot_3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    float q_dot_4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

    const float accel_norm_squared = ax * ax + ay * ay + az * az;
    if (accel_norm_squared > 1e-12f) {
        const float accel_recip_norm = 1.0f / std::sqrt(accel_norm_squared);
        ax *= accel_recip_norm;
        ay *= accel_recip_norm;
        az *= accel_recip_norm;

        const float two_q0 = 2.0f * q0;
        const float two_q1 = 2.0f * q1;
        const float two_q2 = 2.0f * q2;
        const float two_q3 = 2.0f * q3;
        const float four_q0 = 4.0f * q0;
        const float four_q1 = 4.0f * q1;
        const float four_q2 = 4.0f * q2;
        const float eight_q1 = 8.0f * q1;
        const float eight_q2 = 8.0f * q2;
        const float q0_squared = q0 * q0;
        const float q1_squared = q1 * q1;
        const float q2_squared = q2 * q2;
        const float q3_squared = q3 * q3;

        float s0 = four_q0 * q2_squared + two_q2 * ax +
                   four_q0 * q1_squared - two_q1 * ay;
        float s1 = four_q1 * q3_squared - two_q3 * ax +
                   4.0f * q0_squared * q1 - two_q0 * ay - four_q1 +
                   eight_q1 * q1_squared + eight_q1 * q2_squared +
                   four_q1 * az;
        float s2 = 4.0f * q0_squared * q2 + two_q0 * ax +
                   four_q2 * q3_squared - two_q3 * ay - four_q2 +
                   eight_q2 * q1_squared + eight_q2 * q2_squared +
                   four_q2 * az;
        float s3 = 4.0f * q1_squared * q3 - two_q1 * ax +
                   4.0f * q2_squared * q3 - two_q2 * ay;

        const float correction_norm_squared =
            s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3;
        if (correction_norm_squared > 1e-12f) {
            const float correction_recip_norm =
                1.0f / std::sqrt(correction_norm_squared);
            s0 *= correction_recip_norm;
            s1 *= correction_recip_norm;
            s2 *= correction_recip_norm;
            s3 *= correction_recip_norm;
            q_dot_1 -= MADGWICK_BETA * s0;
            q_dot_2 -= MADGWICK_BETA * s1;
            q_dot_3 -= MADGWICK_BETA * s2;
            q_dot_4 -= MADGWICK_BETA * s3;
        }
    }

    q0 += q_dot_1 * dt;
    q1 += q_dot_2 * dt;
    q2 += q_dot_3 * dt;
    q3 += q_dot_4 * dt;

    const float quaternion_norm_squared = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3;
    if (quaternion_norm_squared > 1e-12f) {
        const float quaternion_recip_norm =
            1.0f / std::sqrt(quaternion_norm_squared);
        q0 *= quaternion_recip_norm;
        q1 *= quaternion_recip_norm;
        q2 *= quaternion_recip_norm;
        q3 *= quaternion_recip_norm;
    } else {
        q0 = 1.0f;
        q1 = q2 = q3 = 0.0f;
    }
}

void get_local_gravity(float *gravity_out) {
    // q rotates body vectors into the world frame. joystick.py observes world
    // gravity rotated by inverse(q), i.e. R(q)^T * [0, 0, -1].
    gravity_out[0] = 2.0f * (q0 * q2 - q1 * q3);
    gravity_out[1] = -2.0f * (q0 * q1 + q2 * q3);
    gravity_out[2] = -(q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
}

void write_imu_register(uint8_t reg, uint8_t data) {
    uint8_t buffer[2] = {reg, data};
    gpio_put(PIN_CS, 0);
    spi_write_blocking(SPI_PORT, buffer, 2);
    gpio_put(PIN_CS, 1);
}

void init_imu() {
    spi_init(SPI_PORT, 5000 * 1000);
    gpio_set_function(PIN_MISO, GPIO_FUNC_SPI);
    gpio_set_function(PIN_SCK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MOSI, GPIO_FUNC_SPI);

    gpio_init(PIN_CS);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_put(PIN_CS, 1);

    write_imu_register(CTRL1_XL, 0x48);  // 104 Hz, +/-4 g.
    write_imu_register(CTRL2_G, 0x4C);   // 104 Hz, +/-2000 dps.
    sleep_ms(50);
}

void read_imu(
    float *gx, float *gy, float *gz, float *ax, float *ay, float *az) {
    uint8_t reg = OUTX_L_G | 0x80;
    uint8_t buffer[12];

    gpio_put(PIN_CS, 0);
    spi_write_blocking(SPI_PORT, &reg, 1);
    spi_read_blocking(SPI_PORT, 0x00, buffer, 12);
    gpio_put(PIN_CS, 1);

    const int16_t raw_gx = static_cast<int16_t>(buffer[1] << 8 | buffer[0]);
    const int16_t raw_gy = static_cast<int16_t>(buffer[3] << 8 | buffer[2]);
    const int16_t raw_gz = static_cast<int16_t>(buffer[5] << 8 | buffer[4]);
    const int16_t raw_ax = static_cast<int16_t>(buffer[7] << 8 | buffer[6]);
    const int16_t raw_ay = static_cast<int16_t>(buffer[9] << 8 | buffer[8]);
    const int16_t raw_az = static_cast<int16_t>(buffer[11] << 8 | buffer[10]);

    const float physical_gx = raw_gx * GYRO_SCALE_RAD_S;
    const float physical_gy = raw_gy * GYRO_SCALE_RAD_S;
    const float physical_gz = raw_gz * GYRO_SCALE_RAD_S;
    const float physical_ax = raw_ax * ACCEL_SCALE_G;
    const float physical_ay = raw_ay * ACCEL_SCALE_G;
    const float physical_az = raw_az * ACCEL_SCALE_G;

    // Sensor: +X right, +Y down, +Z forward.
    // MuJoCo: +X right, +Y backward, +Z up.
    *gx = physical_gx;
    *gy = -physical_gz;
    *gz = -physical_gy;
    *ax = physical_ax;
    *ay = -physical_az;
    *az = -physical_ay;
}

void init_servos() {
    for (int index = 0; index < ACTION_DIM; ++index) {
        const uint pin = SERVO_PINS[index];
        gpio_set_function(pin, GPIO_FUNC_PWM);
        const uint slice = pwm_gpio_to_slice_num(pin);
        pwm_set_clkdiv(slice, 150.0f);  // Pico 2 default: 150 MHz -> 1 us ticks.
        pwm_set_wrap(slice, 19999);     // 20 ms / 50 Hz.
        pwm_set_chan_level(
            slice,
            pwm_gpio_to_channel(pin),
            static_cast<uint16_t>(SERVO_CENTERS_US[index]));
        pwm_set_enabled(slice, true);
    }
}

void update_servos(const float *actions) {
    const float action_range_us = POLICY_ACTION_SCALE_RAD * SERVO_US_PER_RAD;
    for (int index = 0; index < ACTION_DIM; ++index) {
        float pulse_width_us = SERVO_CENTERS_US[index] +
                               actions[index] * action_range_us * SERVO_DIRS[index];
        pulse_width_us =
            std::max(SERVO_MIN_US, std::min(SERVO_MAX_US, pulse_width_us));

        const uint slice = pwm_gpio_to_slice_num(SERVO_PINS[index]);
        pwm_set_chan_level(
            slice,
            pwm_gpio_to_channel(SERVO_PINS[index]),
            static_cast<uint16_t>(pulse_width_us));
    }
}

void initialize_observation() {
    for (float &value : current_obs) {
        value = 0.0f;
    }
    current_obs[CLOCK_OFFSET] = 0.0f;
    current_obs[CLOCK_OFFSET + 1] = 1.0f;
    current_obs[TARGET_VELOCITY_OFFSET] = POLICY_TARGET_VELOCITY;
}

void update_imu_and_clock_observation(float gx, float gy, float gz) {
    get_local_gravity(&current_obs[GRAVITY_OFFSET]);
    current_obs[GYRO_OFFSET] = POLICY_GYRO_OBS_SCALE * gx;
    current_obs[GYRO_OFFSET + 1] = POLICY_GYRO_OBS_SCALE * gy;
    current_obs[GYRO_OFFSET + 2] = POLICY_GYRO_OBS_SCALE * gz;
    current_obs[CLOCK_OFFSET] = std::sin(gait_phase);
    current_obs[CLOCK_OFFSET + 1] = std::cos(gait_phase);
    current_obs[TARGET_VELOCITY_OFFSET] = POLICY_TARGET_VELOCITY;

    // joystick.py clips the complete raw observation to [-10, 10].
    for (float &value : current_obs) {
        value = std::max(-10.0f, std::min(10.0f, value));
    }
}

void append_command_history(const float *new_action) {
    for (int index = 0; index < HISTORY_DIM - ACTION_DIM; ++index) {
        current_obs[index] = current_obs[index + ACTION_DIM];
    }
    for (int index = 0; index < ACTION_DIM; ++index) {
        current_obs[HISTORY_DIM - ACTION_DIM + index] = new_action[index];
    }
}

void advance_gait_phase() {
    gait_phase += 2.0f * PI_F * POLICY_STEP_FREQUENCY * POLICY_CONTROL_DT;
    if (gait_phase >= PI_F) {
        gait_phase -= 2.0f * PI_F;
    }
}

bool control_loop_callback(struct repeating_timer *) {
    run_control_step = true;
    return true;
}

int main() {
    stdio_init_all();
    printf(
        "Starting Rick Pico policy checkpoint %u (%d obs, %d actions)...\n",
        static_cast<unsigned>(POLICY_CHECKPOINT_STEP),
        OBS_DIM,
        ACTION_DIM);

    init_imu();
    init_servos();
    initialize_observation();

    for (float &action : target_actions) {
        action = 0.0f;
    }
    update_servos(target_actions);

    printf("\nServos at calibrated centers. Support the robot and verify all joints.\n");
    for (int seconds = 10; seconds > 0; --seconds) {
        printf("Starting in %d...\n", seconds);
        sleep_ms(1000);
    }
    printf("GO!\n\n");

    struct repeating_timer timer;
    const int64_t interval_us =
        -static_cast<int64_t>(POLICY_CONTROL_DT * 1000000.0f + 0.5f);
    add_repeating_timer_us(interval_us, control_loop_callback, nullptr, &timer);

    float gx;
    float gy;
    float gz;
    float ax;
    float ay;
    float az;

    while (true) {
        if (run_control_step) {
            run_control_step = false;

            read_imu(&gx, &gy, &gz, &ax, &ay, &az);
            madgwick_update_6dof(gx, gy, gz, ax, ay, az, POLICY_CONTROL_DT);
            update_imu_and_clock_observation(gx, gy, gz);

            infer_action(current_obs, target_actions);
            update_servos(target_actions);
            append_command_history(target_actions);
            advance_gait_phase();
            ++step_counter;

            // Uncomment for a one-line heartbeat once per second.
            // if (step_counter % 50 == 0) printf("Policy step %u\n", step_counter);
        }
        tight_loop_contents();
    }
}
