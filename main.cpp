#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdio.h>

#include "pico.h"
#include "boot/picobin.h"
#include "boot/picoboot.h"
#include "boot/uf2.h"
#include "btstack.h"
#include "hardware/pwm.h"
#include "hardware/spi.h"
#include "hardware/timer.h"
#include "pico/btstack_cyw43.h"
#include "pico/bootrom.h"
#include "pico/cyw43_arch.h"
#include "pico/sha256.h"
#include "pico/stdlib.h"
#include "pico/util/queue.h"

#include "policy_weights.h"
#include "rick_control.h"

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
constexpr float RESET_ACTION_FULL = 1.0f;

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
    1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f,
};

// Existing lower-leg calibration values are retained at their shifted indices.
float SERVO_CENTERS_US[ACTION_DIM] = {
    1595.0f, 1565.0f, 1690.0f, 1460.0f,
    1455.0f, 1545.0f, 1590.0f, 1570.0f,
};

// A conventional 500--2500 us servo spans approximately pi radians.
// POLICY_ACTION_SCALE_RAD is generated from the training environment config.
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

enum class ControlCommand : uint8_t {
    RESET_DEFAULT = 0,
    RESET_FULL_ACTION = 1,
    START = 2,
    STOP = 3,
};

struct ServoCalibrationCommand {
    uint8_t servo_index;
    int16_t delta_us;
};

constexpr uint8_t SERVO_CALIBRATION_FINISH = 0xff;
constexpr int16_t SERVO_CALIBRATION_MAX_STEP_US = 100;

constexpr uint8_t CONTROL_STATE_DEFAULT = 0;
constexpr uint8_t CONTROL_STATE_FULL_ACTION = 1;
constexpr uint8_t CONTROL_STATE_RUNNING = 2;
constexpr uint8_t CONTROL_STATE_STOPPED = 3;
constexpr uint8_t CONTROL_STATE_CALIBRATING = 4;

queue_t control_command_queue;
queue_t servo_calibration_command_queue;
volatile uint8_t control_state = CONTROL_STATE_DEFAULT;
volatile bool stop_requested = false;
bool robot_running = false;
bool servo_calibration_active = false;
hci_con_handle_t bluetooth_connection_handle = HCI_CON_HANDLE_INVALID;

// BLE OTA protocol. A complete UF2 stream is written to the inactive RP2350
// A/B partition. The currently running partition is never erased.
constexpr uint8_t OTA_PROTOCOL_VERSION = 1;
constexpr uint8_t OTA_CONTROL_BEGIN = 1;
constexpr uint8_t OTA_CONTROL_ABORT = 2;
constexpr uint8_t OTA_CONTROL_COMMIT = 3;
constexpr uint8_t OTA_CONTROL_DIGEST = 4;
constexpr size_t OTA_SHA256_SIZE = SHA256_RESULT_BYTES;
constexpr size_t OTA_BEGIN_PACKET_SIZE = 1 + 4 + 4;
constexpr size_t OTA_MAX_CONTROL_PACKET_SIZE = 20;
constexpr size_t OTA_MAX_DIGEST_CHUNK_SIZE = 18;
constexpr size_t OTA_MAX_DATA_VALUE_SIZE = 244;
constexpr size_t OTA_WORKAREA_SIZE = 4096;
constexpr uint32_t OTA_FLASH_SECTOR_SIZE = 4096;
constexpr uint32_t OTA_FLASH_PAGE_SIZE = 256;
constexpr uint8_t OTA_APPLICATION_PARTITION_A = 0;

enum class OtaState : uint8_t {
    IDLE = 0,
    RECEIVING = 1,
    READY = 2,
    ERROR = 3,
    REBOOTING = 4,
    PREPARING = 5,
};

enum class OtaError : uint8_t {
    NONE = 0,
    INVALID_REQUEST = 1,
    NO_PARTITION = 2,
    HASH_UNAVAILABLE = 3,
    INVALID_UF2 = 4,
    OUT_OF_ORDER = 5,
    OUT_OF_RANGE = 6,
    FLASH_OPERATION = 7,
    HASH_MISMATCH = 8,
    DISCONNECTED = 9,
    IMAGE_REJECTED = 10,
    REBOOT_FAILED = 11,
};

struct OtaControlPacket {
    uint8_t length;
    uint8_t bytes[OTA_MAX_CONTROL_PACKET_SIZE];
};

struct OtaDataPacket {
    uint16_t length;
    uint8_t bytes[OTA_MAX_DATA_VALUE_SIZE];
};

queue_t ota_control_queue;
queue_t ota_data_queue;
volatile OtaState ota_state = OtaState::IDLE;
volatile OtaError ota_error = OtaError::NONE;
volatile bool ota_abort_requested = false;
volatile bool bluetooth_ready = false;
bool pending_update_confirmation = false;
bool update_confirmation_attempted = false;
int8_t current_boot_partition = -1;
uint8_t current_boot_type = 0xff;
uint8_t current_tbyb_and_update_info = 0;
int32_t update_confirmation_result = INT32_MIN;
uint32_t current_boot_diagnostic = 0;
bool ota_sha_active = false;
uint8_t ota_expected_sha256[OTA_SHA256_SIZE] = {0};
uint32_t ota_digest_bytes_received = 0;
uint8_t ota_uf2_block_bytes[sizeof(uf2_block)] = {0};
uint32_t ota_uf2_block_used = 0;
uint32_t ota_total_bytes = 0;
uint32_t ota_bytes_received = 0;
uint32_t ota_blocks_received = 0;
uint32_t ota_target_partition = 0xff;
uint32_t ota_target_storage_base = 0;
uint32_t ota_target_storage_end = 0;
uint32_t ota_last_erased_sector = UINT32_MAX;
pico_sha256_state_t ota_sha_state;
alignas(4) uint8_t ota_workarea[OTA_WORKAREA_SIZE];

#ifdef __riscv
alignas(4) uint32_t bootrom_stack_words[256];
bootrom_stack_t bootrom_stack = {
    bootrom_stack_words,
    sizeof(bootrom_stack_words),
};
#endif

// Bluetooth telemetry uses signed milli-g for acceleration and signed
// deci-degrees/second for angular velocity, all in the physical sensor frame.
volatile int16_t imu_telemetry[6] = {0};

static btstack_packet_callback_registration_t hci_event_callback_registration;
static btstack_packet_callback_registration_t sm_event_callback_registration;

// 7e57a001-4c91-4d8e-8f2a-7dca6d5a1000, least-significant byte first.
const uint8_t BLUETOOTH_ADVERTISEMENT_DATA[] = {
    2, BLUETOOTH_DATA_TYPE_FLAGS, 0x06,
    8, BLUETOOTH_DATA_TYPE_COMPLETE_LOCAL_NAME,
    'R', 'i', 'c', 'k', 'B', 'o', 't',
    17, BLUETOOTH_DATA_TYPE_COMPLETE_LIST_OF_128_BIT_SERVICE_CLASS_UUIDS,
    0x00, 0x10, 0x5a, 0x6d, 0xca, 0x7d, 0x2a, 0x8f,
    0x8e, 0x4d, 0x91, 0x4c, 0x01, 0xa0, 0x57, 0x7e,
};
static_assert(
    sizeof(BLUETOOTH_ADVERTISEMENT_DATA) <= 31,
    "Legacy BLE advertising data cannot exceed 31 bytes");

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

int16_t clamp_to_int16(float value) {
    const float clamped = std::max(-32768.0f, std::min(32767.0f, value));
    return static_cast<int16_t>(std::lround(clamped));
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

    imu_telemetry[0] = clamp_to_int16(physical_ax * 1000.0f);
    imu_telemetry[1] = clamp_to_int16(physical_ay * 1000.0f);
    imu_telemetry[2] = clamp_to_int16(physical_az * 1000.0f);
    imu_telemetry[3] = clamp_to_int16(physical_gx * (1800.0f / PI_F));
    imu_telemetry[4] = clamp_to_int16(physical_gy * (1800.0f / PI_F));
    imu_telemetry[5] = clamp_to_int16(physical_gz * (1800.0f / PI_F));

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

void reset_policy_state(float previous_action) {
    q0 = 1.0f;
    q1 = 0.0f;
    q2 = 0.0f;
    q3 = 0.0f;
    gait_phase = 0.0f;
    step_counter = 0;
    run_control_step = false;

    for (float &value : current_obs) {
        value = 0.0f;
    }
    for (int index = 0; index < HISTORY_DIM; ++index) {
        current_obs[index] = previous_action;
    }
    current_obs[CLOCK_OFFSET] = 0.0f;
    current_obs[CLOCK_OFFSET + 1] = 1.0f;
    current_obs[TARGET_VELOCITY_OFFSET] = POLICY_TARGET_VELOCITY;
}

void reset_servos(float action) {
    robot_running = false;
    for (float &target_action : target_actions) {
        target_action = action;
    }
    reset_policy_state(action);
    update_servos(target_actions);
}

void print_servo_centers() {
    printf("\nCopy this calibration into main.cpp:\n");
    printf("float SERVO_CENTERS_US[ACTION_DIM] = {\n");
    for (int index = 0; index < ACTION_DIM; ++index) {
        if (index % 4 == 0) {
            printf("    ");
        }
        printf("%.1ff%s", SERVO_CENTERS_US[index],
               index + 1 == ACTION_DIM ? "" : ", ");
        if (index % 4 == 3 || index + 1 == ACTION_DIM) {
            printf("\n");
        }
    }
    printf("};\n\n");
}

void apply_servo_calibration_command(const ServoCalibrationCommand &command) {
    if (command.servo_index == SERVO_CALIBRATION_FINISH) {
        reset_servos(0.0f);
        servo_calibration_active = false;
        control_state = CONTROL_STATE_DEFAULT;
        print_servo_centers();
        return;
    }

    if (command.servo_index >= ACTION_DIM) {
        return;
    }

    if (!servo_calibration_active) {
        // Enter calibration at the neutral pose, never from a running or held
        // policy pose. Subsequent nudges preserve all current center values.
        reset_servos(0.0f);
        servo_calibration_active = true;
    }

    SERVO_CENTERS_US[command.servo_index] = std::max(
        SERVO_MIN_US,
        std::min(
            SERVO_MAX_US,
            SERVO_CENTERS_US[command.servo_index] + command.delta_us));
    update_servos(target_actions);
    control_state = CONTROL_STATE_CALIBRATING;
    printf(
        "Servo %u center adjusted by %+d us to %.0f us.\n",
        static_cast<unsigned>(command.servo_index),
        static_cast<int>(command.delta_us),
        SERVO_CENTERS_US[command.servo_index]);
}

void apply_control_command(ControlCommand command) {
    servo_calibration_active = false;
    switch (command) {
        case ControlCommand::RESET_DEFAULT:
            reset_servos(0.0f);
            control_state = CONTROL_STATE_DEFAULT;
            printf("Bluetooth command: reset to calibrated default pose.\n");
            break;

        case ControlCommand::RESET_FULL_ACTION:
            reset_servos(RESET_ACTION_FULL);
            control_state = CONTROL_STATE_FULL_ACTION;
            printf("Bluetooth command: reset every joint to +1.0 action.\n");
            break;

        case ControlCommand::START:
            // Preserve the held pose in command history while resetting the IMU
            // estimate and gait clock for a clean policy start.
            reset_policy_state(target_actions[0]);
            for (int history_index = 0; history_index < HISTORY_LEN; ++history_index) {
                for (int action_index = 0; action_index < ACTION_DIM; ++action_index) {
                    current_obs[history_index * ACTION_DIM + action_index] =
                        target_actions[action_index];
                }
            }
            robot_running = true;
            control_state = CONTROL_STATE_RUNNING;
            printf("Bluetooth command: start policy.\n");
            break;

        case ControlCommand::STOP:
            robot_running = false;
            run_control_step = false;
            control_state = CONTROL_STATE_STOPPED;
            printf("Bluetooth command: stop policy and hold current pose.\n");
            break;
    }
}

void process_control_commands() {
    if (stop_requested) {
        stop_requested = false;
        uint8_t discarded_command;
        while (queue_try_remove(&control_command_queue, &discarded_command)) {
        }
        ServoCalibrationCommand discarded_calibration_command;
        while (queue_try_remove(
            &servo_calibration_command_queue,
            &discarded_calibration_command)) {
        }
        apply_control_command(ControlCommand::STOP);
        return;
    }

    uint8_t command_value;
    while (queue_try_remove(&control_command_queue, &command_value)) {
        apply_control_command(static_cast<ControlCommand>(command_value));
    }
}

void process_servo_calibration_commands() {
    ServoCalibrationCommand command;
    while (queue_try_remove(&servo_calibration_command_queue, &command)) {
        apply_servo_calibration_command(command);
    }
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

uint32_t read_little_endian_u32(const uint8_t *bytes) {
    return static_cast<uint32_t>(bytes[0]) |
           (static_cast<uint32_t>(bytes[1]) << 8) |
           (static_cast<uint32_t>(bytes[2]) << 16) |
           (static_cast<uint32_t>(bytes[3]) << 24);
}

void write_little_endian_u32(uint8_t *bytes, uint32_t value) {
    bytes[0] = static_cast<uint8_t>(value);
    bytes[1] = static_cast<uint8_t>(value >> 8);
    bytes[2] = static_cast<uint8_t>(value >> 16);
    bytes[3] = static_cast<uint8_t>(value >> 24);
}

bool ota_locks_robot() {
    return ota_state == OtaState::PREPARING ||
           ota_state == OtaState::RECEIVING ||
           ota_state == OtaState::READY ||
           ota_state == OtaState::REBOOTING;
}

void drain_ota_data_queue() {
    OtaDataPacket discarded;
    while (queue_try_remove(&ota_data_queue, &discarded)) {
    }
}

void stop_robot_for_update() {
    stop_requested = false;
    robot_running = false;
    servo_calibration_active = false;
    run_control_step = false;
    control_state = CONTROL_STATE_STOPPED;

    uint8_t discarded_control;
    while (queue_try_remove(&control_command_queue, &discarded_control)) {
    }
    ServoCalibrationCommand discarded_calibration;
    while (queue_try_remove(
        &servo_calibration_command_queue, &discarded_calibration)) {
    }
}

void ota_cleanup_hash() {
    if (ota_sha_active) {
        pico_sha256_cleanup(&ota_sha_state);
        ota_sha_active = false;
    }
}

void ota_fail(OtaError error) {
    ota_cleanup_hash();
    drain_ota_data_queue();
    ota_error = error;
    ota_state = OtaState::ERROR;
    printf(
        "Firmware update stopped with error %u after %lu bytes.\n",
        static_cast<unsigned>(error),
        static_cast<unsigned long>(ota_bytes_received));
}

void ota_reset() {
    ota_cleanup_hash();
    drain_ota_data_queue();
    ota_uf2_block_used = 0;
    ota_digest_bytes_received = 0;
    ota_total_bytes = 0;
    ota_bytes_received = 0;
    ota_blocks_received = 0;
    ota_target_partition = 0xff;
    ota_target_storage_base = 0;
    ota_target_storage_end = 0;
    ota_last_erased_sector = UINT32_MAX;
    ota_error = OtaError::NONE;
    ota_state = OtaState::IDLE;
}

void initialize_bootrom_support() {
#ifdef __riscv
    const int stack_result = rom_set_bootrom_stack(&bootrom_stack);
    if (stack_result != BOOTROM_OK) {
        printf("Could not install the RP2350 boot ROM stack (%d).\n", stack_result);
    }
#endif

    boot_info_t boot_info = {};
    if (rom_get_boot_info(&boot_info)) {
        current_boot_partition = boot_info.partition;
        current_boot_type =
            static_cast<uint8_t>(rom_get_last_boot_type());
        current_tbyb_and_update_info = boot_info.tbyb_and_update_info;
        current_boot_diagnostic = boot_info.boot_diagnostic;
    }

    pending_update_confirmation =
        current_boot_type == BOOT_TYPE_FLASH_UPDATE &&
        (current_tbyb_and_update_info &
         BOOT_TBYB_AND_UPDATE_FLAG_BUY_PENDING) != 0;
    printf(
        "Booted from partition %d (type %u, update info 0x%02x, diagnostic 0x%08lx).\n",
        current_boot_partition,
        current_boot_type,
        current_tbyb_and_update_info,
        static_cast<unsigned long>(current_boot_diagnostic));
    if (pending_update_confirmation) {
        printf("Firmware is running in the try-before-you-buy window.\n");
    }
}

void confirm_pending_update_if_ready() {
    if (!pending_update_confirmation || update_confirmation_attempted ||
        !bluetooth_ready) {
        return;
    }

    update_confirmation_attempted = true;
    const int result = rom_explicit_buy(ota_workarea, sizeof(ota_workarea));
    update_confirmation_result = result;
    if (result == BOOTROM_OK) {
        pending_update_confirmation = false;
        boot_info_t boot_info = {};
        if (rom_get_boot_info(&boot_info)) {
            current_tbyb_and_update_info = boot_info.tbyb_and_update_info;
            current_boot_diagnostic = boot_info.boot_diagnostic;
        }
        printf("Firmware update accepted; this partition is now permanent.\n");
    } else {
        // Do not disable the TBYB watchdog. A failed confirmation must fall
        // back to the previously working partition.
        printf("Firmware update confirmation failed (%d); rollback remains armed.\n", result);
    }
}

void begin_ota_update(const OtaControlPacket &packet) {
    if (packet.length != OTA_BEGIN_PACKET_SIZE ||
        pending_update_confirmation) {
        ota_fail(OtaError::INVALID_REQUEST);
        return;
    }

    const uint32_t total_bytes = read_little_endian_u32(&packet.bytes[1]);
    const uint32_t family_id = read_little_endian_u32(&packet.bytes[5]);
    if (total_bytes == 0 || total_bytes % sizeof(uf2_block) != 0 ||
        family_id != RP2350_RISCV_FAMILY_ID) {
        ota_fail(OtaError::INVALID_REQUEST);
        return;
    }

    ota_reset();
    stop_robot_for_update();

    resident_partition_t target_partition = {};
    rom_flash_flush_cache();
    const int partition_result = rom_get_uf2_target_partition(
        ota_workarea,
        sizeof(ota_workarea),
        family_id,
        &target_partition);
    if (partition_result < 0) {
        ota_fail(OtaError::NO_PARTITION);
        return;
    }

    const uint16_t first_sector = static_cast<uint16_t>(
        (target_partition.permissions_and_location &
         PICOBIN_PARTITION_LOCATION_FIRST_SECTOR_BITS) >>
        PICOBIN_PARTITION_LOCATION_FIRST_SECTOR_LSB);
    const uint16_t last_sector = static_cast<uint16_t>(
        (target_partition.permissions_and_location &
         PICOBIN_PARTITION_LOCATION_LAST_SECTOR_BITS) >>
        PICOBIN_PARTITION_LOCATION_LAST_SECTOR_LSB);
    if (last_sector < first_sector) {
        ota_fail(OtaError::NO_PARTITION);
        return;
    }

    ota_target_storage_base =
        XIP_BASE + static_cast<uint32_t>(first_sector) * OTA_FLASH_SECTOR_SIZE;
    ota_target_storage_end =
        XIP_BASE + (static_cast<uint32_t>(last_sector) + 1) *
                       OTA_FLASH_SECTOR_SIZE;
    const uint32_t partition_size =
        ota_target_storage_end - ota_target_storage_base;
    const uint32_t block_count = total_bytes / sizeof(uf2_block);
    if (static_cast<uint64_t>(block_count) * OTA_FLASH_PAGE_SIZE >
        partition_size) {
        ota_fail(OtaError::OUT_OF_RANGE);
        return;
    }

    const int sha_result = pico_sha256_start_blocking(
        &ota_sha_state, SHA256_BIG_ENDIAN, false);
    if (sha_result != PICO_OK) {
        ota_fail(OtaError::HASH_UNAVAILABLE);
        return;
    }

    ota_sha_active = true;
    ota_total_bytes = total_bytes;
    ota_target_partition = static_cast<uint32_t>(partition_result);
    std::memset(ota_expected_sha256, 0, sizeof(ota_expected_sha256));
    ota_error = OtaError::NONE;
    ota_state = OtaState::PREPARING;
    printf(
        "Firmware update prepared: %lu UF2 blocks to partition %d.\n",
        static_cast<unsigned long>(block_count),
        partition_result);
}

void receive_ota_digest(const OtaControlPacket &packet) {
    if (ota_state != OtaState::PREPARING || packet.length <= 2 ||
        packet.length > OTA_MAX_CONTROL_PACKET_SIZE) {
        ota_fail(OtaError::INVALID_REQUEST);
        return;
    }

    const uint32_t digest_offset = packet.bytes[1];
    const uint32_t digest_length = packet.length - 2;
    if (digest_length > OTA_MAX_DIGEST_CHUNK_SIZE ||
        digest_offset != ota_digest_bytes_received ||
        digest_offset + digest_length > sizeof(ota_expected_sha256)) {
        ota_fail(OtaError::OUT_OF_ORDER);
        return;
    }

    std::memcpy(
        &ota_expected_sha256[digest_offset],
        &packet.bytes[2],
        digest_length);
    ota_digest_bytes_received += digest_length;
    if (ota_digest_bytes_received == sizeof(ota_expected_sha256)) {
        ota_state = OtaState::RECEIVING;
        printf("Firmware update digest received; accepting UF2 data.\n");
    }
}

bool program_ota_uf2_block() {
    const uf2_block *block =
        reinterpret_cast<const uf2_block *>(ota_uf2_block_bytes);
    const uint32_t expected_block_count =
        ota_total_bytes / sizeof(uf2_block);
    const uint32_t expected_target_address =
        XIP_BASE + ota_blocks_received * OTA_FLASH_PAGE_SIZE;

    if (block->magic_start0 != UF2_MAGIC_START0 ||
        block->magic_start1 != UF2_MAGIC_START1 ||
        block->magic_end != UF2_MAGIC_END ||
        (block->flags & UF2_FLAG_FAMILY_ID_PRESENT) == 0 ||
        (block->flags & UF2_FLAG_NOT_MAIN_FLASH) != 0 ||
        block->file_size != RP2350_RISCV_FAMILY_ID ||
        block->payload_size != OTA_FLASH_PAGE_SIZE ||
        block->block_no != ota_blocks_received ||
        block->num_blocks != expected_block_count ||
        block->target_addr != expected_target_address) {
        ota_fail(OtaError::INVALID_UF2);
        return false;
    }

    const uint32_t target_address =
        ota_target_storage_base + (block->target_addr - XIP_BASE);
    if (target_address < ota_target_storage_base ||
        target_address + OTA_FLASH_PAGE_SIZE > ota_target_storage_end) {
        ota_fail(OtaError::OUT_OF_RANGE);
        return false;
    }

    const uint32_t sector_address =
        target_address & ~(OTA_FLASH_SECTOR_SIZE - 1);
    cflash_flags_t flags = {};
    int result;
    if (sector_address != ota_last_erased_sector) {
        flags.flags =
            (CFLASH_OP_VALUE_ERASE << CFLASH_OP_LSB) |
            (CFLASH_SECLEVEL_VALUE_SECURE << CFLASH_SECLEVEL_LSB) |
            (CFLASH_ASPACE_VALUE_STORAGE << CFLASH_ASPACE_LSB);
        result = rom_flash_op(
            flags, sector_address, OTA_FLASH_SECTOR_SIZE, nullptr);
        if (result != BOOTROM_OK) {
            ota_fail(OtaError::FLASH_OPERATION);
            return false;
        }
        ota_last_erased_sector = sector_address;
    }

    flags.flags =
        (CFLASH_OP_VALUE_PROGRAM << CFLASH_OP_LSB) |
        (CFLASH_SECLEVEL_VALUE_SECURE << CFLASH_SECLEVEL_LSB) |
        (CFLASH_ASPACE_VALUE_STORAGE << CFLASH_ASPACE_LSB);
    result = rom_flash_op(
        flags,
        target_address,
        OTA_FLASH_PAGE_SIZE,
        const_cast<uint8_t *>(block->data));
    if (result != BOOTROM_OK) {
        ota_fail(OtaError::FLASH_OPERATION);
        return false;
    }

    ++ota_blocks_received;
    ota_uf2_block_used = 0;
    return true;
}

void finish_ota_receive() {
    if (!ota_sha_active || ota_uf2_block_used != 0 ||
        ota_blocks_received != ota_total_bytes / sizeof(uf2_block)) {
        ota_fail(OtaError::INVALID_UF2);
        return;
    }

    sha256_result_t actual_sha256;
    pico_sha256_finish(&ota_sha_state, &actual_sha256);
    ota_sha_active = false;
    if (std::memcmp(
            actual_sha256.bytes,
            ota_expected_sha256,
            sizeof(ota_expected_sha256)) != 0) {
        ota_fail(OtaError::HASH_MISMATCH);
        return;
    }

    // The transfer hash above proves that BLE delivered the selected UF2.
    // Ask the boot ROM to parse and verify the programmed image too, so READY
    // means this exact A/B slot is bootable rather than merely byte-complete.
    rom_flash_flush_cache();
    const int picked_partition = rom_pick_ab_partition(
        ota_workarea,
        sizeof(ota_workarea),
        OTA_APPLICATION_PARTITION_A,
        ota_target_storage_base);
    if (picked_partition < 0 ||
        static_cast<uint32_t>(picked_partition) != ota_target_partition) {
        printf(
            "Boot ROM rejected update partition %lu (selection result %d).\n",
            static_cast<unsigned long>(ota_target_partition),
            picked_partition);
        ota_fail(OtaError::IMAGE_REJECTED);
        return;
    }

    ota_state = OtaState::READY;
    printf("Firmware image verified by SHA-256 and the boot ROM; ready to boot.\n");
}

void process_ota_data_packet(const OtaDataPacket &packet) {
    if (ota_state != OtaState::RECEIVING || packet.length <= 4) {
        return;
    }

    const uint32_t stream_offset = read_little_endian_u32(packet.bytes);
    const uint32_t data_length = packet.length - 4;
    if (stream_offset != ota_bytes_received) {
        ota_fail(OtaError::OUT_OF_ORDER);
        return;
    }
    if (data_length > ota_total_bytes - ota_bytes_received) {
        ota_fail(OtaError::OUT_OF_RANGE);
        return;
    }

    pico_sha256_update_blocking(
        &ota_sha_state, &packet.bytes[4], data_length);

    uint32_t source_offset = 0;
    while (source_offset < data_length &&
           ota_state == OtaState::RECEIVING) {
        const uint32_t copy_size = std::min(
            data_length - source_offset,
            static_cast<uint32_t>(sizeof(uf2_block)) - ota_uf2_block_used);
        std::memcpy(
            &ota_uf2_block_bytes[ota_uf2_block_used],
            &packet.bytes[4 + source_offset],
            copy_size);
        ota_uf2_block_used += copy_size;
        ota_bytes_received += copy_size;
        source_offset += copy_size;

        if (ota_uf2_block_used == sizeof(uf2_block) &&
            !program_ota_uf2_block()) {
            return;
        }
    }

    if (ota_bytes_received == ota_total_bytes) {
        finish_ota_receive();
    }
}

void process_ota_control_packets() {
    OtaControlPacket packet;
    while (queue_try_remove(&ota_control_queue, &packet)) {
        if (packet.length == 0) {
            continue;
        }

        switch (packet.bytes[0]) {
            case OTA_CONTROL_BEGIN:
                begin_ota_update(packet);
                break;

            case OTA_CONTROL_ABORT:
                ota_reset();
                printf("Firmware update aborted.\n");
                break;

            case OTA_CONTROL_COMMIT: {
                if (packet.length != 1 || ota_state != OtaState::READY) {
                    ota_fail(OtaError::INVALID_REQUEST);
                    break;
                }
                ota_state = OtaState::REBOOTING;
                printf("Rebooting into the verified firmware image.\n");
                const int reboot_result = rom_reboot(
                    REBOOT2_FLAG_REBOOT_TYPE_FLASH_UPDATE |
                        REBOOT2_FLAG_NO_RETURN_ON_SUCCESS,
                    1000,
                    ota_target_storage_base,
                    0);
                // NO_RETURN_ON_SUCCESS means reaching here is always failure.
                printf("Boot ROM refused the firmware reboot (%d).\n", reboot_result);
                ota_fail(OtaError::REBOOT_FAILED);
                break;
            }

            case OTA_CONTROL_DIGEST:
                receive_ota_digest(packet);
                break;

            default:
                ota_fail(OtaError::INVALID_REQUEST);
                break;
        }
    }
}

void process_ota_data_packets() {
    OtaDataPacket packet;
    // Limit work per pass so Bluetooth and the safety stop remain responsive.
    for (int count = 0;
         count < 4 && queue_try_remove(&ota_data_queue, &packet);
         ++count) {
        process_ota_data_packet(packet);
    }
}

void process_ota_abort_request() {
    if (!ota_abort_requested) {
        return;
    }
    ota_abort_requested = false;
    if (ota_state == OtaState::PREPARING ||
        ota_state == OtaState::RECEIVING) {
        ota_fail(OtaError::DISCONNECTED);
    }
}

bool control_loop_callback(struct repeating_timer *) {
    run_control_step = true;
    return true;
}

uint16_t bluetooth_read_callback(
    hci_con_handle_t connection_handle,
    uint16_t attribute_handle,
    uint16_t offset,
    uint8_t *buffer,
    uint16_t buffer_size) {
    UNUSED(connection_handle);

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A002_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        const uint8_t state = control_state;
        return att_read_callback_handle_blob(
            &state, sizeof(state), offset, buffer, buffer_size);
    }

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A003_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        uint8_t telemetry_bytes[12];
        for (int index = 0; index < 6; ++index) {
            const uint16_t value = static_cast<uint16_t>(imu_telemetry[index]);
            telemetry_bytes[index * 2] = value & 0xff;
            telemetry_bytes[index * 2 + 1] = value >> 8;
        }
        return att_read_callback_handle_blob(
            telemetry_bytes,
            sizeof(telemetry_bytes),
            offset,
            buffer,
            buffer_size);
    }

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A004_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        uint8_t center_bytes[ACTION_DIM * 2];
        for (int index = 0; index < ACTION_DIM; ++index) {
            const uint16_t center =
                static_cast<uint16_t>(std::lround(SERVO_CENTERS_US[index]));
            center_bytes[index * 2] = center & 0xff;
            center_bytes[index * 2 + 1] = center >> 8;
        }
        return att_read_callback_handle_blob(
            center_bytes,
            sizeof(center_bytes),
            offset,
            buffer,
            buffer_size);
    }

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A007_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        uint8_t status_bytes[28] = {
            OTA_PROTOCOL_VERSION,
            static_cast<uint8_t>(ota_state),
            static_cast<uint8_t>(ota_error),
            static_cast<uint8_t>(ota_target_partition),
        };
        write_little_endian_u32(&status_bytes[4], ota_bytes_received);
        write_little_endian_u32(&status_bytes[8], ota_total_bytes);
        write_little_endian_u32(&status_bytes[12], ota_blocks_received);
        status_bytes[16] = static_cast<uint8_t>(current_boot_partition);
        status_bytes[17] = current_boot_type;
        status_bytes[18] = current_tbyb_and_update_info;
        status_bytes[19] = update_confirmation_attempted ? 1 : 0;
        write_little_endian_u32(
            &status_bytes[20],
            static_cast<uint32_t>(update_confirmation_result));
        write_little_endian_u32(&status_bytes[24], current_boot_diagnostic);
        return att_read_callback_handle_blob(
            status_bytes,
            sizeof(status_bytes),
            offset,
            buffer,
            buffer_size);
    }

    return 0;
}

int bluetooth_write_callback(
    hci_con_handle_t connection_handle,
    uint16_t attribute_handle,
    uint16_t transaction_mode,
    uint16_t offset,
    uint8_t *buffer,
    uint16_t buffer_size) {
    UNUSED(connection_handle);

    if (transaction_mode != ATT_TRANSACTION_MODE_NONE) {
        return ATT_ERROR_REQUEST_NOT_SUPPORTED;
    }
    if (offset != 0) {
        return ATT_ERROR_INVALID_OFFSET;
    }

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A005_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        if (buffer_size == 0 || buffer_size > OTA_MAX_CONTROL_PACKET_SIZE) {
            return ATT_ERROR_INVALID_ATTRIBUTE_VALUE_LENGTH;
        }

        const bool valid_size =
            (buffer[0] == OTA_CONTROL_BEGIN &&
             buffer_size == OTA_BEGIN_PACKET_SIZE) ||
            (buffer[0] == OTA_CONTROL_DIGEST &&
             buffer_size > 2) ||
            ((buffer[0] == OTA_CONTROL_ABORT ||
              buffer[0] == OTA_CONTROL_COMMIT) &&
             buffer_size == 1);
        if (!valid_size) {
            return ATT_ERROR_VALUE_NOT_ALLOWED;
        }

        OtaControlPacket packet = {};
        packet.length = static_cast<uint8_t>(buffer_size);
        std::memcpy(packet.bytes, buffer, buffer_size);
        if (!queue_try_add(&ota_control_queue, &packet)) {
            return ATT_ERROR_INSUFFICIENT_RESOURCES;
        }
        return 0;
    }

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A006_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        if (buffer_size <= 4 || buffer_size > OTA_MAX_DATA_VALUE_SIZE) {
            return ATT_ERROR_INVALID_ATTRIBUTE_VALUE_LENGTH;
        }
        if (ota_state != OtaState::RECEIVING) {
            return ATT_ERROR_WRITE_REQUEST_REJECTED;
        }

        OtaDataPacket packet = {};
        packet.length = buffer_size;
        std::memcpy(packet.bytes, buffer, buffer_size);
        if (!queue_try_add(&ota_data_queue, &packet)) {
            return ATT_ERROR_INSUFFICIENT_RESOURCES;
        }
        return 0;
    }

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A002_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        if (ota_locks_robot()) {
            return ATT_ERROR_WRITE_REQUEST_REJECTED;
        }
        if (buffer_size != 1) {
            return ATT_ERROR_INVALID_ATTRIBUTE_VALUE_LENGTH;
        }
        if (buffer[0] > static_cast<uint8_t>(ControlCommand::STOP)) {
            return ATT_ERROR_REQUEST_NOT_SUPPORTED;
        }

        if (buffer[0] == static_cast<uint8_t>(ControlCommand::STOP)) {
            // This flag is observed before the command queue so Stop cannot be
            // delayed behind other commands.
            stop_requested = true;
            return 0;
        }
        if (!queue_try_add(&control_command_queue, buffer)) {
            return ATT_ERROR_INSUFFICIENT_RESOURCES;
        }
        return 0;
    }

    if (attribute_handle ==
        ATT_CHARACTERISTIC_7E57A004_4C91_4D8E_8F2A_7DCA6D5A1000_01_VALUE_HANDLE) {
        if (ota_locks_robot()) {
            return ATT_ERROR_WRITE_REQUEST_REJECTED;
        }
        if (buffer_size != 3) {
            return ATT_ERROR_INVALID_ATTRIBUTE_VALUE_LENGTH;
        }

        int32_t delta_us =
            static_cast<int32_t>(buffer[1] | (buffer[2] << 8));
        if ((delta_us & 0x8000) != 0) {
            delta_us -= 0x10000;
        }

        const bool finish = buffer[0] == SERVO_CALIBRATION_FINISH;
        if ((finish && delta_us != 0) ||
            (!finish &&
             (buffer[0] >= ACTION_DIM || delta_us == 0 ||
              delta_us < -SERVO_CALIBRATION_MAX_STEP_US ||
              delta_us > SERVO_CALIBRATION_MAX_STEP_US))) {
            return ATT_ERROR_REQUEST_NOT_SUPPORTED;
        }

        const ServoCalibrationCommand command = {
            buffer[0], static_cast<int16_t>(delta_us)};
        if (!queue_try_add(&servo_calibration_command_queue, &command)) {
            return ATT_ERROR_INSUFFICIENT_RESOURCES;
        }
        return 0;
    }

    return 0;
}

void start_bluetooth_advertising() {
    constexpr uint16_t ADVERTISEMENT_INTERVAL = 0x00a0;  // 100 ms.
    bd_addr_t null_address = {0};
    gap_advertisements_set_params(
        ADVERTISEMENT_INTERVAL,
        ADVERTISEMENT_INTERVAL,
        0,
        0,
        null_address,
        0x07,
        0x00);
    gap_advertisements_set_data(
        sizeof(BLUETOOTH_ADVERTISEMENT_DATA),
        const_cast<uint8_t *>(BLUETOOTH_ADVERTISEMENT_DATA));
    gap_advertisements_enable(1);
}

void bluetooth_packet_handler(
    uint8_t packet_type,
    uint16_t channel,
    uint8_t *packet,
    uint16_t size) {
    UNUSED(channel);
    UNUSED(size);

    if (packet_type != HCI_EVENT_PACKET) {
        return;
    }

    switch (hci_event_packet_get_type(packet)) {
        case BTSTACK_EVENT_STATE:
            if (btstack_event_state_get_state(packet) == HCI_STATE_WORKING) {
                bluetooth_ready = true;
                bd_addr_t local_address;
                gap_local_bd_addr(local_address);
                printf(
                    "Bluetooth ready as RickBot (%s). Waiting for Android.\n",
                    bd_addr_to_str(local_address));
                start_bluetooth_advertising();
            }
            break;

        case HCI_EVENT_META_GAP:
            if (hci_event_gap_meta_get_subevent_code(packet) ==
                GAP_SUBEVENT_LE_CONNECTION_COMPLETE) {
                bluetooth_connection_handle =
                    gap_subevent_le_connection_complete_get_connection_handle(packet);
                // Android Web Bluetooth pairs transparently when an encrypted
                // characteristic is first written. Starting pairing here can
                // race Android's GATT discovery and leave its first read
                // pending until the connection times out.
                printf("Bluetooth connected; waiting for an encrypted command.\n");
            }
            break;

        case HCI_EVENT_DISCONNECTION_COMPLETE:
            if (hci_event_disconnection_complete_get_connection_handle(packet) ==
                bluetooth_connection_handle) {
                bluetooth_connection_handle = HCI_CON_HANDLE_INVALID;
                stop_requested = true;
                ota_abort_requested = true;
                printf("Bluetooth disconnected; stopping policy.\n");
                gap_advertisements_enable(1);
            }
            break;

        case SM_EVENT_JUST_WORKS_REQUEST:
            sm_just_works_confirm(
                sm_event_just_works_request_get_handle(packet));
            break;

        case SM_EVENT_PAIRING_COMPLETE:
            if (sm_event_pairing_complete_get_status(packet) == ERROR_CODE_SUCCESS) {
                printf("Bluetooth pairing complete.\n");
            } else {
                stop_requested = true;
                printf(
                    "Bluetooth pairing failed (status 0x%02x).\n",
                    sm_event_pairing_complete_get_status(packet));
            }
            break;

        default:
            break;
    }
}

bool init_bluetooth() {
    if (cyw43_arch_init() != 0) {
        printf("Failed to initialize the Pico 2 W wireless chip.\n");
        return false;
    }

    l2cap_init();
    sm_init();
    sm_set_io_capabilities(IO_CAPABILITY_NO_INPUT_NO_OUTPUT);
    sm_set_authentication_requirements(
        SM_AUTHREQ_BONDING | SM_AUTHREQ_SECURE_CONNECTION);
    att_server_init(profile_data, bluetooth_read_callback, bluetooth_write_callback);

    hci_event_callback_registration.callback = bluetooth_packet_handler;
    hci_add_event_handler(&hci_event_callback_registration);
    sm_event_callback_registration.callback = bluetooth_packet_handler;
    sm_add_event_handler(&sm_event_callback_registration);

    hci_power_control(HCI_POWER_ON);
    return true;
}

int main() {
    stdio_init_all();
    initialize_bootrom_support();
    printf(
        "Starting Rick Pico policy checkpoint %u (%d obs, %d actions)...\n",
        static_cast<unsigned>(POLICY_CHECKPOINT_STEP),
        OBS_DIM,
        ACTION_DIM);

    init_imu();
    init_servos();
    queue_init(&control_command_queue, sizeof(uint8_t), 8);
    queue_init(
        &servo_calibration_command_queue,
        sizeof(ServoCalibrationCommand),
        8);
    queue_init(&ota_control_queue, sizeof(OtaControlPacket), 4);
    queue_init(&ota_data_queue, sizeof(OtaDataPacket), 8);
    reset_servos(0.0f);
    control_state = CONTROL_STATE_DEFAULT;

    printf("\nServos at calibrated centers; policy is stopped.\n");
    if (!init_bluetooth()) {
        return 1;
    }

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
        confirm_pending_update_if_ready();
        process_ota_abort_request();
        process_ota_control_packets();
        process_ota_data_packets();

        if (!ota_locks_robot()) {
            process_control_commands();
            process_servo_calibration_commands();
        }

        if (run_control_step) {
            run_control_step = false;
            if (ota_locks_robot()) {
                tight_loop_contents();
                continue;
            }
            read_imu(&gx, &gy, &gz, &ax, &ay, &az);

            if (robot_running) {
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
        }
        tight_loop_contents();
    }
}
