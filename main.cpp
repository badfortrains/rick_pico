#include <Arduino.h>
#include <BLEAdvertising.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLESecurity.h>
#include <Wire.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "driver/ledc.h"
#include "esp_ota_ops.h"
#include "esp_pm.h"
#include "esp_bt.h"
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "mbedtls/sha256.h"

#include "control_timing.h"
#include "gait_diagnostics.h"
#include "heading_control.h"
#include "imu_axes.h"
#include "imu_calibration.h"
#include "imu_mounting.h"
#include "motion_control.h"
#include "imu_raw_capture.h"
#include "imu_wake.h"
#include "policy_inference.h"

#if !CONFIG_PM_ENABLE || !CONFIG_FREERTOS_USE_TICKLESS_IDLE || !CONFIG_BT_CTRL_MODEM_SLEEP
#error "Rick BLE sleep requires the Arduino + ESP-IDF build and sdkconfig.defaults"
#endif

// Route diagnostics through native USB CDC instead of the exposed UART0 pins.
#define printf(...) Serial.printf(__VA_ARGS__)

#define OBS_DIM POLICY_OBS_DIM
#define ACTION_DIM POLICY_ACTION_DIM
#define HISTORY_LEN POLICY_COMMAND_HISTORY_LENGTH
#define HISTORY_DIM (HISTORY_LEN * ACTION_DIM)
#ifndef POLICY_HEADING_OBSERVATION
#define POLICY_HEADING_OBSERVATION 0
#endif
#ifndef POLICY_GYRO_BIAS_CALIBRATION
#define POLICY_GYRO_BIAS_CALIBRATION 0
#endif
#ifndef POLICY_GYRO_BIAS_CALIBRATION_SAMPLES
#define POLICY_GYRO_BIAS_CALIBRATION_SAMPLES 100
#endif

// Old exported weights remain usable at their single trained speed. They
// advertise no motion capability; exporting a new command policy enables it.
#ifndef POLICY_MOTION_COMMAND_VERSION
#define POLICY_MOTION_COMMAND_VERSION 0
const float POLICY_FORWARD_MIN = POLICY_TARGET_VELOCITY;
const float POLICY_FORWARD_MAX = POLICY_TARGET_VELOCITY;
constexpr float POLICY_YAW_MIN = 0, POLICY_YAW_MAX = 0;
constexpr float POLICY_LINEAR_ACCELERATION = 0.05f, POLICY_YAW_ACCELERATION = 1.0f;
#endif

constexpr int GRAVITY_OFFSET = HISTORY_DIM;
constexpr int GYRO_OFFSET = GRAVITY_OFFSET + 3;
constexpr int CLOCK_OFFSET = GYRO_OFFSET + 3;
constexpr int TARGET_VELOCITY_OFFSET = CLOCK_OFFSET + 2;
constexpr int TARGET_YAW_OFFSET = TARGET_VELOCITY_OFFSET + 1;
constexpr int HEADING_OFFSET = TARGET_YAW_OFFSET + POLICY_MOTION_COMMAND_VERSION;
constexpr float PI_F = 3.14159265358979323846f;

static_assert(POLICY_MOTION_COMMAND_VERSION == 0 || POLICY_MOTION_COMMAND_VERSION == 1,
              "Unsupported motion command version");
static_assert(ACTION_DIM == 8, "Rick v2 firmware expects eight servos");
static_assert(HEADING_OFFSET + 2 * POLICY_HEADING_OBSERVATION == OBS_DIM,
              "Firmware observation layout does not match the exported policy");
static_assert(POLICY_OUTPUT_DIM == 2 * ACTION_DIM,
              "Expected tanh-normal mean and scale outputs");

// ESP32-S3-Zero carrier-board wiring.
constexpr int PIN_STATUS_LED = 21;
constexpr int PIN_BATTERY_ADC = 1;  // J2 pin 4, ESP32-S3 ADC1 channel 0.
constexpr int PIN_IMU_SDA = 2;
constexpr int PIN_IMU_SCL = 4;
constexpr int PIN_IMU_INT1 = 5;
constexpr uint32_t BATTERY_DIVIDER_RATIO = 3;  // 200 kOhm / 100 kOhm divider.
constexpr int BATTERY_ADC_SAMPLES = 8;

// MuJoCo actuator order: right (Mirror) roll/upper/lower/foot, then left.
// With -Y forward and +Z up, Rick's right side is -X in the model.
constexpr int SERVO_PINS[ACTION_DIM] = {6, 7, 8, 9, 10, 11, 12, 13};
constexpr bool servo_pins_are_valid() {
    for (int first = 0; first < ACTION_DIM; ++first) {
        if (SERVO_PINS[first] == PIN_IMU_SDA ||
            SERVO_PINS[first] == PIN_IMU_SCL ||
            SERVO_PINS[first] == PIN_IMU_INT1) return false;
        for (int second = first + 1; second < ACTION_DIM; ++second) {
            if (SERVO_PINS[first] == SERVO_PINS[second]) return false;
        }
    }
    return true;
}
static_assert(servo_pins_are_valid(), "Servo and IMU GPIO assignments overlap");
constexpr float SERVO_DIRS[ACTION_DIM] = {
    // Right then left: roll, upper, knee, ankle. The six pitch joints need
    // negative pulse polarity to match the model's knees-forward basic pose.
    // Roll polarity is unchanged; its zero default target cannot verify it.
    -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,
};

float SERVO_CENTERS_US[ACTION_DIM] = {
    // Centers follow their physical servos in policy order: right, then left.
    1715.0f, 1595.0f, 1605.0f, 1560.0f,
    1340.0f, 1435.0f, 1535.0f, 1475.0f,
};

constexpr float SERVO_US_PER_RAD = 1000.0f / (PI_F / 2.0f);
constexpr float SERVO_MIN_US = 800.0f;
constexpr float SERVO_MAX_US = 2200.0f;
constexpr uint32_t SERVO_FREQUENCY_HZ = 50;
constexpr uint32_t SERVO_PERIOD_US = 1000000 / SERVO_FREQUENCY_HZ;
constexpr ledc_timer_bit_t SERVO_PWM_BITS = LEDC_TIMER_14_BIT;
constexpr uint32_t SERVO_PWM_MAX = (1u << 14) - 1;

// Tokmas ICM-42688-P (LCSC/JLCPCB C54308212), not the TDK device with the
// same part name. These registers and the 0x36 address are Tokmas-specific.
constexpr uint8_t IMU_I2C_ADDRESS = 0x36;
constexpr uint8_t IMU_REG_ACC_DATA_XL = 0x00;
constexpr uint8_t IMU_REG_CHIP_ID = 0x1f;
constexpr uint8_t IMU_REG_ACC_CONFIG_0 = 0x20;
constexpr uint8_t IMU_REG_ACC_CONFIG_1 = 0x21;
constexpr uint8_t IMU_REG_ACC_CONFIG_2 = 0x22;
constexpr uint8_t IMU_REG_GYRO_CONFIG_1 = 0x23;
constexpr uint8_t IMU_REG_GYRO_CONFIG_2 = 0x24;
constexpr uint8_t IMU_REG_GYRO_CONFIG_3 = 0x25;
constexpr uint8_t IMU_REG_AGC_CONFIG_1 = 0x2b;
constexpr uint8_t IMU_REG_ADC_CONFIG = 0xd1;
constexpr uint8_t IMU_CHIP_ID = 0xa1;
constexpr uint8_t IMU_ODR_110_HZ = 0x03;
constexpr uint8_t IMU_ACCEL_RANGE_4G = 0x10;
constexpr uint8_t IMU_GYRO_RANGE_2000_DPS = 0x60;
constexpr float GYRO_SCALE_RAD_S = (1.0f / 16.4f) * (PI_F / 180.0f);
static_assert(IMU_ACCEL_RANGE_4G == 0x10 && rick::IMU_CALIBRATION_RANGE_G == 4.0f,
              "Measured accelerometer calibration requires the +/-4 g setting");

float current_obs[OBS_DIM] = {0.0f};
float target_actions[ACTION_DIM] = {0.0f};
uint16_t applied_servo_pulse_us[ACTION_DIM] = {};
rick::PolicyNetwork policy_network;
rick::GaitDiagnostics gait_diagnostics;
portMUX_TYPE gait_diagnostics_mutex = portMUX_INITIALIZER_UNLOCKED;

void finish_gait_diagnostics(rick::GaitDiagnostics::Reason reason) {
    portENTER_CRITICAL(&gait_diagnostics_mutex);
    gait_diagnostics.finish(reason);
    portEXIT_CRITICAL(&gait_diagnostics_mutex);
}

float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
constexpr float MADGWICK_BETA = 0.1f;

const uint32_t CONTROL_PERIOD_US =
    static_cast<uint32_t>(POLICY_CONTROL_DT * 1000000.0f + 0.5f);
rick::ControlSchedule control_schedule(CONTROL_PERIOD_US);
rick::SampleClock imu_clock;
rick::GaitClock gait_clock;
rick::ControlTiming control_timing;
uint32_t step_counter = 0;
float gait_phase = 0.0f;
float heading_reference = 0.0f;
rick::MotionCommand motion_command;
rick::MotionCommand target_motion_command = {POLICY_TARGET_VELOCITY, 0.0f};
const rick::MotionCommand motion_minimum = {POLICY_FORWARD_MIN, POLICY_YAW_MIN};
const rick::MotionCommand motion_maximum = {POLICY_FORWARD_MAX, POLICY_YAW_MAX};
const rick::MotionCommand motion_acceleration = {POLICY_LINEAR_ACCELERATION, POLICY_YAW_ACCELERATION};
QueueHandle_t motion_command_queue = nullptr;
// BLE reads a short snapshot; policy state itself is owned by the main loop.
portMUX_TYPE motion_mutex = portMUX_INITIALIZER_UNLOCKED;
int16_t requested_motion_units[2] = {int16_t(POLICY_TARGET_VELOCITY * 1000), 0};
void set_target_motion(rick::MotionCommand command) {
    target_motion_command = command;
    portENTER_CRITICAL(&motion_mutex);
    requested_motion_units[0] = int16_t(std::lround(command.forward * 1000));
    requested_motion_units[1] = int16_t(std::lround(command.yaw * 1000));
    portEXIT_CRITICAL(&motion_mutex);
}
rick::GyroBiasCalibration gyro_calibration(POLICY_GYRO_BIAS_CALIBRATION_SAMPLES);

enum class ControlCommand : uint8_t {
    RESET_DEFAULT = 0, RESET_SERVO_CENTERS = 1, START = 2, STOP = 3,
    SLEEP = 4, WAKE = 5,
};
struct ServoCalibrationCommand { uint8_t servo_index; int16_t delta_us; };
constexpr uint8_t SERVO_CALIBRATION_FINISH = 0xff;
constexpr int16_t SERVO_CALIBRATION_MAX_STEP_US = 100;
constexpr uint8_t CONTROL_STATE_DEFAULT = 0;
constexpr uint8_t CONTROL_STATE_SERVO_CENTERS = 1;
constexpr uint8_t CONTROL_STATE_RUNNING = 2;
constexpr uint8_t CONTROL_STATE_STOPPED = 3;
constexpr uint8_t CONTROL_STATE_CALIBRATING = 4;
constexpr uint8_t CONTROL_STATE_SLEEPING = 5;
constexpr uint8_t CONTROL_STATE_RELAXED = 6;
constexpr uint8_t CONTROL_STATE_SLEEP_FAULT = 7;
constexpr uint8_t CONTROL_STATE_WAKING = 8;
volatile rick::ImuWakeError wake_error = rick::ImuWakeError::NONE;
volatile bool robot_sleeping = false;
bool servo_outputs_enabled = false;
uint64_t motion_ready_at = 0;
uint16_t sleep_battery_mv = 0;
esp_pm_lock_handle_t awake_cpu_lock = nullptr;
esp_pm_lock_handle_t awake_sleep_lock = nullptr;
volatile bool ble_connected = false;
esp_bd_addr_t ble_peer = {};

void configure_bluetooth_standby(bool sleeping);
void enter_robot_sleep();
void wake_robot();

void init_power_management() {
    esp_pm_config_esp32s3_t config = {};
    config.max_freq_mhz = 240;
    config.min_freq_mhz = 40;
    config.light_sleep_enable = true;
    ESP_ERROR_CHECK(esp_pm_lock_create(ESP_PM_CPU_FREQ_MAX, 0, "rick_cpu", &awake_cpu_lock));
    ESP_ERROR_CHECK(esp_pm_lock_create(ESP_PM_NO_LIGHT_SLEEP, 0, "rick_awake", &awake_sleep_lock));
    ESP_ERROR_CHECK(esp_pm_lock_acquire(awake_cpu_lock));
    ESP_ERROR_CHECK(esp_pm_lock_acquire(awake_sleep_lock));
    ESP_ERROR_CHECK(esp_pm_configure(&config));
}

QueueHandle_t control_command_queue = nullptr;
QueueHandle_t servo_calibration_command_queue = nullptr;
struct ImuRawCommand { uint16_t token; uint8_t pose; };
QueueHandle_t imu_raw_command_queue = nullptr;
rick::ImuRawCapture imu_raw_capture;
portMUX_TYPE imu_raw_mutex = portMUX_INITIALIZER_UNLOCKED;
uint8_t imu_raw_packet[rick::ImuRawCapture::PACKET_SIZE] = {1};
volatile uint8_t control_state = CONTROL_STATE_DEFAULT;
volatile bool stop_requested = false;
bool robot_running = false;
bool servo_calibration_active = false;

// Protocol 3 transfers PlatformIO's raw firmware.bin to an inactive OTA slot.
constexpr uint32_t ESP32_S3_FAMILY_ID = 0x53335045;
constexpr uint8_t OTA_PROTOCOL_VERSION = 3;
constexpr uint8_t OTA_CONTROL_BEGIN = 1;
constexpr uint8_t OTA_CONTROL_ABORT = 2;
constexpr uint8_t OTA_CONTROL_COMMIT = 3;
constexpr uint8_t OTA_CONTROL_DIGEST = 4;
constexpr size_t OTA_SHA256_SIZE = 32;
constexpr size_t OTA_BEGIN_PACKET_SIZE = 9;
constexpr size_t OTA_MAX_CONTROL_PACKET_SIZE = 20;
constexpr size_t OTA_MAX_DATA_VALUE_SIZE = 512;

enum class OtaState : uint8_t {
    IDLE = 0, RECEIVING = 1, READY = 2, ERROR = 3, REBOOTING = 4, PREPARING = 5,
};
enum class OtaError : uint8_t {
    NONE = 0, INVALID_REQUEST = 1, NO_PARTITION = 2, HASH_UNAVAILABLE = 3,
    INVALID_IMAGE = 4, OUT_OF_ORDER = 5, OUT_OF_RANGE = 6,
    FLASH_OPERATION = 7, HASH_MISMATCH = 8, DISCONNECTED = 9,
    IMAGE_REJECTED = 10, REBOOT_FAILED = 11,
};
struct OtaControlPacket {
    uint8_t length;
    uint8_t bytes[OTA_MAX_CONTROL_PACKET_SIZE];
};
struct OtaDataPacket {
    uint16_t length;
    uint8_t bytes[OTA_MAX_DATA_VALUE_SIZE];
};

QueueHandle_t ota_control_queue = nullptr;
QueueHandle_t ota_data_queue = nullptr;
volatile OtaState ota_state = OtaState::IDLE;
volatile OtaError ota_error = OtaError::NONE;
volatile bool ota_abort_requested = false;
bool ota_sha_active = false;
uint8_t ota_expected_sha256[OTA_SHA256_SIZE] = {0};
uint32_t ota_digest_bytes_received = 0;
uint32_t ota_total_bytes = 0;
uint32_t ota_bytes_received = 0;
uint32_t ota_chunks_received = 0;
uint8_t ota_target_partition = 0xff;
esp_ota_handle_t ota_handle = 0;
bool ota_writer_active = false;
const esp_partition_t *ota_partition = nullptr;
mbedtls_sha256_context ota_sha_state;

volatile int16_t imu_telemetry[6] = {0};
uint32_t imu_read_errors = 0;
uint32_t imu_consecutive_read_errors = 0;

constexpr char SERVICE_UUID[] = "7e57a001-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char CONTROL_UUID[] = "7e57a002-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char IMU_UUID[] = "7e57a003-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char SERVO_UUID[] = "7e57a004-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char OTA_CONTROL_UUID[] = "7e57a005-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char OTA_DATA_UUID[] = "7e57a006-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char OTA_STATUS_UUID[] = "7e57a007-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char BATTERY_UUID[] = "7e57a008-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char IMU_RAW_UUID[] = "7e57a009-4c91-4d8e-8f2a-7dca6d5a1000";
constexpr char GAIT_DIAGNOSTICS_UUID[] = "7e57a00a-4c91-4d8e-8f2a-7dca6d5a1000";

uint64_t time_us_64() { return static_cast<uint64_t>(esp_timer_get_time()); }

uint16_t read_battery_voltage_mv() {
    // Discard one conversion so the ADC sample-and-hold can settle through the
    // divider, then average calibrated millivolt readings to reduce noise.
    analogReadMilliVolts(PIN_BATTERY_ADC);
    uint32_t adc_mv_sum = 0;
    for (int i = 0; i < BATTERY_ADC_SAMPLES; ++i) {
        adc_mv_sum += analogReadMilliVolts(PIN_BATTERY_ADC);
    }
    const uint32_t battery_mv =
        (adc_mv_sum * BATTERY_DIVIDER_RATIO + BATTERY_ADC_SAMPLES / 2) /
        BATTERY_ADC_SAMPLES;
    return static_cast<uint16_t>(std::min<uint32_t>(battery_mv, 0xffffu));
}

void init_battery_monitor() {
    pinMode(PIN_BATTERY_ADC, INPUT);
    analogReadResolution(12);
    // The PCB uses a 2S pack: 8.4 V / 3 = 2.8 V at GPIO1. The 6 dB
    // range clips above ~1.75 V; ADC_11db supports ~3.1 V on ESP32-S3
    // (this Arduino core's name for the highest attenuation setting).
    analogSetPinAttenuation(PIN_BATTERY_ADC, ADC_11db);
    printf("Battery monitor ready on GPIO%d: %.3f V.\n", PIN_BATTERY_ADC,
           read_battery_voltage_mv() / 1000.0f);
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
        ax *= accel_recip_norm; ay *= accel_recip_norm; az *= accel_recip_norm;
        const float two_q0 = 2.0f * q0, two_q1 = 2.0f * q1;
        const float two_q2 = 2.0f * q2, two_q3 = 2.0f * q3;
        const float four_q0 = 4.0f * q0, four_q1 = 4.0f * q1;
        const float four_q2 = 4.0f * q2;
        const float eight_q1 = 8.0f * q1, eight_q2 = 8.0f * q2;
        const float q0s = q0*q0, q1s = q1*q1, q2s = q2*q2, q3s = q3*q3;
        float s0 = four_q0*q2s + two_q2*ax + four_q0*q1s - two_q1*ay;
        float s1 = four_q1*q3s - two_q3*ax + 4*q0s*q1 - two_q0*ay -
                   four_q1 + eight_q1*q1s + eight_q1*q2s + four_q1*az;
        float s2 = 4*q0s*q2 + two_q0*ax + four_q2*q3s - two_q3*ay -
                   four_q2 + eight_q2*q1s + eight_q2*q2s + four_q2*az;
        float s3 = 4*q1s*q3 - two_q1*ax + 4*q2s*q3 - two_q2*ay;
        const float correction = s0*s0 + s1*s1 + s2*s2 + s3*s3;
        if (correction > 1e-12f) {
            const float reciprocal = 1.0f / std::sqrt(correction);
            s0*=reciprocal; s1*=reciprocal; s2*=reciprocal; s3*=reciprocal;
            q_dot_1 -= MADGWICK_BETA*s0; q_dot_2 -= MADGWICK_BETA*s1;
            q_dot_3 -= MADGWICK_BETA*s2; q_dot_4 -= MADGWICK_BETA*s3;
        }
    }
    q0 += q_dot_1*dt; q1 += q_dot_2*dt; q2 += q_dot_3*dt; q3 += q_dot_4*dt;
    const float norm = q0*q0 + q1*q1 + q2*q2 + q3*q3;
    if (norm > 1e-12f) {
        const float reciprocal = 1.0f / std::sqrt(norm);
        q0*=reciprocal; q1*=reciprocal; q2*=reciprocal; q3*=reciprocal;
    } else {
        q0 = 1.0f; q1 = q2 = q3 = 0.0f;
    }
}

void get_local_gravity(float *out) {
    out[0] = 2.0f * (q0*q2 - q1*q3);
    out[1] = -2.0f * (q0*q1 + q2*q3);
    out[2] = -(q0*q0 - q1*q1 - q2*q2 + q3*q3);
}

bool write_imu_register(uint8_t reg, uint8_t data) {
    Wire.beginTransmission(IMU_I2C_ADDRESS);
    Wire.write(reg);
    Wire.write(data);
    const uint8_t error = Wire.endTransmission(true);
    if (error != 0) {
        printf("Tokmas I2C write 0x%02x <- 0x%02x failed (Wire error %u).\n",
               reg, data, error);
    }
    return error == 0;
}

bool read_imu_registers(uint8_t first_reg, uint8_t *data, size_t length) {
    Wire.beginTransmission(IMU_I2C_ADDRESS);
    Wire.write(first_reg);
    const uint8_t address_error = Wire.endTransmission(false);
    if (address_error != 0) {
        printf("Tokmas I2C read address 0x%02x failed (Wire error %u).\n",
               first_reg, address_error);
        return false;
    }
    const size_t received = Wire.requestFrom(
        IMU_I2C_ADDRESS, static_cast<uint8_t>(length), static_cast<uint8_t>(true));
    if (received != length) {
        printf("Tokmas I2C short read at 0x%02x: received %u of %u bytes.\n",
               first_reg, static_cast<unsigned>(received),
               static_cast<unsigned>(length));
        while (Wire.available()) Wire.read();
        return false;
    }
    for (size_t index = 0; index < length; ++index) {
        data[index] = static_cast<uint8_t>(Wire.read());
    }
    return true;
}

bool read_imu_register(uint8_t reg, uint8_t &value) {
    return read_imu_registers(reg, &value, 1);
}

struct ImuRawIO {
    bool configure(uint8_t value) { return write_imu_register(IMU_REG_ACC_CONFIG_1, value); }
    bool config(uint8_t &value) { return read_imu_register(IMU_REG_ACC_CONFIG_1, value); }
    bool read(int16_t values[3]) {
        uint8_t bytes[6];
        if (!read_imu_registers(IMU_REG_ACC_DATA_XL, bytes, sizeof(bytes))) return false;
        for (int axis = 0; axis < 3; ++axis)
            values[axis] = int16_t(uint16_t(bytes[2 * axis]) | (uint16_t(bytes[2 * axis + 1]) << 8));
        return true;
    }
} imu_raw_io;

void publish_imu_raw() {
    uint8_t packet[rick::ImuRawCapture::PACKET_SIZE];
    imu_raw_capture.encode(packet);
    portENTER_CRITICAL(&imu_raw_mutex);
    std::memcpy(imu_raw_packet, packet, sizeof(packet));
    portEXIT_CRITICAL(&imu_raw_mutex);
}
void cancel_imu_raw() {
    imu_raw_capture.cancel(imu_raw_io);
    publish_imu_raw();
    imu_clock.reset();
    gyro_calibration.reset_window();
}
bool imu_raw_locks_robot() {
    return imu_raw_capture.active() || imu_raw_capture.restore_failed;
}

rick::ImuWakeError configure_imu_registers() {
    uint8_t chip_id = 0;
    if (!read_imu_register(IMU_REG_CHIP_ID, chip_id) || chip_id != IMU_CHIP_ID) {
        printf("Tokmas IMU chip ID mismatch: expected 0x%02x, read 0x%02x.\n",
               IMU_CHIP_ID, chip_id);
        return rick::ImuWakeError::CHIP_ID;
    }

    // Enable digital filters. Use 110 Hz with 44 Hz cutoff, +/-4 g, and
    // +/-2000 dps: the closest Tokmas settings to the old 104 Hz setup.
    if (!write_imu_register(IMU_REG_ACC_CONFIG_0, 0x01) ||
        !write_imu_register(IMU_REG_ACC_CONFIG_1,
                            IMU_ACCEL_RANGE_4G | IMU_ODR_110_HZ) ||
        !write_imu_register(IMU_REG_ACC_CONFIG_2, 0x00) ||
        !write_imu_register(IMU_REG_GYRO_CONFIG_1, 0x01) ||
        !write_imu_register(IMU_REG_GYRO_CONFIG_2,
                            IMU_GYRO_RANGE_2000_DPS | IMU_ODR_110_HZ) ||
        !write_imu_register(IMU_REG_GYRO_CONFIG_3, 0x00)) {
        return rick::ImuWakeError::CONFIG;
    }
    delay(20);

    uint8_t accel_config = 0, gyro_config = 0;
    const bool configured =
        read_imu_register(IMU_REG_ACC_CONFIG_1, accel_config) &&
        read_imu_register(IMU_REG_GYRO_CONFIG_2, gyro_config) &&
        accel_config == (IMU_ACCEL_RANGE_4G | IMU_ODR_110_HZ) &&
        gyro_config == (IMU_GYRO_RANGE_2000_DPS | IMU_ODR_110_HZ);
    if (configured) {
        printf("Tokmas ICM-42688-P ready at I2C address 0x36.\n");
    } else {
        printf("Tokmas configuration readback failed: accel=0x%02x gyro=0x%02x.\n",
               accel_config, gyro_config);
    }
    return configured ? rick::ImuWakeError::NONE : rick::ImuWakeError::CONFIG;
}

bool init_imu() {
    if (!Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL, 400000)) return false;
    Wire.setTimeOut(10);
    pinMode(PIN_IMU_INT1, INPUT);
    // Tokmas specifies up to 320 ms from power-on before register access.
    delay(320);

    // Tokmas quick-start initialization sequence (datasheet figure 5).
    if (!write_imu_register(IMU_REG_AGC_CONFIG_1, 0x01) ||
        !write_imu_register(IMU_REG_ACC_DATA_XL, 0x73)) {
        return false;
    }
    delay(100);
    if (!write_imu_register(IMU_REG_ADC_CONFIG, 0x6b) ||
        !write_imu_register(IMU_REG_ADC_CONFIG, 0x68)) {
        return false;
    }

    return configure_imu_registers() == rick::ImuWakeError::NONE;
}

int16_t clamp_to_int16(float value) {
    return static_cast<int16_t>(std::lround(
        std::max(-32768.0f, std::min(32767.0f, value))));
}

bool read_imu(float *gx, float *gy, float *gz, float *ax, float *ay, float *az,
              int16_t *raw_counts = nullptr) {
    uint8_t buffer[12];
    if (!read_imu_registers(IMU_REG_ACC_DATA_XL, buffer, sizeof(buffer))) return false;
    const int16_t raw_ax = static_cast<int16_t>(buffer[1]<<8 | buffer[0]);
    const int16_t raw_ay = static_cast<int16_t>(buffer[3]<<8 | buffer[2]);
    const int16_t raw_az = static_cast<int16_t>(buffer[5]<<8 | buffer[4]);
    const int16_t raw_gx = static_cast<int16_t>(buffer[7]<<8 | buffer[6]);
    const int16_t raw_gy = static_cast<int16_t>(buffer[9]<<8 | buffer[8]);
    const int16_t raw_gz = static_cast<int16_t>(buffer[11]<<8 | buffer[10]);
    if (raw_counts) {
        const int16_t counts[6] = {raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz};
        std::memcpy(raw_counts, counts, sizeof(counts));
    }
    const float pgx=raw_gx*GYRO_SCALE_RAD_S, pgy=raw_gy*GYRO_SCALE_RAD_S;
    const float pgz=raw_gz*GYRO_SCALE_RAD_S;
    const auto sensor_accel = rick::calibrated_sensor_accel(raw_ax, raw_ay, raw_az);
    const rick::ImuVector3 robot_accel = rick::correct_imu_mounting(rick::sensor_to_robot_axes(
        sensor_accel.x, sensor_accel.y, sensor_accel.z));
    const rick::ImuVector3 robot_gyro = rick::correct_imu_mounting(
        rick::sensor_to_robot_axes(pgx, pgy, pgz));
    imu_telemetry[0]=clamp_to_int16(robot_accel.x*1000);
    imu_telemetry[1]=clamp_to_int16(robot_accel.y*1000);
    imu_telemetry[2]=clamp_to_int16(robot_accel.z*1000);
    imu_telemetry[3]=clamp_to_int16(robot_gyro.x*(1800/PI_F));
    imu_telemetry[4]=clamp_to_int16(robot_gyro.y*(1800/PI_F));
    imu_telemetry[5]=clamp_to_int16(robot_gyro.z*(1800/PI_F));
    const rick::ImuVector3 policy_accel = rick::robot_to_policy_axes(robot_accel);
    const rick::ImuVector3 policy_gyro = rick::robot_to_policy_axes(robot_gyro);
    *gx=policy_gyro.x; *gy=policy_gyro.y; *gz=policy_gyro.z;
    *ax=policy_accel.x; *ay=policy_accel.y; *az=policy_accel.z;
    return true;
}

struct ImuWakeIO {
    bool begin() {
        // Recreate the driver/clock configuration after automatic light sleep.
        Wire.end();
        const bool ready = Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL, 400000);
        Wire.setTimeOut(10);
        return ready;
    }
    bool write(uint8_t reg, uint8_t value) { return write_imu_register(reg, value); }
    rick::ImuWakeError configure() { return configure_imu_registers(); }
    bool read_accel(float &ax, float &ay, float &az) {
        float gx, gy, gz;
        return read_imu(&gx, &gy, &gz, &ax, &ay, &az);
    }
    void wait_ms(unsigned ms) { delay(ms); }
} imu_wake_io;

float servo_pulse_width_us(int index, float action, bool use_default_pose=true) {
    const float target = (use_default_pose ? POLICY_DEFAULT_POSE_RAD[index] : 0) +
                         action * POLICY_ACTION_SCALE_RAD;
    return std::max(SERVO_MIN_US, std::min(SERVO_MAX_US,
        SERVO_CENTERS_US[index] + target*SERVO_US_PER_RAD*SERVO_DIRS[index]));
}
uint32_t servo_duty(float pulse_us) {
    return static_cast<uint32_t>(pulse_us * SERVO_PWM_MAX / SERVO_PERIOD_US + .5f);
}
void init_servos() {
    ledc_timer_config_t timer = {};
    timer.speed_mode=LEDC_LOW_SPEED_MODE; timer.duty_resolution=SERVO_PWM_BITS;
    timer.timer_num=LEDC_TIMER_0; timer.freq_hz=SERVO_FREQUENCY_HZ;
    timer.clk_cfg=LEDC_AUTO_CLK; ESP_ERROR_CHECK(ledc_timer_config(&timer));
    for (int i=0;i<ACTION_DIM;++i) {
        ledc_channel_config_t channel = {};
        channel.gpio_num=SERVO_PINS[i]; channel.speed_mode=LEDC_LOW_SPEED_MODE;
        channel.channel=static_cast<ledc_channel_t>(i); channel.intr_type=LEDC_INTR_DISABLE;
        channel.timer_sel=LEDC_TIMER_0; channel.duty=servo_duty(servo_pulse_width_us(i,0));
        ESP_ERROR_CHECK(ledc_channel_config(&channel));
    }
}
void update_servos(const float *actions, bool use_default_pose=true) {
    if (!servo_outputs_enabled) {
        ESP_ERROR_CHECK(ledc_timer_resume(LEDC_LOW_SPEED_MODE, LEDC_TIMER_0));
        servo_outputs_enabled = true;
    }
    for (int i=0;i<ACTION_DIM;++i) {
        const auto channel=static_cast<ledc_channel_t>(i);
        const float pulse = servo_pulse_width_us(i,actions[i],use_default_pose);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, channel,
            servo_duty(pulse));
        ledc_update_duty(LEDC_LOW_SPEED_MODE, channel);
        applied_servo_pulse_us[i] = uint16_t(std::lround(pulse));
    }
}

void start_gait_diagnostics() {
    rick::GaitDiagnostics::Metadata meta;
    meta.checkpoint = POLICY_CHECKPOINT_STEP;
    meta.control_dt = POLICY_CONTROL_DT; meta.frequency = POLICY_STEP_FREQUENCY;
    meta.target_velocity = POLICY_TARGET_VELOCITY; meta.action_scale = POLICY_ACTION_SCALE_RAD;
    meta.heading_reference = heading_reference; meta.gyro_scale = GYRO_SCALE_RAD_S;
    meta.gyro_obs_scale = POLICY_GYRO_OBS_SCALE; meta.beta = MADGWICK_BETA;
    meta.obs_dim = OBS_DIM; meta.heading_enabled = POLICY_HEADING_OBSERVATION;
    meta.calibration_enabled = POLICY_GYRO_BIAS_CALIBRATION;
    // Three cycles plus the initial control interval, bounded independently of
    // a checkpoint's cadence. Slow checkpoints get a clearly bounded 4.5 s log.
    meta.duration_us = uint32_t(std::min(4500000.0f,
        3000000.0f / std::max(POLICY_STEP_FREQUENCY, 0.1f) + CONTROL_PERIOD_US));
    meta.accel_config = IMU_ACCEL_RANGE_4G | IMU_ODR_110_HZ;
    meta.gyro_config = IMU_GYRO_RANGE_2000_DPS | IMU_ODR_110_HZ;
    meta.accel_zero[0] = rick::IMU_ACCEL_ZERO_COUNTS.x;
    meta.accel_zero[1] = rick::IMU_ACCEL_ZERO_COUNTS.y;
    meta.accel_zero[2] = rick::IMU_ACCEL_ZERO_COUNTS.z;
    meta.accel_counts_per_g = rick::IMU_ACCEL_COUNTS_PER_G;
    meta.mounting_calibration_id = rick::IMU_MOUNTING_CALIBRATION_ID;
    const float quat[4] = {q0, q1, q2, q3};
    std::memcpy(meta.initial_quat, quat, sizeof(quat));
    std::memcpy(meta.initial_action, target_actions, sizeof(target_actions));
    for (int i = 0; i < 3; ++i)
        meta.bias[i] = POLICY_GYRO_BIAS_CALIBRATION ? gyro_calibration.bias(i) : 0.0f;
    for (int i = 0; i < ACTION_DIM; ++i) meta.centers[i] = uint16_t(std::lround(SERVO_CENTERS_US[i]));
    std::strncpy(meta.build, __DATE__ " " __TIME__, sizeof(meta.build) - 1);
    portENTER_CRITICAL(&gait_diagnostics_mutex);
    gait_diagnostics.start(time_us_64(), meta);
    portEXIT_CRITICAL(&gait_diagnostics_mutex);
}

void record_gait_diagnostics(rick::GaitDiagnostics::Frame &frame) {
    const float quat[4] = {q0, q1, q2, q3};
    std::memcpy(frame.quat, quat, sizeof(quat));
    get_local_gravity(frame.gravity);
    frame.phase = gait_phase;
    frame.heading = rick::heading_from_quat(q0, q1, q2, q3);
    std::memcpy(frame.action, target_actions, sizeof(target_actions));
    std::memcpy(frame.pulse_us, applied_servo_pulse_us, sizeof(applied_servo_pulse_us));
    frame.read_errors = imu_read_errors;
    if (gyro_calibration.ready()) frame.flags |= rick::GaitDiagnostics::BIAS_READY;
    if (servo_outputs_enabled) frame.flags |= rick::GaitDiagnostics::PWM_ENABLED;
    for (int i = 0; i < 3; ++i)
        frame.bias[i] = POLICY_GYRO_BIAS_CALIBRATION ? gyro_calibration.bias(i) : 0.0f;
    portENTER_CRITICAL(&gait_diagnostics_mutex);
    gait_diagnostics.record(frame, robot_running);
    portEXIT_CRITICAL(&gait_diagnostics_mutex);
}

void reset_policy_history_and_clock(float previous_action) {
    gait_phase=0; gait_clock.reset(); const uint64_t now=time_us_64();
    control_timing.reset(now); step_counter=0; control_schedule.reset(now);
    std::fill(std::begin(current_obs),std::end(current_obs),0.0f);
    std::fill(current_obs,current_obs+HISTORY_DIM,previous_action);
    current_obs[CLOCK_OFFSET]=0; current_obs[CLOCK_OFFSET+1]=1;
    motion_command = {};
    current_obs[TARGET_VELOCITY_OFFSET]=POLICY_MOTION_COMMAND_VERSION ? 0 : POLICY_TARGET_VELOCITY;
    if constexpr(POLICY_MOTION_COMMAND_VERSION) current_obs[TARGET_YAW_OFFSET]=0;
    heading_reference=rick::heading_from_quat(q0,q1,q2,q3);
    if constexpr(POLICY_HEADING_OBSERVATION)
        rick::relative_heading_observation(heading_reference,heading_reference,
                                            &current_obs[HEADING_OFFSET]);
}
void reset_servos(float action, bool use_default_pose=true) {
    finish_gait_diagnostics(rick::GaitDiagnostics::RESET);
    robot_running=false; std::fill(std::begin(target_actions),std::end(target_actions),action);
    reset_policy_history_and_clock(action); update_servos(target_actions,use_default_pose);
}
void print_servo_centers() {
    printf("\nCopy this calibration into main.cpp:\nfloat SERVO_CENTERS_US[ACTION_DIM] = {\n");
    for(int i=0;i<ACTION_DIM;++i){ if(i%4==0)printf("    ");
        printf("%.1ff%s",SERVO_CENTERS_US[i],i+1==ACTION_DIM?"":", ");
        if(i%4==3||i+1==ACTION_DIM)printf("\n"); }
    printf("};\n\n");
}
void apply_servo_calibration_command(const ServoCalibrationCommand &command) {
    if (imu_raw_locks_robot()) return;
    if(command.servo_index==SERVO_CALIBRATION_FINISH){reset_servos(0);servo_calibration_active=false;
        control_state=CONTROL_STATE_DEFAULT;print_servo_centers();return;}
    if(command.servo_index>=ACTION_DIM)return;
    if(!servo_calibration_active){reset_servos(0,false);servo_calibration_active=true;}
    SERVO_CENTERS_US[command.servo_index]=std::max(SERVO_MIN_US,std::min(SERVO_MAX_US,
        SERVO_CENTERS_US[command.servo_index]+command.delta_us));
    update_servos(target_actions,false);control_state=CONTROL_STATE_CALIBRATING;
    printf("Servo %u center adjusted by %+d us to %.0f us.\n",command.servo_index,
           command.delta_us,SERVO_CENTERS_US[command.servo_index]);
}
void apply_control_command(ControlCommand command) {
    if (robot_sleeping) {
        if (command == ControlCommand::WAKE) wake_robot();
        return;
    }
    if (command == ControlCommand::WAKE) return;
    if (command == ControlCommand::SLEEP) { enter_robot_sleep(); return; }
    if (command == ControlCommand::STOP) cancel_imu_raw();
    else if (imu_raw_locks_robot()) return;
    servo_calibration_active=false;
    switch(command){
    case ControlCommand::RESET_DEFAULT: reset_servos(0);control_state=CONTROL_STATE_DEFAULT;
        printf("Bluetooth command: reset to calibrated default pose.\n");break;
    case ControlCommand::RESET_SERVO_CENTERS: reset_servos(0,false);
        control_state=CONTROL_STATE_SERVO_CENTERS;
        printf("Bluetooth command: reset every servo to its raw calibrated center.\n");break;
    case ControlCommand::START:
        if (time_us_64() < motion_ready_at) return;
        if(POLICY_GYRO_BIAS_CALIBRATION&&!gyro_calibration.ready()){
            printf("Hold Rick still for gyro calibration before Start (%.1f seconds).\n",
                   POLICY_GYRO_BIAS_CALIBRATION_SAMPLES*POLICY_CONTROL_DT);break;}
        gyro_calibration.reset_window();reset_policy_history_and_clock(target_actions[0]);
        for(int h=0;h<HISTORY_LEN;++h)for(int a=0;a<ACTION_DIM;++a)
            current_obs[h*ACTION_DIM+a]=target_actions[a];
        start_gait_diagnostics();
        robot_running=true;control_state=CONTROL_STATE_RUNNING;printf("Bluetooth command: start policy.\n");break;
    case ControlCommand::STOP: finish_gait_diagnostics(rick::GaitDiagnostics::STOP);
        robot_running=false;control_schedule.reset(time_us_64());
        if constexpr(POLICY_MOTION_COMMAND_VERSION) set_target_motion({});
        control_state=servo_outputs_enabled ? CONTROL_STATE_STOPPED : CONTROL_STATE_RELAXED;
        printf("Bluetooth command: stop policy.\n");break;
    case ControlCommand::SLEEP: case ControlCommand::WAKE: break;
    }
}

template<typename T> void drain_queue(QueueHandle_t queue){T item;while(xQueueReceive(queue,&item,0)==pdTRUE){}}
// Tokmas datasheet figure 8; this register has no documented readback contract.
bool power_down_imu() { return write_imu_register(0x30, 0x07); }

void enter_robot_sleep() {
    finish_gait_diagnostics(rick::GaitDiagnostics::SLEEP);
    robot_running = false;
    servo_calibration_active = false;
    cancel_imu_raw();
    // Drive every signal LOW, not floating, while the servo supply remains on.
    for (int i = 0; i < ACTION_DIM; ++i)
        ESP_ERROR_CHECK(ledc_stop(LEDC_LOW_SPEED_MODE, static_cast<ledc_channel_t>(i), 0));
    ESP_ERROR_CHECK(ledc_timer_pause(LEDC_LOW_SPEED_MODE, LEDC_TIMER_0));
    servo_outputs_enabled = false;
    control_state = CONTROL_STATE_RELAXED;
    // Close the callback gate before discarding commands that raced with Sleep.
    robot_sleeping = true;
    drain_queue<uint8_t>(control_command_queue);
    drain_queue<ServoCalibrationCommand>(servo_calibration_command_queue);
    drain_queue<ImuRawCommand>(imu_raw_command_queue);
    drain_queue<rick::MotionCommand>(motion_command_queue);
    if constexpr(POLICY_MOTION_COMMAND_VERSION) set_target_motion({});
    drain_queue<OtaControlPacket>(ota_control_queue);
    drain_queue<OtaDataPacket>(ota_data_queue);
    stop_requested = false;
    uint8_t sleep_state = CONTROL_STATE_SLEEPING;
    wake_error = rick::ImuWakeError::NONE;
    if (!power_down_imu()) {
        wake_error = rick::ImuWakeError::POWER_DOWN;
        // Keep motion inhibited if a partial I2C transaction left the IMU asleep.
        sleep_state = CONTROL_STATE_SLEEP_FAULT;
        printf("IMU power-down failed; asleep with servo outputs disabled. Wake to retry initialization.\n");
    }
    sleep_battery_mv = read_battery_voltage_mv();
    std::fill(std::begin(imu_telemetry), std::end(imu_telemetry), 0);
    neopixelWrite(PIN_STATUS_LED, 0, 0, 0);
    configure_bluetooth_standby(true);
    printf("Sleeping: servo PWM off; encrypted Wake required.\n");
    // USB is deliberately unavailable in standby; BOOT + reset still recovers it.
    Serial.end();
    control_state = sleep_state;
    ESP_ERROR_CHECK(esp_pm_lock_release(awake_cpu_lock));
    ESP_ERROR_CHECK(esp_pm_lock_release(awake_sleep_lock));
}

void wake_robot() {
    ESP_ERROR_CHECK(esp_pm_lock_acquire(awake_sleep_lock));
    ESP_ERROR_CHECK(esp_pm_lock_acquire(awake_cpu_lock));
    Serial.begin(115200);
    control_state = CONTROL_STATE_WAKING;
    wake_error = rick::ImuWakeError::NONE;
    // A dedicated resume restores the bus, follows the vendor sequence, verifies
    // configuration, and waits for usable samples without issuing a cold reset.
    // Servo pulses remain disabled throughout, including on every failure path.
    float attitude[4];
    wake_error = rick::resume_imu(imu_wake_io, attitude);
    if (wake_error != rick::ImuWakeError::NONE) {
        printf("Wake failed at step %u; remaining asleep with PWM off.\n", unsigned(wake_error));
        power_down_imu();
        Serial.end();
        control_state = CONTROL_STATE_SLEEP_FAULT;
        ESP_ERROR_CHECK(esp_pm_lock_release(awake_cpu_lock));
        ESP_ERROR_CHECK(esp_pm_lock_release(awake_sleep_lock));
        return;
    }
    imu_raw_capture = rick::ImuRawCapture{};
    publish_imu_raw();
    imu_clock.reset();
    gyro_calibration.reset_window();
    imu_consecutive_read_errors = 0;
    q0 = attitude[0]; q1 = attitude[1]; q2 = attitude[2]; q3 = attitude[3];
    std::fill(std::begin(target_actions), std::end(target_actions), 0.0f);
    reset_policy_history_and_clock(0);
    motion_ready_at = time_us_64() + 1000000;
    stop_requested = false;
    // Discard motion commands received during the wake transition.
    drain_queue<uint8_t>(control_command_queue);
    configure_bluetooth_standby(false);
    control_state = CONTROL_STATE_RELAXED;
    robot_sleeping = false;
    neopixelWrite(PIN_STATUS_LED, 16, 16, 16);
    printf("Awake; PWM remains off until a pose or Start command.\n");
}

void process_control_commands(){
    if(stop_requested){stop_requested=false;
        rick::MotionCommand discarded; xQueueReceive(motion_command_queue, &discarded, 0);drain_queue<uint8_t>(control_command_queue);
        drain_queue<ImuRawCommand>(imu_raw_command_queue);
        drain_queue<ServoCalibrationCommand>(servo_calibration_command_queue);
        apply_control_command(ControlCommand::STOP);return;}
    rick::MotionCommand motion;
    if (xQueueReceive(motion_command_queue, &motion, 0) == pdTRUE &&
        !robot_sleeping && !imu_raw_locks_robot()) set_target_motion(motion);
    uint8_t value;while(xQueueReceive(control_command_queue,&value,0)==pdTRUE)
        apply_control_command(static_cast<ControlCommand>(value));
}
void process_servo_calibration_commands(){if(robot_sleeping)return;ServoCalibrationCommand command;
    while(xQueueReceive(servo_calibration_command_queue,&command,0)==pdTRUE)
        apply_servo_calibration_command(command);}
void process_imu_raw_commands() {
    if (robot_sleeping) return;
    ImuRawCommand command;
    while (xQueueReceive(imu_raw_command_queue, &command, 0) == pdTRUE) {
        if (command.pose == 0) {
            if (command.token == imu_raw_capture.token) cancel_imu_raw();
        } else if (!imu_raw_locks_robot()) {
            finish_gait_diagnostics(rick::GaitDiagnostics::RAW_TEST);
            robot_running = false;
            servo_calibration_active = false;
            control_state = CONTROL_STATE_STOPPED;
            drain_queue<uint8_t>(control_command_queue);
            drain_queue<ServoCalibrationCommand>(servo_calibration_command_queue);
            imu_clock.reset();
            gyro_calibration.reset_window();
            imu_raw_capture.start(imu_raw_io, command.token, command.pose, time_us_64());
            publish_imu_raw();
        }
    }
}
void update_imu_and_clock_observation(float gx,float gy,float gz){
    get_local_gravity(&current_obs[GRAVITY_OFFSET]);
    current_obs[GYRO_OFFSET]=POLICY_GYRO_OBS_SCALE*gx;
    current_obs[GYRO_OFFSET+1]=POLICY_GYRO_OBS_SCALE*gy;
    current_obs[GYRO_OFFSET+2]=POLICY_GYRO_OBS_SCALE*gz;
    current_obs[CLOCK_OFFSET]=std::sin(gait_phase);current_obs[CLOCK_OFFSET+1]=std::cos(gait_phase);
    current_obs[TARGET_VELOCITY_OFFSET]=POLICY_MOTION_COMMAND_VERSION ? motion_command.forward : POLICY_TARGET_VELOCITY;
    if constexpr(POLICY_MOTION_COMMAND_VERSION) current_obs[TARGET_YAW_OFFSET]=motion_command.yaw;
    if constexpr(POLICY_HEADING_OBSERVATION)rick::relative_heading_observation(
        rick::heading_from_quat(q0,q1,q2,q3),heading_reference,&current_obs[HEADING_OFFSET]);
    for(float &value:current_obs)value=std::max(-10.0f,std::min(10.0f,value));
}
void append_command_history(const float *action){std::memmove(current_obs,current_obs+ACTION_DIM,
    (HISTORY_DIM-ACTION_DIM)*sizeof(float));std::memcpy(current_obs+HISTORY_DIM-ACTION_DIM,
    action,ACTION_DIM*sizeof(float));}
void report_control_timing(uint64_t now){const uint64_t elapsed=now-control_timing.window_started_us;
    if(elapsed<1000000||control_timing.samples==0)return;if(Serial){const uint64_t rate=control_timing.samples*100000000/elapsed;
    printf("Timing: hz=%llu.%02llu infer_us(avg/max)=%llu/%llu step_us(max)=%llu dt_us(min/max)=%llu/%llu missed=%llu overruns=%llu\n",
        rate/100,rate%100,control_timing.inference_total_us/control_timing.samples,
        control_timing.inference_max_us,control_timing.step_max_us,
        control_timing.interval_samples?control_timing.interval_min_us:0,
        control_timing.interval_max_us,control_timing.missed_ticks,control_timing.overruns);}
    control_timing.reset(now,true);}

uint32_t read_u32(const uint8_t *b){return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24);}
void write_u32(uint8_t *b,uint32_t v){b[0]=v;b[1]=v>>8;b[2]=v>>16;b[3]=v>>24;}
bool ota_locks_robot(){return ota_state==OtaState::PREPARING||ota_state==OtaState::RECEIVING||
    ota_state==OtaState::READY||ota_state==OtaState::REBOOTING;}
void ota_cleanup_hash(){if(ota_sha_active){mbedtls_sha256_free(&ota_sha_state);ota_sha_active=false;}}
void ota_abort_writer(){if(ota_writer_active){esp_ota_abort(ota_handle);ota_writer_active=false;ota_handle=0;}}
void ota_fail(OtaError error){ota_abort_writer();ota_cleanup_hash();drain_queue<OtaDataPacket>(ota_data_queue);
    ota_error=error;ota_state=OtaState::ERROR;printf("Firmware update stopped with error %u after %lu bytes.\n",
    unsigned(error),static_cast<unsigned long>(ota_bytes_received));}
void ota_reset(){ota_abort_writer();ota_cleanup_hash();drain_queue<OtaDataPacket>(ota_data_queue);
    ota_digest_bytes_received=ota_total_bytes=ota_bytes_received=ota_chunks_received=0;
    ota_target_partition=0xff;ota_partition=nullptr;ota_error=OtaError::NONE;ota_state=OtaState::IDLE;}
void stop_robot_for_update(){stop_requested=false;robot_running=false;servo_calibration_active=false;
    finish_gait_diagnostics(rick::GaitDiagnostics::OTA);
    cancel_imu_raw();drain_queue<ImuRawCommand>(imu_raw_command_queue);
    control_schedule.reset(time_us_64());control_state=CONTROL_STATE_STOPPED;
    drain_queue<uint8_t>(control_command_queue);drain_queue<ServoCalibrationCommand>(servo_calibration_command_queue);}
void begin_ota_update(const OtaControlPacket &packet){
    if(packet.length!=OTA_BEGIN_PACKET_SIZE||(ota_state!=OtaState::IDLE&&ota_state!=OtaState::ERROR)){
        ota_fail(OtaError::INVALID_REQUEST);return;}
    const uint32_t size=read_u32(&packet.bytes[1]),family=read_u32(&packet.bytes[5]);
    ota_partition=esp_ota_get_next_update_partition(nullptr);
    if(family!=ESP32_S3_FAMILY_ID||size<1024){ota_fail(OtaError::INVALID_IMAGE);return;}
    if(!ota_partition){ota_fail(OtaError::NO_PARTITION);return;}
    if(size>ota_partition->size){ota_fail(OtaError::OUT_OF_RANGE);return;}
    ota_state=OtaState::PREPARING;stop_robot_for_update();drain_queue<OtaDataPacket>(ota_data_queue);
    ota_total_bytes=size;ota_bytes_received=ota_chunks_received=ota_digest_bytes_received=0;
    std::memset(ota_expected_sha256,0,sizeof(ota_expected_sha256));
    ota_target_partition=uint8_t(ota_partition->subtype-ESP_PARTITION_SUBTYPE_APP_OTA_MIN);
    mbedtls_sha256_init(&ota_sha_state);
    if(mbedtls_sha256_starts_ret(&ota_sha_state,0)!=0){ota_fail(OtaError::HASH_UNAVAILABLE);return;}
    ota_sha_active=true;const esp_err_t result=esp_ota_begin(ota_partition,size,&ota_handle);
    if(result!=ESP_OK){printf("esp_ota_begin failed: %s\n",esp_err_to_name(result));
        ota_fail(OtaError::FLASH_OPERATION);return;}
    ota_writer_active=true;
    printf("Receiving %lu-byte ESP32 image into OTA slot %u.\n",static_cast<unsigned long>(size),ota_target_partition);
}
void receive_ota_digest(const OtaControlPacket &packet){
    if(packet.length<=2||ota_state!=OtaState::PREPARING){ota_fail(OtaError::INVALID_REQUEST);return;}
    const uint8_t offset=packet.bytes[1];const size_t length=packet.length-2;
    if(offset!=ota_digest_bytes_received||offset+length>sizeof(ota_expected_sha256)){
        ota_fail(OtaError::OUT_OF_ORDER);return;}
    std::memcpy(&ota_expected_sha256[offset],&packet.bytes[2],length);ota_digest_bytes_received+=length;
    if(ota_digest_bytes_received==sizeof(ota_expected_sha256))ota_state=OtaState::RECEIVING;
}
void finish_ota_receive(){uint8_t digest[OTA_SHA256_SIZE];
    if(!ota_sha_active||mbedtls_sha256_finish_ret(&ota_sha_state,digest)!=0){ota_fail(OtaError::HASH_UNAVAILABLE);return;}
    ota_cleanup_hash();if(std::memcmp(digest,ota_expected_sha256,sizeof(digest))!=0){ota_fail(OtaError::HASH_MISMATCH);return;}
    const esp_err_t result=esp_ota_end(ota_handle);ota_writer_active=false;ota_handle=0;
    if(result!=ESP_OK){printf("esp_ota_end rejected image: %s\n",esp_err_to_name(result));
        ota_fail(OtaError::IMAGE_REJECTED);return;}
    ota_state=OtaState::READY;printf("Firmware image verified. Waiting for commit.\n");}
void process_ota_data_packet(const OtaDataPacket &packet){
    if(ota_state!=OtaState::RECEIVING||packet.length<=4){ota_fail(OtaError::INVALID_REQUEST);return;}
    const uint32_t offset=read_u32(packet.bytes);const size_t length=packet.length-4;
    if(offset!=ota_bytes_received){ota_fail(OtaError::OUT_OF_ORDER);return;}
    if(length>ota_total_bytes-ota_bytes_received){ota_fail(OtaError::OUT_OF_RANGE);return;}
    const uint8_t *data=&packet.bytes[4];if(offset==0&&data[0]!=0xe9){ota_fail(OtaError::INVALID_IMAGE);return;}
    if(esp_ota_write(ota_handle,data,length)!=ESP_OK){ota_fail(OtaError::FLASH_OPERATION);return;}
    if(mbedtls_sha256_update_ret(&ota_sha_state,data,length)!=0){ota_fail(OtaError::HASH_UNAVAILABLE);return;}
    ota_bytes_received+=length;++ota_chunks_received;if(ota_bytes_received==ota_total_bytes)finish_ota_receive();}
void process_ota_control_packets(){OtaControlPacket packet;while(xQueueReceive(ota_control_queue,&packet,0)==pdTRUE){
    if(!packet.length)continue;switch(packet.bytes[0]){
    case OTA_CONTROL_BEGIN:begin_ota_update(packet);break;
    case OTA_CONTROL_ABORT:ota_reset();printf("Firmware update aborted.\n");break;
    case OTA_CONTROL_COMMIT:{if(packet.length!=1||ota_state!=OtaState::READY||!ota_partition){ota_fail(OtaError::INVALID_REQUEST);break;}
        const esp_err_t result=esp_ota_set_boot_partition(ota_partition);if(result!=ESP_OK){
            printf("Could not select OTA boot partition: %s\n",esp_err_to_name(result));ota_fail(OtaError::REBOOT_FAILED);break;}
        ota_state=OtaState::REBOOTING;printf("Rebooting into verified firmware.\n");delay(150);esp_restart();break;}
    case OTA_CONTROL_DIGEST:receive_ota_digest(packet);break;
    default:ota_fail(OtaError::INVALID_REQUEST);break;}}}
void process_ota_data_packets(){OtaDataPacket packet;for(int i=0;i<4&&xQueueReceive(ota_data_queue,&packet,0)==pdTRUE;++i)
    process_ota_data_packet(packet);}
void process_ota_abort_request(){if(!ota_abort_requested)return;ota_abort_requested=false;
    if(ota_state==OtaState::PREPARING||ota_state==OtaState::RECEIVING)ota_fail(OtaError::DISCONNECTED);}

void set_value(BLECharacteristic *characteristic,const uint8_t *data,size_t length){
    characteristic->setValue(const_cast<uint8_t*>(data),length);}
class RickCharacteristicCallbacks:public BLECharacteristicCallbacks{
public:enum class Kind{CONTROL,IMU,SERVO,OTA_CONTROL,OTA_DATA,OTA_STATUS,BATTERY,IMU_RAW,GAIT_DIAGNOSTICS};
    explicit RickCharacteristicCallbacks(Kind kind):kind_(kind){}
    void onRead(BLECharacteristic *c)override{
        if(kind_==Kind::GAIT_DIAGNOSTICS){uint8_t b[rick::GaitDiagnostics::FRAME_SIZE];
            portENTER_CRITICAL(&gait_diagnostics_mutex);
            const unsigned size=gait_diagnostics.encode(b);
            portEXIT_CRITICAL(&gait_diagnostics_mutex);set_value(c,b,size);return;}
        if(kind_==Kind::CONTROL){
            uint8_t state[16]={control_state,1,uint8_t(wake_error),POLICY_MOTION_COMMAND_VERSION};
            const int16_t limits[4] = {int16_t(std::lround(POLICY_FORWARD_MIN*1000)),
                int16_t(std::lround(POLICY_FORWARD_MAX*1000)),int16_t(std::lround(POLICY_YAW_MIN*1000)),
                int16_t(std::lround(POLICY_YAW_MAX*1000))};
            for(int i=0;i<4;++i){state[4+2*i]=uint16_t(limits[i]);state[5+2*i]=uint16_t(limits[i])>>8;}
            portENTER_CRITICAL(&motion_mutex);
            for(int i=0;i<2;++i){state[12+2*i]=uint16_t(requested_motion_units[i]);state[13+2*i]=uint16_t(requested_motion_units[i])>>8;}
            portEXIT_CRITICAL(&motion_mutex);set_value(c,state,sizeof(state));}
        else if(kind_==Kind::IMU_RAW){uint8_t b[rick::ImuRawCapture::PACKET_SIZE];
            portENTER_CRITICAL(&imu_raw_mutex);std::memcpy(b,imu_raw_packet,sizeof(b));portEXIT_CRITICAL(&imu_raw_mutex);
            set_value(c,b,sizeof(b));}
        else if(kind_==Kind::IMU){uint8_t b[12];for(int i=0;i<6;++i){const uint16_t v=imu_telemetry[i];b[2*i]=v;b[2*i+1]=v>>8;}set_value(c,b,sizeof(b));}
        else if(kind_==Kind::BATTERY){const uint16_t v=robot_sleeping ? sleep_battery_mv : read_battery_voltage_mv();const uint8_t b[2]={uint8_t(v),uint8_t(v>>8)};set_value(c,b,sizeof(b));}
        else if(kind_==Kind::SERVO){uint8_t b[ACTION_DIM*2];for(int i=0;i<ACTION_DIM;++i){const uint16_t v=std::lround(SERVO_CENTERS_US[i]);b[2*i]=v;b[2*i+1]=v>>8;}set_value(c,b,sizeof(b));}
        else if(kind_==Kind::OTA_STATUS){uint8_t b[32]={OTA_PROTOCOL_VERSION,uint8_t(ota_state),uint8_t(ota_error),ota_target_partition};
            write_u32(&b[4],ota_bytes_received);write_u32(&b[8],ota_total_bytes);write_u32(&b[12],ota_chunks_received);
            const esp_partition_t *running=esp_ota_get_running_partition();b[16]=running&&running->subtype>=ESP_PARTITION_SUBTYPE_APP_OTA_MIN?
                uint8_t(running->subtype-ESP_PARTITION_SUBTYPE_APP_OTA_MIN):0xff;b[19]=1;write_u32(&b[28],ESP32_S3_FAMILY_ID);set_value(c,b,sizeof(b));}}
    void onWrite(BLECharacteristic *c)override{const std::string value=c->getValue();
        const auto *data=reinterpret_cast<const uint8_t*>(value.data());const size_t size=value.size();
        if(kind_==Kind::GAIT_DIAGNOSTICS){
            portENTER_CRITICAL(&gait_diagnostics_mutex);
            if(size==1&&data[0]==0)gait_diagnostics.select_header();
            else if(size==7&&data[0]==1)gait_diagnostics.select_frame(read_u32(data+1),
                uint16_t(data[5])|(uint16_t(data[6])<<8));
            portEXIT_CRITICAL(&gait_diagnostics_mutex);return;}
        if(control_state==CONTROL_STATE_WAKING)return;
        if(robot_sleeping && (kind_!=Kind::CONTROL || size!=1 || data[0]!=uint8_t(ControlCommand::WAKE)))return;
        if(kind_==Kind::IMU_RAW){if(ota_locks_robot()||size!=4||data[0]!=1||data[3]>6)return;
            const ImuRawCommand command={uint16_t(uint16_t(data[1])|(uint16_t(data[2])<<8)),data[3]};
            xQueueSend(imu_raw_command_queue,&command,0);return;}
        if(kind_==Kind::CONTROL){
            if(POLICY_MOTION_COMMAND_VERSION && !ota_locks_robot() && !imu_raw_locks_robot()) {
                rick::MotionCommand motion;
                if(rick::decode_motion_command(data,size,motion_minimum,motion_maximum,motion)){
                    xQueueOverwrite(motion_command_queue,&motion);return;}
            }
            if(ota_locks_robot()||size!=1||data[0]>uint8_t(ControlCommand::WAKE))return;
            if(data[0]==uint8_t(ControlCommand::STOP))stop_requested=true;else xQueueSend(control_command_queue,data,0);}
        else if(kind_==Kind::SERVO){if(ota_locks_robot()||size!=3)return;const int16_t delta=int16_t(uint16_t(data[1])|(uint16_t(data[2])<<8));
            const bool finish=data[0]==SERVO_CALIBRATION_FINISH;if((finish&&delta)||( !finish&&(data[0]>=ACTION_DIM||!delta||delta< -SERVO_CALIBRATION_MAX_STEP_US||delta>SERVO_CALIBRATION_MAX_STEP_US)))return;
            const ServoCalibrationCommand command={data[0],delta};xQueueSend(servo_calibration_command_queue,&command,0);}
        else if(kind_==Kind::OTA_CONTROL){if(!size||size>OTA_MAX_CONTROL_PACKET_SIZE)return;const bool valid=
            (data[0]==OTA_CONTROL_BEGIN&&size==OTA_BEGIN_PACKET_SIZE)||(data[0]==OTA_CONTROL_DIGEST&&size>2)||
            ((data[0]==OTA_CONTROL_ABORT||data[0]==OTA_CONTROL_COMMIT)&&size==1);if(!valid)return;
            OtaControlPacket packet={};packet.length=size;std::memcpy(packet.bytes,data,size);xQueueSend(ota_control_queue,&packet,0);}
        else if(kind_==Kind::OTA_DATA){if(size<=4||size>OTA_MAX_DATA_VALUE_SIZE||ota_state!=OtaState::RECEIVING)return;
            OtaDataPacket packet={};packet.length=size;std::memcpy(packet.bytes,data,size);xQueueSend(ota_data_queue,&packet,0);}}
private:Kind kind_;};
void request_connection_interval(bool sleeping) {
    if (!ble_connected) return;
    esp_ble_conn_update_params_t params = {};
    std::memcpy(params.bda, ble_peer, sizeof(params.bda));
    // Units: 1.25 ms intervals, 10 ms timeout. The phone may negotiate these.
    params.min_int = sleeping ? 160 : 12;
    params.max_int = sleeping ? 320 : 24;
    params.latency = sleeping ? 4 : 0;
    params.timeout = 600;
    esp_ble_gap_update_conn_params(&params);
}
void configure_bluetooth_standby(bool sleeping) {
    auto *advertising = BLEDevice::getAdvertising();
    if (!ble_connected) advertising->stop();
    // Advertising units are 0.625 ms: standby 1–2 seconds, awake 20–40 ms.
    advertising->setMinInterval(sleeping ? 1600 : 32);
    advertising->setMaxInterval(sleeping ? 3200 : 64);
    advertising->setMinPreferred(sleeping ? 160 : 12);
    advertising->setMaxPreferred(sleeping ? 320 : 24);
    if (!ble_connected) advertising->start();
    request_connection_interval(sleeping);
}
class RickServerCallbacks:public BLEServerCallbacks {
    void onConnect(BLEServer*, esp_ble_gatts_cb_param_t *param) override {
        std::memcpy(ble_peer, param->connect.remote_bda, sizeof(ble_peer));
        ble_connected = true;
        request_connection_interval(robot_sleeping);
    }
    void onDisconnect(BLEServer*) override {
        ble_connected = false;
        if (!robot_sleeping) stop_requested = true;
        ota_abort_requested = true;
        BLEDevice::startAdvertising();
    }
};
BLECharacteristic *create_characteristic(BLEService *service,const char *uuid,uint32_t properties,
    esp_gatt_perm_t permissions,RickCharacteristicCallbacks::Kind kind){auto *c=service->createCharacteristic(uuid,properties);
    c->setAccessPermissions(permissions);c->setCallbacks(new RickCharacteristicCallbacks(kind));return c;}
bool init_bluetooth(){BLEDevice::init("RickBot");BLEDevice::setMTU(517);auto *server=BLEDevice::createServer();
    ESP_ERROR_CHECK(esp_bt_sleep_enable());
    server->setCallbacks(new RickServerCallbacks());auto *service=server->createService(BLEUUID(SERVICE_UUID),36);
    create_characteristic(service,CONTROL_UUID,BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_WRITE,
        esp_gatt_perm_t(ESP_GATT_PERM_READ|ESP_GATT_PERM_WRITE_ENCRYPTED),RickCharacteristicCallbacks::Kind::CONTROL);
    create_characteristic(service,IMU_UUID,BLECharacteristic::PROPERTY_READ,ESP_GATT_PERM_READ,RickCharacteristicCallbacks::Kind::IMU);
    create_characteristic(service,SERVO_UUID,BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_WRITE,
        esp_gatt_perm_t(ESP_GATT_PERM_READ|ESP_GATT_PERM_WRITE_ENCRYPTED),RickCharacteristicCallbacks::Kind::SERVO);
    create_characteristic(service,OTA_CONTROL_UUID,BLECharacteristic::PROPERTY_WRITE,ESP_GATT_PERM_WRITE_ENCRYPTED,RickCharacteristicCallbacks::Kind::OTA_CONTROL);
    create_characteristic(service,OTA_DATA_UUID,BLECharacteristic::PROPERTY_WRITE,ESP_GATT_PERM_WRITE_ENCRYPTED,RickCharacteristicCallbacks::Kind::OTA_DATA);
    create_characteristic(service,OTA_STATUS_UUID,BLECharacteristic::PROPERTY_READ,ESP_GATT_PERM_READ,RickCharacteristicCallbacks::Kind::OTA_STATUS);
    create_characteristic(service,BATTERY_UUID,BLECharacteristic::PROPERTY_READ,ESP_GATT_PERM_READ,RickCharacteristicCallbacks::Kind::BATTERY);
    create_characteristic(service,IMU_RAW_UUID,BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_WRITE,
        esp_gatt_perm_t(ESP_GATT_PERM_READ|ESP_GATT_PERM_WRITE_ENCRYPTED),RickCharacteristicCallbacks::Kind::IMU_RAW);
    create_characteristic(service,GAIT_DIAGNOSTICS_UUID,BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_WRITE,
        esp_gatt_perm_t(ESP_GATT_PERM_READ_ENCRYPTED|ESP_GATT_PERM_WRITE_ENCRYPTED),RickCharacteristicCallbacks::Kind::GAIT_DIAGNOSTICS);
    service->start();auto *security=new BLESecurity();security->setAuthenticationMode(ESP_LE_AUTH_REQ_SC_BOND);
    security->setCapability(ESP_IO_CAP_NONE);security->setKeySize(16);
    security->setInitEncryptionKey(esp_ble_key_type_t(ESP_BLE_ENC_KEY_MASK|ESP_BLE_ID_KEY_MASK));
    auto *advertising=BLEDevice::getAdvertising();advertising->addServiceUUID(SERVICE_UUID);advertising->setScanResponse(true);
    advertising->setMinPreferred(0x0c);advertising->setMaxPreferred(0x0c);BLEDevice::startAdvertising();
    printf("Bluetooth ready as RickBot. Waiting for Android.\n");return true;}

#if defined(RICK_IMU_RAW_TEST) && RICK_IMU_RAW_TEST
#include "imu_raw_diagnostic.h"
#else
void setup(){neopixelWrite(PIN_STATUS_LED, 16, 16, 16);Serial.begin(115200);delay(250);printf("\nStarting Rick ESP32-S3 policy checkpoint %u (%d obs, %d actions)...\n",
    unsigned(POLICY_CHECKPOINT_STEP),OBS_DIM,ACTION_DIM);printf("Controller: ESP32-S3 LX7 at %u MHz, target period=%lu us.\n",
    getCpuFrequencyMhz(),static_cast<unsigned long>(CONTROL_PERIOD_US));
    init_power_management();
    init_battery_monitor();
    motion_command_queue=xQueueCreate(1,sizeof(rick::MotionCommand));
    control_command_queue=xQueueCreate(8,sizeof(uint8_t));servo_calibration_command_queue=xQueueCreate(8,sizeof(ServoCalibrationCommand));
    imu_raw_command_queue=xQueueCreate(4,sizeof(ImuRawCommand));
    ota_control_queue=xQueueCreate(4,sizeof(OtaControlPacket));ota_data_queue=xQueueCreate(8,sizeof(OtaDataPacket));
    if(!motion_command_queue||!control_command_queue||!servo_calibration_command_queue||!imu_raw_command_queue||!ota_control_queue||!ota_data_queue){printf("Could not allocate command queues.\n");abort();}
    if(!init_imu()){printf("Tokmas ICM-42688-P initialization failed; servos remain disabled.\n");
        while(true)delay(1000);}
    init_servos();reset_servos(0);control_state=CONTROL_STATE_DEFAULT;
    printf("Servos at calibrated default pose; policy is stopped.\n");init_bluetooth();control_schedule.reset(time_us_64());}
void loop(){
    if (robot_sleeping) {
        uint8_t command;
        // Block indefinitely: BLE/RTOS schedules radio events; no busy polling.
        if (xQueueReceive(control_command_queue, &command, portMAX_DELAY) == pdTRUE)
            apply_control_command(static_cast<ControlCommand>(command));
        return;
    }
    process_ota_abort_request();process_ota_control_packets();process_ota_data_packets();
    if(!ota_locks_robot()){process_control_commands();process_imu_raw_commands();process_servo_calibration_commands();}
    if(robot_sleeping)return;
    if(imu_raw_locks_robot()){
        imu_raw_capture.tick(imu_raw_io,time_us_64());publish_imu_raw();
        control_schedule.reset(time_us_64());imu_clock.reset();delay(1);return;}
    const uint64_t now=time_us_64();if(ota_locks_robot()){control_schedule.reset(now);imu_clock.reset();}
    else {
        uint64_t skipped = 0;
        if (control_schedule.take(now, skipped)) {
            rick::GaitDiagnostics::Frame frame;
            frame.skipped = uint32_t(skipped);
            float gx, gy, gz, ax, ay, az;
            if (!read_imu(&gx, &gy, &gz, &ax, &ay, &az, frame.raw)) {
                ++imu_consecutive_read_errors;
                if ((++imu_read_errors & 0x3f) == 1)
                    printf("Tokmas IMU I2C read failed (%lu errors).\n", static_cast<unsigned long>(imu_read_errors));
                if (imu_consecutive_read_errors >= 3) stop_requested = true;
                // Explicit failed sample, not a duplicate of the preceding IMU.
                frame.time_us = time_us_64(); frame.imu_dt = NAN;
                for (int i = 0; i < 3; ++i)
                    frame.accel[i] = frame.gyro_before[i] = frame.gyro[i] = NAN;
                frame.control_us = uint32_t(frame.time_us - now);
                record_gait_diagnostics(frame);
                imu_clock.reset(); delay(0); return;
            }
            imu_consecutive_read_errors = 0;
            const uint64_t sample = time_us_64();
            const float imu_dt = imu_clock.step(sample, POLICY_CONTROL_DT);
            frame.time_us = sample; frame.imu_dt = imu_dt;
            frame.flags = rick::GaitDiagnostics::READ_OK;
            for (int16_t value : frame.raw)
                if (value <= -32760 || value >= 32760) frame.flags |= rick::GaitDiagnostics::RAW_CLIPPED;
            const float gyro[3] = {gx, gy, gz}, accel[3] = {ax, ay, az};
            std::memcpy(frame.accel, accel, sizeof(accel));
            std::memcpy(frame.gyro_before, gyro, sizeof(gyro));
            if constexpr (POLICY_GYRO_BIAS_CALIBRATION) {
                if (skipped || imu_dt > 1.5f * POLICY_CONTROL_DT) gyro_calibration.reset_window();
                if (gyro_calibration.observe(gyro, accel, !robot_running && !servo_calibration_active))
                    printf("Gyro bias calibrated: %.5f %.5f %.5f rad/s.\n",
                        gyro_calibration.bias(0), gyro_calibration.bias(1), gyro_calibration.bias(2));
                gx -= gyro_calibration.bias(0); gy -= gyro_calibration.bias(1); gz -= gyro_calibration.bias(2);
            }
            frame.gyro[0] = gx; frame.gyro[1] = gy; frame.gyro[2] = gz;
            madgwick_update_6dof(gx, gy, gz, ax, ay, az, imu_dt);
            if (robot_running) {
                if constexpr(POLICY_MOTION_COMMAND_VERSION)
                    rick::step_motion_command(motion_command, target_motion_command,
                        motion_acceleration, imu_dt, heading_reference);
                gait_phase = gait_clock.sample(sample, POLICY_STEP_FREQUENCY);
                update_imu_and_clock_observation(gx, gy, gz);
                const uint64_t started = time_us_64();
                policy_network.infer(current_obs, target_actions);
                frame.inference_us = uint32_t(time_us_64() - started);
                update_servos(target_actions); append_command_history(target_actions);
                frame.step = step_counter++;
                frame.flags |= rick::GaitDiagnostics::POLICY_APPLIED;
            }
            frame.control_us = uint32_t(time_us_64() - now);
            record_gait_diagnostics(frame);
            if (robot_running) {
                const uint64_t finished = time_us_64();
                control_timing.record(now, finished, frame.inference_us, skipped, control_schedule.next_us());
                report_control_timing(finished);
            }
        }
    }
    delay(0);}
#endif
