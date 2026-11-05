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
#include "esp_partition.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "mbedtls/sha256.h"

#include "control_timing.h"
#include "heading_control.h"
#include "policy_inference.h"

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

constexpr int GRAVITY_OFFSET = HISTORY_DIM;
constexpr int GYRO_OFFSET = GRAVITY_OFFSET + 3;
constexpr int CLOCK_OFFSET = GYRO_OFFSET + 3;
constexpr int TARGET_VELOCITY_OFFSET = CLOCK_OFFSET + 2;
constexpr int HEADING_OFFSET = TARGET_VELOCITY_OFFSET + 1;
constexpr float PI_F = 3.14159265358979323846f;

static_assert(ACTION_DIM == 8, "Rick v2 firmware expects eight servos");
static_assert(HEADING_OFFSET + 2 * POLICY_HEADING_OBSERVATION == OBS_DIM,
              "Firmware observation layout does not match the exported policy");
static_assert(POLICY_OUTPUT_DIM == 2 * ACTION_DIM,
              "Expected tanh-normal mean and scale outputs");

// ESP32-S3-Zero carrier-board wiring.
constexpr int PIN_STATUS_LED = 21;
constexpr int PIN_IMU_SDA = 2;
constexpr int PIN_IMU_SCL = 4;
constexpr int PIN_IMU_INT1 = 5;

// MuJoCo actuator order: left roll/upper/lower/foot, then right equivalents.
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
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
float SERVO_CENTERS_US[ACTION_DIM] = {
    1595.0f, 1565.0f, 1700.0f, 1500.0f,
    1455.0f, 1545.0f, 1590.0f, 1560.0f,
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
constexpr float ACCEL_SCALE_G = 4.0f / 32768.0f;

float current_obs[OBS_DIM] = {0.0f};
float target_actions[ACTION_DIM] = {0.0f};
rick::PolicyNetwork policy_network;

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
rick::GyroBiasCalibration gyro_calibration(POLICY_GYRO_BIAS_CALIBRATION_SAMPLES);

enum class ControlCommand : uint8_t {
    RESET_DEFAULT = 0, RESET_SERVO_CENTERS = 1, START = 2, STOP = 3,
};
struct ServoCalibrationCommand { uint8_t servo_index; int16_t delta_us; };
constexpr uint8_t SERVO_CALIBRATION_FINISH = 0xff;
constexpr int16_t SERVO_CALIBRATION_MAX_STEP_US = 100;
constexpr uint8_t CONTROL_STATE_DEFAULT = 0;
constexpr uint8_t CONTROL_STATE_SERVO_CENTERS = 1;
constexpr uint8_t CONTROL_STATE_RUNNING = 2;
constexpr uint8_t CONTROL_STATE_STOPPED = 3;
constexpr uint8_t CONTROL_STATE_CALIBRATING = 4;

QueueHandle_t control_command_queue = nullptr;
QueueHandle_t servo_calibration_command_queue = nullptr;
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

uint64_t time_us_64() { return static_cast<uint64_t>(esp_timer_get_time()); }

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

bool init_imu() {
    Wire.begin(PIN_IMU_SDA, PIN_IMU_SCL, 400000);
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

    uint8_t chip_id = 0;
    if (!read_imu_register(IMU_REG_CHIP_ID, chip_id) || chip_id != IMU_CHIP_ID) {
        printf("Tokmas IMU chip ID mismatch: expected 0x%02x, read 0x%02x.\n",
               IMU_CHIP_ID, chip_id);
        return false;
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
        return false;
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
    return configured;
}

int16_t clamp_to_int16(float value) {
    return static_cast<int16_t>(std::lround(
        std::max(-32768.0f, std::min(32767.0f, value))));
}

bool read_imu(float *gx, float *gy, float *gz, float *ax, float *ay, float *az) {
    uint8_t buffer[12];
    if (!read_imu_registers(IMU_REG_ACC_DATA_XL, buffer, sizeof(buffer))) return false;
    const int16_t raw_ax = static_cast<int16_t>(buffer[1]<<8 | buffer[0]);
    const int16_t raw_ay = static_cast<int16_t>(buffer[3]<<8 | buffer[2]);
    const int16_t raw_az = static_cast<int16_t>(buffer[5]<<8 | buffer[4]);
    const int16_t raw_gx = static_cast<int16_t>(buffer[7]<<8 | buffer[6]);
    const int16_t raw_gy = static_cast<int16_t>(buffer[9]<<8 | buffer[8]);
    const int16_t raw_gz = static_cast<int16_t>(buffer[11]<<8 | buffer[10]);
    const float pgx=raw_gx*GYRO_SCALE_RAD_S, pgy=raw_gy*GYRO_SCALE_RAD_S;
    const float pgz=raw_gz*GYRO_SCALE_RAD_S, pax=raw_ax*ACCEL_SCALE_G;
    const float pay=raw_ay*ACCEL_SCALE_G, paz=raw_az*ACCEL_SCALE_G;
    imu_telemetry[0]=clamp_to_int16(pax*1000); imu_telemetry[1]=clamp_to_int16(pay*1000);
    imu_telemetry[2]=clamp_to_int16(paz*1000);
    imu_telemetry[3]=clamp_to_int16(pgx*(1800/PI_F));
    imu_telemetry[4]=clamp_to_int16(pgy*(1800/PI_F));
    imu_telemetry[5]=clamp_to_int16(pgz*(1800/PI_F));
    // Preserve the previous sensor-to-robot transform. Confirm the assembled
    // Tokmas package orientation with Rick Control's IMU axis check.
    *gx=pgx; *gy=-pgz; *gz=-pgy; *ax=pax; *ay=-paz; *az=-pay;
    return true;
}

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
    for (int i=0;i<ACTION_DIM;++i) {
        const auto channel=static_cast<ledc_channel_t>(i);
        ledc_set_duty(LEDC_LOW_SPEED_MODE, channel,
            servo_duty(servo_pulse_width_us(i,actions[i],use_default_pose)));
        ledc_update_duty(LEDC_LOW_SPEED_MODE, channel);
    }
}

void reset_policy_history_and_clock(float previous_action) {
    gait_phase=0; gait_clock.reset(); const uint64_t now=time_us_64();
    control_timing.reset(now); step_counter=0; control_schedule.reset(now);
    std::fill(std::begin(current_obs),std::end(current_obs),0.0f);
    std::fill(current_obs,current_obs+HISTORY_DIM,previous_action);
    current_obs[CLOCK_OFFSET]=0; current_obs[CLOCK_OFFSET+1]=1;
    current_obs[TARGET_VELOCITY_OFFSET]=POLICY_TARGET_VELOCITY;
    heading_reference=rick::heading_from_quat(q0,q1,q2,q3);
    if constexpr(POLICY_HEADING_OBSERVATION)
        rick::relative_heading_observation(heading_reference,heading_reference,
                                            &current_obs[HEADING_OFFSET]);
}
void reset_servos(float action, bool use_default_pose=true) {
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
    servo_calibration_active=false;
    switch(command){
    case ControlCommand::RESET_DEFAULT: reset_servos(0);control_state=CONTROL_STATE_DEFAULT;
        printf("Bluetooth command: reset to calibrated default pose.\n");break;
    case ControlCommand::RESET_SERVO_CENTERS: reset_servos(0,false);
        control_state=CONTROL_STATE_SERVO_CENTERS;
        printf("Bluetooth command: reset every servo to its raw calibrated center.\n");break;
    case ControlCommand::START:
        if(POLICY_GYRO_BIAS_CALIBRATION&&!gyro_calibration.ready()){
            printf("Hold Rick still for gyro calibration before Start (%.1f seconds).\n",
                   POLICY_GYRO_BIAS_CALIBRATION_SAMPLES*POLICY_CONTROL_DT);break;}
        gyro_calibration.reset_window();reset_policy_history_and_clock(target_actions[0]);
        for(int h=0;h<HISTORY_LEN;++h)for(int a=0;a<ACTION_DIM;++a)
            current_obs[h*ACTION_DIM+a]=target_actions[a];
        robot_running=true;control_state=CONTROL_STATE_RUNNING;printf("Bluetooth command: start policy.\n");break;
    case ControlCommand::STOP: robot_running=false;control_schedule.reset(time_us_64());
        control_state=CONTROL_STATE_STOPPED;printf("Bluetooth command: stop policy and hold current pose.\n");break;
    }
}

template<typename T> void drain_queue(QueueHandle_t queue){T item;while(xQueueReceive(queue,&item,0)==pdTRUE){}}
void process_control_commands(){
    if(stop_requested){stop_requested=false;drain_queue<uint8_t>(control_command_queue);
        drain_queue<ServoCalibrationCommand>(servo_calibration_command_queue);
        apply_control_command(ControlCommand::STOP);return;}
    uint8_t value;while(xQueueReceive(control_command_queue,&value,0)==pdTRUE)
        apply_control_command(static_cast<ControlCommand>(value));
}
void process_servo_calibration_commands(){ServoCalibrationCommand command;
    while(xQueueReceive(servo_calibration_command_queue,&command,0)==pdTRUE)
        apply_servo_calibration_command(command);}
void update_imu_and_clock_observation(float gx,float gy,float gz){
    get_local_gravity(&current_obs[GRAVITY_OFFSET]);
    current_obs[GYRO_OFFSET]=POLICY_GYRO_OBS_SCALE*gx;
    current_obs[GYRO_OFFSET+1]=POLICY_GYRO_OBS_SCALE*gy;
    current_obs[GYRO_OFFSET+2]=POLICY_GYRO_OBS_SCALE*gz;
    current_obs[CLOCK_OFFSET]=std::sin(gait_phase);current_obs[CLOCK_OFFSET+1]=std::cos(gait_phase);
    current_obs[TARGET_VELOCITY_OFFSET]=POLICY_TARGET_VELOCITY;
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
public:enum class Kind{CONTROL,IMU,SERVO,OTA_CONTROL,OTA_DATA,OTA_STATUS};
    explicit RickCharacteristicCallbacks(Kind kind):kind_(kind){}
    void onRead(BLECharacteristic *c)override{
        if(kind_==Kind::CONTROL){const uint8_t state=control_state;set_value(c,&state,1);}
        else if(kind_==Kind::IMU){uint8_t b[12];for(int i=0;i<6;++i){const uint16_t v=imu_telemetry[i];b[2*i]=v;b[2*i+1]=v>>8;}set_value(c,b,sizeof(b));}
        else if(kind_==Kind::SERVO){uint8_t b[ACTION_DIM*2];for(int i=0;i<ACTION_DIM;++i){const uint16_t v=std::lround(SERVO_CENTERS_US[i]);b[2*i]=v;b[2*i+1]=v>>8;}set_value(c,b,sizeof(b));}
        else if(kind_==Kind::OTA_STATUS){uint8_t b[32]={OTA_PROTOCOL_VERSION,uint8_t(ota_state),uint8_t(ota_error),ota_target_partition};
            write_u32(&b[4],ota_bytes_received);write_u32(&b[8],ota_total_bytes);write_u32(&b[12],ota_chunks_received);
            const esp_partition_t *running=esp_ota_get_running_partition();b[16]=running&&running->subtype>=ESP_PARTITION_SUBTYPE_APP_OTA_MIN?
                uint8_t(running->subtype-ESP_PARTITION_SUBTYPE_APP_OTA_MIN):0xff;b[19]=1;write_u32(&b[28],ESP32_S3_FAMILY_ID);set_value(c,b,sizeof(b));}}
    void onWrite(BLECharacteristic *c)override{const std::string value=c->getValue();
        const auto *data=reinterpret_cast<const uint8_t*>(value.data());const size_t size=value.size();
        if(kind_==Kind::CONTROL){if(ota_locks_robot()||size!=1||data[0]>uint8_t(ControlCommand::STOP))return;
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
class RickServerCallbacks:public BLEServerCallbacks{void onConnect(BLEServer*)override{printf("Bluetooth connected.\n");}
    void onDisconnect(BLEServer*)override{stop_requested=true;ota_abort_requested=true;printf("Bluetooth disconnected; stopping policy.\n");BLEDevice::startAdvertising();}};
BLECharacteristic *create_characteristic(BLEService *service,const char *uuid,uint32_t properties,
    esp_gatt_perm_t permissions,RickCharacteristicCallbacks::Kind kind){auto *c=service->createCharacteristic(uuid,properties);
    c->setAccessPermissions(permissions);c->setCallbacks(new RickCharacteristicCallbacks(kind));return c;}
bool init_bluetooth(){BLEDevice::init("RickBot");BLEDevice::setMTU(517);auto *server=BLEDevice::createServer();
    server->setCallbacks(new RickServerCallbacks());auto *service=server->createService(SERVICE_UUID);
    create_characteristic(service,CONTROL_UUID,BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_WRITE,
        esp_gatt_perm_t(ESP_GATT_PERM_READ|ESP_GATT_PERM_WRITE_ENCRYPTED),RickCharacteristicCallbacks::Kind::CONTROL);
    create_characteristic(service,IMU_UUID,BLECharacteristic::PROPERTY_READ,ESP_GATT_PERM_READ,RickCharacteristicCallbacks::Kind::IMU);
    create_characteristic(service,SERVO_UUID,BLECharacteristic::PROPERTY_READ|BLECharacteristic::PROPERTY_WRITE,
        esp_gatt_perm_t(ESP_GATT_PERM_READ|ESP_GATT_PERM_WRITE_ENCRYPTED),RickCharacteristicCallbacks::Kind::SERVO);
    create_characteristic(service,OTA_CONTROL_UUID,BLECharacteristic::PROPERTY_WRITE,ESP_GATT_PERM_WRITE_ENCRYPTED,RickCharacteristicCallbacks::Kind::OTA_CONTROL);
    create_characteristic(service,OTA_DATA_UUID,BLECharacteristic::PROPERTY_WRITE,ESP_GATT_PERM_WRITE_ENCRYPTED,RickCharacteristicCallbacks::Kind::OTA_DATA);
    create_characteristic(service,OTA_STATUS_UUID,BLECharacteristic::PROPERTY_READ,ESP_GATT_PERM_READ,RickCharacteristicCallbacks::Kind::OTA_STATUS);
    service->start();auto *security=new BLESecurity();security->setAuthenticationMode(ESP_LE_AUTH_REQ_SC_BOND);
    security->setCapability(ESP_IO_CAP_NONE);security->setKeySize(16);
    security->setInitEncryptionKey(esp_ble_key_type_t(ESP_BLE_ENC_KEY_MASK|ESP_BLE_ID_KEY_MASK));
    auto *advertising=BLEDevice::getAdvertising();advertising->addServiceUUID(SERVICE_UUID);advertising->setScanResponse(true);
    advertising->setMinPreferred(0x0c);advertising->setMaxPreferred(0x0c);BLEDevice::startAdvertising();
    printf("Bluetooth ready as RickBot. Waiting for Android.\n");return true;}

void setup(){neopixelWrite(PIN_STATUS_LED, 16, 16, 16);Serial.begin(115200);delay(250);printf("\nStarting Rick ESP32-S3 policy checkpoint %u (%d obs, %d actions)...\n",
    unsigned(POLICY_CHECKPOINT_STEP),OBS_DIM,ACTION_DIM);printf("Controller: ESP32-S3 LX7 at %u MHz, target period=%lu us.\n",
    getCpuFrequencyMhz(),static_cast<unsigned long>(CONTROL_PERIOD_US));
    control_command_queue=xQueueCreate(8,sizeof(uint8_t));servo_calibration_command_queue=xQueueCreate(8,sizeof(ServoCalibrationCommand));
    ota_control_queue=xQueueCreate(4,sizeof(OtaControlPacket));ota_data_queue=xQueueCreate(8,sizeof(OtaDataPacket));
    if(!control_command_queue||!servo_calibration_command_queue||!ota_control_queue||!ota_data_queue){printf("Could not allocate command queues.\n");abort();}
    if(!init_imu()){printf("Tokmas ICM-42688-P initialization failed; servos remain disabled.\n");
        while(true)delay(1000);}
    init_servos();reset_servos(0);control_state=CONTROL_STATE_DEFAULT;
    printf("Servos at calibrated default pose; policy is stopped.\n");init_bluetooth();control_schedule.reset(time_us_64());}
void loop(){process_ota_abort_request();process_ota_control_packets();process_ota_data_packets();
    if(!ota_locks_robot()){process_control_commands();process_servo_calibration_commands();}
    const uint64_t now=time_us_64();if(ota_locks_robot()){control_schedule.reset(now);imu_clock.reset();}
    else{uint64_t skipped=0;if(control_schedule.take(now,skipped)){float gx,gy,gz,ax,ay,az;
        if(!read_imu(&gx,&gy,&gz,&ax,&ay,&az)){++imu_consecutive_read_errors;if((++imu_read_errors&0x3f)==1)printf("Tokmas IMU I2C read failed (%lu errors).\n",static_cast<unsigned long>(imu_read_errors));if(imu_consecutive_read_errors>=3)stop_requested=true;imu_clock.reset();delay(0);return;}imu_consecutive_read_errors=0;
        const uint64_t sample=time_us_64();const float imu_dt=imu_clock.step(sample,POLICY_CONTROL_DT);
        if constexpr(POLICY_GYRO_BIAS_CALIBRATION){const float gyro[3]={gx,gy,gz},accel[3]={ax,ay,az};
            if(skipped||imu_dt>1.5f*POLICY_CONTROL_DT)gyro_calibration.reset_window();
            if(gyro_calibration.observe(gyro,accel,!robot_running&&!servo_calibration_active))printf("Gyro bias calibrated: %.5f %.5f %.5f rad/s.\n",
                gyro_calibration.bias(0),gyro_calibration.bias(1),gyro_calibration.bias(2));
            gx-=gyro_calibration.bias(0);gy-=gyro_calibration.bias(1);gz-=gyro_calibration.bias(2);}
        madgwick_update_6dof(gx,gy,gz,ax,ay,az,imu_dt);if(robot_running){gait_phase=gait_clock.sample(sample,POLICY_STEP_FREQUENCY);
            update_imu_and_clock_observation(gx,gy,gz);const uint64_t started=time_us_64();policy_network.infer(current_obs,target_actions);
            const uint64_t inference=time_us_64()-started;update_servos(target_actions);append_command_history(target_actions);++step_counter;
            const uint64_t finished=time_us_64();control_timing.record(now,finished,inference,skipped,control_schedule.next_us());report_control_timing(finished);}}}
    delay(0);}
