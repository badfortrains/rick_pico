#include <stdio.h>
#include <cmath>
#include <algorithm>
#include "pico/stdlib.h"
#include "hardware/spi.h"
#include "hardware/timer.h"
#include "hardware/pwm.h"
#include "policy_weights.h"

// --- Configuration & Constants ---
#define OBS_DIM 245
#define HIDDEN_DIM 32
#define ACTION_DIM 6
#define HISTORY_LEN 40
#define HISTORY_DIM (HISTORY_LEN * ACTION_DIM) // 240

// SPI & LSM6DSO Pins (Adjust these to match your wiring!)
#define SPI_PORT spi1
#define PIN_MISO 8
#define PIN_CS 9
#define PIN_SCK 10
#define PIN_MOSI 11

// Servo Pins (Adjust to match where you plugged in your 6 servos)
const uint SERVO_PINS[ACTION_DIM] = {4, 3, 2, 7, 5, 6};

// IMU Registers & Scaling
#define CTRL1_XL 0x10
#define CTRL2_G 0x11
#define OUTX_L_G 0x22

// Assuming ±4g for Accel and ±2000 dps for Gyro
#define GYRO_SCALE_RAD_S (0.070f * (M_PI / 180.0f))
#define ACCEL_SCALE_G (0.122f / 1000.0f)

// --- Global Variables ---
float current_obs[OBS_DIM] = {0.0f}; // Holds history (0-239), gravity (240-242), clock (243-244)
float target_actions[ACTION_DIM] = {0.0f};

// Neural Net Buffers
float buffer_A[HIDDEN_DIM];
float buffer_B[HIDDEN_DIM];

// Madgwick Filter State
float q0 = 1.0f, q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;
float beta = 0.1f;

// Control Loop State
volatile int step_counter = 0;
volatile bool imu_data_ready = false;

// --- Servo Configuration ---

// 1.0f means normal, -1.0f means reversed.
// Right leg goes backward at 0 degrees (500us), forward at 180 degrees (2500us).
// Left leg goes forward at 0 degrees (500us), backward at 180 degrees (2500us).
const float SERVO_DIRS[ACTION_DIM] = {
    -1.0f, -1.0f, -1.0f, // Left Leg (Inverted)
    1.0f, 1.0f, 1.0f     // Right Leg
};

// 90 degrees corresponds perfectly to 1500us based on your script
float SERVO_CENTERS[ACTION_DIM] = {
    1500.0f, 1500.0f, 1500.0f,
    1500.0f, 1500.0f, 1500.0f};

// Since center is 1500us, and max is 2500us, the max swing is 1000us.
// A network output of 1.0 will add 1000us. An output of -1.0 will subtract 1000us.
const float SERVO_RANGE = 1000.0f;

volatile bool run_control_step = false;

// --- Neural Network Inference ---

inline float swish(float x)
{
    return x / (1.0f + std::exp(-x));
}

void normalize_obs(const float *raw_obs, float *norm_obs)
{
    for (int i = 0; i < OBS_DIM; ++i)
    {
        // Brax already computed the standard deviation (with epsilon) for us.
        // Subtract the mean, divide by the std. No slow square roots needed!
        norm_obs[i] = (raw_obs[i] - OBS_MEAN[i]) / OBS_STD[i];
    }
}

void dense_layer(const float *input, const float *weights, const float *biases,
                 float *output, int in_features, int out_features, bool apply_swish)
{
    for (int j = 0; j < out_features; ++j)
    {
        float sum = biases[j];
        for (int i = 0; i < in_features; ++i)
        {
            sum += input[i] * weights[i * out_features + j];
        }
        output[j] = apply_swish ? swish(sum) : sum;
    }
}

void infer_action(const float *raw_obs, float *final_action)
{
    // 0. Normalize observation into buffer_A
    normalize_obs(raw_obs, buffer_A);

    // 1. Layer 0 (Input -> Hidden)
    // Reads buffer_A, writes buffer_B
    dense_layer(buffer_A, PARAMS_HIDDEN_0_KERNEL, PARAMS_HIDDEN_0_BIAS, buffer_B, OBS_DIM, HIDDEN_DIM, true);

    // 2. Layer 1 (Hidden -> Hidden)
    // Reads buffer_B, writes buffer_A
    dense_layer(buffer_B, PARAMS_HIDDEN_1_KERNEL, PARAMS_HIDDEN_1_BIAS, buffer_A, HIDDEN_DIM, HIDDEN_DIM, true);

    // 3. Layer 2 (Hidden -> Hidden)
    // Reads buffer_A, writes buffer_B
    dense_layer(buffer_A, PARAMS_HIDDEN_2_KERNEL, PARAMS_HIDDEN_2_BIAS, buffer_B, HIDDEN_DIM, HIDDEN_DIM, true);

    // 4. Layer 3 (Hidden -> Hidden)
    // Reads buffer_B, writes buffer_A
    dense_layer(buffer_B, PARAMS_HIDDEN_3_KERNEL, PARAMS_HIDDEN_3_BIAS, buffer_A, HIDDEN_DIM, HIDDEN_DIM, true);

    // 5. Layer 4 / Output (Hidden -> Action * 2)
    // Reads buffer_A, writes buffer_B
    // Important: false for the last layer so we don't apply the Swish activation!
    dense_layer(buffer_A, PARAMS_HIDDEN_4_KERNEL, PARAMS_HIDDEN_4_BIAS, buffer_B, HIDDEN_DIM, ACTION_DIM * 2, false);

    // 6. Extract Actions
    for (int i = 0; i < ACTION_DIM; ++i)
    {
        // Because our final layer wrote its output into buffer_B,
        // we read the final values from buffer_B!
        final_action[i] = std::tanh(buffer_B[i]);
    }
}

// --- Madgwick & IMU ---

void madgwick_update_6dof(float gx, float gy, float gz, float ax, float ay, float az, float dt)
{
    float recipNorm, s0, s1, s2, s3, qDot1, qDot2, qDot3, qDot4;
    float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

    qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f)))
    {
        recipNorm = 1.0f / std::sqrt(ax * ax + ay * ay + az * az);
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        _2q0 = 2.0f * q0;
        _2q1 = 2.0f * q1;
        _2q2 = 2.0f * q2;
        _2q3 = 2.0f * q3;
        _4q0 = 4.0f * q0;
        _4q1 = 4.0f * q1;
        _4q2 = 4.0f * q2;
        _8q1 = 8.0f * q1;
        _8q2 = 8.0f * q2;
        q0q0 = q0 * q0;
        q1q1 = q1 * q1;
        q2q2 = q2 * q2;
        q3q3 = q3 * q3;

        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
        s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
        s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;

        recipNorm = 1.0f / std::sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
        s0 *= recipNorm;
        s1 *= recipNorm;
        s2 *= recipNorm;
        s3 *= recipNorm;

        qDot1 -= beta * s0;
        qDot2 -= beta * s1;
        qDot3 -= beta * s2;
        qDot4 -= beta * s3;
    }

    q0 += qDot1 * dt;
    q1 += qDot2 * dt;
    q2 += qDot3 * dt;
    q3 += qDot4 * dt;

    recipNorm = 1.0f / std::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    q0 *= recipNorm;
    q1 *= recipNorm;
    q2 *= recipNorm;
    q3 *= recipNorm;
}

void get_local_gravity(float *gravity_out)
{
    gravity_out[0] = -2.0f * (q1 * q3 + q0 * q2);
    gravity_out[1] = -2.0f * (q2 * q3 - q0 * q1);
    gravity_out[2] = -(q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3);
}

void write_imu_register(uint8_t reg, uint8_t data)
{
    uint8_t buf[2] = {reg, data}; // MSB is 0 for write
    gpio_put(PIN_CS, 0);
    spi_write_blocking(SPI_PORT, buf, 2);
    gpio_put(PIN_CS, 1);
}

void init_imu()
{
    spi_init(SPI_PORT, 5000 * 1000); // 5 MHz SPI
    gpio_set_function(PIN_MISO, GPIO_FUNC_SPI);
    gpio_set_function(PIN_SCK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_MOSI, GPIO_FUNC_SPI);

    gpio_init(PIN_CS);
    gpio_set_dir(PIN_CS, GPIO_OUT);
    gpio_put(PIN_CS, 1);

    // Wake up LSM6DSO: 104Hz, ±4g Accel, ±2000dps Gyro
    write_imu_register(CTRL1_XL, 0x48);
    write_imu_register(CTRL2_G, 0x4C);
    sleep_ms(50); // Give IMU time to start up
}

void read_imu(float *gx, float *gy, float *gz, float *ax, float *ay, float *az)
{
    uint8_t reg = OUTX_L_G | 0x80; // MSB 1 for read
    uint8_t buffer[12];

    gpio_put(PIN_CS, 0);
    spi_write_blocking(SPI_PORT, &reg, 1);
    spi_read_blocking(SPI_PORT, 0x00, buffer, 12);
    gpio_put(PIN_CS, 1);

    int16_t raw_gx = (int16_t)(buffer[1] << 8 | buffer[0]);
    int16_t raw_gy = (int16_t)(buffer[3] << 8 | buffer[2]);
    int16_t raw_gz = (int16_t)(buffer[5] << 8 | buffer[4]);

    int16_t raw_ax = (int16_t)(buffer[7] << 8 | buffer[6]);
    int16_t raw_ay = (int16_t)(buffer[9] << 8 | buffer[8]);
    int16_t raw_az = (int16_t)(buffer[11] << 8 | buffer[10]);

    // 1. Convert to physical units in the sensor's local frame
    float phys_gx = raw_gx * GYRO_SCALE_RAD_S;
    float phys_gy = raw_gy * GYRO_SCALE_RAD_S;
    float phys_gz = raw_gz * GYRO_SCALE_RAD_S;

    float phys_ax = raw_ax * ACCEL_SCALE_G;
    float phys_ay = raw_ay * ACCEL_SCALE_G;
    float phys_az = raw_az * ACCEL_SCALE_G;

    // 2. Map to MuJoCo Coordinate Frame!
    // Physical: X = Right, Y = Down, Z = Forward
    // MuJoCo:   X = Right, Y = Backward (-Forward), Z = Up (-Down)

    *gx = phys_gx;
    *gy = -phys_gz;
    *gz = -phys_gy;

    *ax = phys_ax;
    *ay = -phys_az;
    *az = -phys_ay;
}

// --- Servos & Observation Management ---

void init_servos()
{

    for (int i = 0; i < ACTION_DIM; i++)
    {
        uint pin = SERVO_PINS[i];
        gpio_set_function(pin, GPIO_FUNC_PWM);
        uint slice_num = pwm_gpio_to_slice_num(pin);

        // System clock is 150MHz. Divide by 150 to get 1MHz (1 tick = 1 microsecond)
        pwm_set_clkdiv(slice_num, 150.0f);
        // Wrap at 20000 ticks = 20ms period (50Hz)
        pwm_set_wrap(slice_num, 20000);
        // Default to center (1500us)
        pwm_set_chan_level(slice_num, pwm_gpio_to_channel(pin), 1500);
        pwm_set_enabled(slice_num, true);
    }
}

void update_servos(const float *actions)
{
    for (int i = 0; i < ACTION_DIM; i++)
    {

        // Map network output [-1.0, 1.0] to pulse width
        float pulse_width_us = SERVO_CENTERS[i] + (actions[i] * SERVO_RANGE * SERVO_DIRS[i]);

        // Safety clamping based directly on your MIN_DUTY and MAX_DUTY script limits!
        // Prevents the neural network from commanding <500us or >2500us and destroying the gears.
        pulse_width_us = std::max(500.0f, std::min(2500.0f, pulse_width_us));

        uint slice_num = pwm_gpio_to_slice_num(SERVO_PINS[i]);
        pwm_set_chan_level(slice_num, pwm_gpio_to_channel(SERVO_PINS[i]), (uint16_t)pulse_width_us);
    }
}

void update_observation_buffer(const float *new_action)
{
    // 1. Shift action history left by one frame (ACTION_DIM)
    for (int i = 0; i < HISTORY_DIM - ACTION_DIM; ++i)
    {
        current_obs[i] = current_obs[i + ACTION_DIM];
    }
    // 2. Insert new action at the end of the history section
    for (int i = 0; i < ACTION_DIM; ++i)
    {
        current_obs[(HISTORY_DIM - ACTION_DIM) + i] = new_action[i];
    }

    // 3. Update Clock (frequency = 1.0Hz as per Python code)
    float t = step_counter * 0.02f; // dt is 0.02s
    current_obs[243] = std::sin(2.0f * M_PI * 1.0f * t);
    current_obs[244] = std::cos(2.0f * M_PI * 1.0f * t);
}

// --- Main Control Loop (50Hz) ---
bool control_loop_callback(struct repeating_timer *t)
{
    run_control_step = true;
    return true; // Keep timer repeating
}

int main()
{
    stdio_init_all();
    printf("Starting Rick V2 Pico Setup...\n");

    init_imu();
    init_servos();

    // 1. Force all actions to 0.0f (Center position)
    for (int i = 0; i < ACTION_DIM; i++)
    {
        target_actions[i] = 0.0f;
    }

    // 2. Send the center command to the servos immediately
    update_servos(target_actions);

    // 3. The 10-Second Countdown
    printf("\nServos locked at center. Place the robot on the ground!\n");
    for (int i = 10; i > 0; i--)
    {
        printf("Starting in %d...\n", i);
        sleep_ms(1000); // Sleep for 1 second
    }
    printf("GO!\n\n");

    // 4. Initialize the observation buffer before the loop starts
    update_observation_buffer(target_actions);

    // 5. Start the 50Hz (20ms) hardware timer flag generator
    struct repeating_timer timer;
    add_repeating_timer_us(-20000, control_loop_callback, NULL, &timer);

    float gx, gy, gz, ax, ay, az;

    // The Main Game Loop
    while (true)
    {
        if (run_control_step)
        {
            run_control_step = false;

            read_imu(&gx, &gy, &gz, &ax, &ay, &az);
            madgwick_update_6dof(gx, gy, gz, ax, ay, az, 0.02f);
            get_local_gravity(&current_obs[240]);

            infer_action(current_obs, target_actions);

            update_servos(target_actions);

            update_observation_buffer(target_actions);

            step_counter++;

            // // Optional diagnostic print (runs once per second)
            // if (step_counter % 50 == 0)
            // {
            //     printf("Step %d | Executing Policy...\n", step_counter);
            // }
        }
        tight_loop_contents();
    }
    return 0;
}

// calibration
// int main()
// {
//     stdio_init_all();

//     // Give USB serial a moment to connect
//     sleep_ms(2000);

//     init_servos();

//     // Create an array of pure 0.0f actions.
//     // When passed to update_servos, this forces them to sit exactly at SERVO_CENTERS.
//     float zero_actions[ACTION_DIM] = {0.0f};
//     update_servos(zero_actions);

//     printf("\n====================================\n");
//     printf("--- LIVE SERVO CALIBRATION MODE ---\n");
//     printf("====================================\n");
//     printf("Controls:\n");
//     printf("  [0] to [5] : Select a Servo to tune\n");
//     printf("  [W] / [S]  : Increase/Decrease center by 5us\n");
//     printf("  [P]        : Print the final array to copy/paste\n\n");

//     int active_servo = 0;
//     printf("Currently tuning Servo %d. Press W or S to adjust.\n", active_servo);

//     while (true)
//     {
//         // Read keyboard input without freezing the microcontroller
//         int c = getchar_timeout_us(10000);

//         if (c != PICO_ERROR_TIMEOUT)
//         {
//             if (c >= '0' && c <= '5')
//             {
//                 active_servo = c - '0';
//                 printf("\n-> Selected Servo %d\n", active_servo);
//             }
//             else if (c == 'w' || c == 'W')
//             {
//                 SERVO_CENTERS[active_servo] += 5.0f;
//                 update_servos(zero_actions); // Apply immediately
//                 printf("Servo %d Center: %.1f us\n", active_servo, SERVO_CENTERS[active_servo]);
//             }
//             else if (c == 's' || c == 'S')
//             {
//                 SERVO_CENTERS[active_servo] -= 5.0f;
//                 update_servos(zero_actions); // Apply immediately
//                 printf("Servo %d Center: %.1f us\n", active_servo, SERVO_CENTERS[active_servo]);
//             }
//             else if (c == 'p' || c == 'P')
//             {
//                 printf("\n--- CALIBRATION COMPLETE ---\n");
//                 printf("Copy and paste this array into the top of your main.cpp:\n\n");
//                 printf("float SERVO_CENTERS[ACTION_DIM] = {\n");
//                 printf("    %.1ff, %.1ff, %.1ff, \n", SERVO_CENTERS[0], SERVO_CENTERS[1], SERVO_CENTERS[2]);
//                 printf("    %.1ff, %.1ff, %.1ff\n", SERVO_CENTERS[3], SERVO_CENTERS[4], SERVO_CENTERS[5]);
//                 printf("};\n\n");
//             }
//         }
//         tight_loop_contents();
//     }
//     return 0;
// }
