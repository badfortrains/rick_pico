#ifndef RICK_GAIT_DIAGNOSTICS_H
#define RICK_GAIT_DIAGNOSTICS_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace rick {

// Bounded RAM capture, independent of BLE latency. Caller holds its mutex for
// start/record/finish/select/encode. Wire format is explicitly little endian.
class GaitDiagnostics {
public:
    static constexpr unsigned CAPACITY = 256, PRE_ROLL = 25;
    static constexpr unsigned HEADER_SIZE = 184, FRAME_SIZE = 188;
    enum State : uint8_t { EMPTY, RECORDING, COMPLETE, INTERRUPTED, FULL };
    enum Reason : uint8_t { NONE, DURATION, STOP, RESET, SLEEP, OTA, RAW_TEST, CAPACITY_LIMIT };
    enum Flags : uint32_t { READ_OK = 1, POLICY_APPLIED = 2, BIAS_READY = 4, RAW_CLIPPED = 8, PWM_ENABLED = 16 };
    struct Frame {
        uint64_t time_us = 0;
        uint32_t step = UINT32_MAX, flags = 0, skipped = 0, read_errors = 0;
        uint32_t inference_us = 0, control_us = 0;
        float imu_dt = 0;
        int16_t raw[6] = {}; // Sensor axes: accel XYZ, gyro XYZ.
        float accel[3] = {}, gyro_before[3] = {}, gyro[3] = {};
        float quat[4] = {}, gravity[3] = {}, phase = 0, heading = 0;
        float action[8] = {}, bias[3] = {};
        uint16_t pulse_us[8] = {};
    };
    struct Metadata {
        uint32_t checkpoint = 0, duration_us = 0;
        float control_dt = 0, frequency = 0, target_velocity = 0, action_scale = 0;
        float heading_reference = 0, gyro_scale = 0, beta = 0;
        float bias[3] = {}, initial_action[8] = {}, initial_quat[4] = {};
        uint16_t centers[8] = {}, obs_dim = 0;
        uint8_t heading_enabled = 0, calibration_enabled = 0;
        uint8_t accel_config = 0, gyro_config = 0;
        float accel_zero[3] = {}, accel_counts_per_g = 0, gyro_obs_scale = 0;
        char build[24] = {};
        uint32_t mounting_calibration_id = 0;
    };

    void start(uint64_t now, const Metadata &metadata) {
        if (++generation_ == 0) ++generation_;
        start_us_ = now;
        metadata_ = metadata;
        count_ = 0;
        for (unsigned i = 0; i < pre_count_; ++i) {
            const Frame &frame = pre_[(pre_next_ + PRE_ROLL - pre_count_ + i) % PRE_ROLL];
            if (frame.time_us <= now && now - frame.time_us <= 500000)
                frames_[count_++] = frame;
        }
        pre_count_at_start_ = count_;
        pre_count_ = pre_next_ = 0;
        state_ = RECORDING;
        reason_ = NONE;
        selected_index_ = UINT16_MAX;
    }

    void record(const Frame &frame, bool running) {
        if (state_ == RECORDING) {
            if (count_ < CAPACITY) frames_[count_++] = frame;
            if (frame.time_us >= start_us_ + metadata_.duration_us) {
                state_ = COMPLETE; reason_ = DURATION;
            } else if (count_ == CAPACITY) {
                state_ = FULL; reason_ = CAPACITY_LIMIT;
            }
        } else if (!running) {
            pre_[pre_next_] = frame;
            pre_next_ = (pre_next_ + 1) % PRE_ROLL;
            pre_count_ = std::min(pre_count_ + 1, PRE_ROLL);
        }
    }

    void finish(Reason reason) {
        if (state_ == RECORDING) { state_ = INTERRUPTED; reason_ = reason; }
    }

    // Selection writes never start/stop motion, consume data, or reset a log.
    void select_header() { selected_index_ = UINT16_MAX; }
    void select_frame(uint32_t generation, uint16_t index) {
        selected_generation_ = generation; selected_index_ = index;
    }
    unsigned encode(uint8_t *out) const {
        std::fill(out, out + FRAME_SIZE, uint8_t(0));
        const bool header = selected_index_ == UINT16_MAX;
        const bool valid = !header && state_ != RECORDING && state_ != EMPTY &&
                           selected_generation_ == generation_ && selected_index_ < count_;
        out[0] = 1; out[1] = header ? 0 : (valid ? 1 : 2);
        out[2] = state_; out[3] = reason_;
        put(out + 4, generation_, 4); put(out + 8, selected_index_, 2); put(out + 10, count_, 2);
        if (header) {
            put(out + 12, pre_count_at_start_, 2); put(out + 14, CAPACITY, 2);
            put(out + 16, metadata_.checkpoint, 4);
            scalar(out + 20, metadata_.control_dt); scalar(out + 24, metadata_.frequency);
            scalar(out + 28, metadata_.target_velocity); scalar(out + 32, metadata_.action_scale);
            scalar(out + 36, metadata_.heading_reference); scalar(out + 40, metadata_.gyro_scale);
            scalar(out + 44, metadata_.beta); floats(out + 48, metadata_.bias, 3);
            floats(out + 60, metadata_.initial_action, 8);
            for (unsigned i = 0; i < 8; ++i) put(out + 92 + 2*i, metadata_.centers[i], 2);
            put(out + 108, metadata_.obs_dim, 2); put(out + 110, 8, 2);
            put(out + 112, metadata_.duration_us, 4);
            out[116] = metadata_.heading_enabled; out[117] = metadata_.calibration_enabled;
            out[118] = metadata_.accel_config; out[119] = metadata_.gyro_config;
            floats(out + 120, metadata_.accel_zero, 3);
            scalar(out + 132, metadata_.accel_counts_per_g); scalar(out + 136, metadata_.gyro_obs_scale);
            floats(out + 140, metadata_.initial_quat, 4);
            std::memcpy(out + 156, metadata_.build, sizeof(metadata_.build));
            put(out + 180, metadata_.mounting_calibration_id, 4);
            return HEADER_SIZE;
        }
        if (!valid) return 12;
        const Frame &f = frames_[selected_index_];
        const int64_t relative = int64_t(f.time_us) - int64_t(start_us_);
        put(out + 12, uint32_t(int32_t(relative)), 4); put(out + 16, f.step, 4);
        put(out + 20, f.flags, 4); scalar(out + 24, f.imu_dt);
        put(out + 28, f.skipped, 4); put(out + 32, f.read_errors, 4);
        put(out + 36, f.inference_us, 4); put(out + 40, f.control_us, 4);
        for (unsigned i = 0; i < 6; ++i) put(out + 44 + 2*i, uint16_t(f.raw[i]), 2);
        floats(out + 56, f.accel, 3); floats(out + 68, f.gyro_before, 3);
        floats(out + 80, f.gyro, 3); floats(out + 92, f.quat, 4);
        floats(out + 108, f.gravity, 3); scalar(out + 120, f.phase);
        // Pre-roll samples were acquired before the new START reference existed.
        const float relative_heading = f.heading - metadata_.heading_reference;
        scalar(out + 124, std::atan2(std::sin(relative_heading), std::cos(relative_heading)));
        floats(out + 128, f.action, 8);
        for (unsigned i = 0; i < 8; ++i) put(out + 160 + 2*i, f.pulse_us[i], 2);
        floats(out + 176, f.bias, 3);
        return FRAME_SIZE;
    }

private:
    Frame frames_[CAPACITY], pre_[PRE_ROLL];
    Metadata metadata_;
    uint64_t start_us_ = 0;
    uint32_t generation_ = 0, selected_generation_ = 0;
    unsigned count_ = 0, pre_count_ = 0, pre_next_ = 0, pre_count_at_start_ = 0;
    uint16_t selected_index_ = UINT16_MAX;
    State state_ = EMPTY;
    Reason reason_ = NONE;
    static void put(uint8_t *out, uint32_t value, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) out[i] = uint8_t(value >> (8*i));
    }
    static void scalar(uint8_t *out, float value) {
        static_assert(sizeof(float) == 4, "Diagnostic wire format requires float32");
        uint32_t bits; std::memcpy(&bits, &value, 4); put(out, bits, 4);
    }
    static void floats(uint8_t *out, const float *values, unsigned count) {
        for (unsigned i = 0; i < count; ++i) scalar(out + 4*i, values[i]);
    }
};
}  // namespace rick
#endif
