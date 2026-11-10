#ifndef RICK_IMU_RAW_CAPTURE_H
#define RICK_IMU_RAW_CAPTURE_H

#include <algorithm>
#include <cstdint>

namespace rick {

// Version 1 BLE response: 12-byte header, then three 28-byte range records.
// All multibyte fields are little endian. Counts stay in sensor coordinates.
class ImuRawCapture {
public:
    enum State : uint8_t { IDLE, RUNNING, DONE, ERROR, CANCELLED };
    enum Error : uint8_t { NONE, READ_FAILED, CONFIG_FAILED, RESTORE_FAILED };
    static constexpr unsigned PACKET_SIZE = 96;
    static constexpr unsigned SAMPLE_COUNT = 50;
    struct Row {
        uint8_t range = 0, config = 0, samples = 0, clipped = 0;
        int32_t sum[3] = {};
        int16_t minimum[3] = {32767, 32767, 32767};
        int16_t maximum[3] = {-32768, -32768, -32768};
    };
    State state = IDLE;
    Error error = NONE;
    bool restore_failed = false;
    uint16_t token = 0;
    uint8_t pose = 0, completed = 0;
    Row rows[3];

    bool active() const { return state == RUNNING; }

    // IO supplies configure(uint8_t), config(uint8_t&), read(int16_t[3]).
    template<class IO>
    bool start(IO &io, uint16_t request_token, uint8_t request_pose, uint64_t now) {
        if (active() || restore_failed || request_pose < 1 || request_pose > 6) return false;
        token = request_token;
        pose = request_pose;
        completed = 0;
        error = NONE;
        for (auto &row : rows) row = Row{};
        state = RUNNING;
        select_range(io, now);
        return true;
    }

    template<class IO>
    void cancel(IO &io) {
        if (active()) finish(io, CANCELLED, NONE);
    }

    template<class IO>
    void tick(IO &io, uint64_t now) {
        if (!active() || now < next_sample_) return;
        Row &row = rows[completed];
        // Check again after settling and at the end of every range.
        if (row.samples == 0 && !check_config(io, row)) {
            finish(io, ERROR, CONFIG_FAILED);
            return;
        }
        int16_t values[3];
        if (!io.read(values)) {
            finish(io, ERROR, READ_FAILED);
            return;
        }
        for (unsigned axis = 0; axis < 3; ++axis) {
            row.sum[axis] += values[axis];
            row.minimum[axis] = std::min(row.minimum[axis], values[axis]);
            row.maximum[axis] = std::max(row.maximum[axis], values[axis]);
            row.clipped |= values[axis] <= -32760 || values[axis] >= 32760;
        }
        ++row.samples;
        next_sample_ = now + 20000;  // Never catch up by reading duplicates.
        if (row.samples == SAMPLE_COUNT) {
            if (!check_config(io, row)) {
                finish(io, ERROR, CONFIG_FAILED);
                return;
            }
            if (++completed == 3) finish(io, DONE, NONE);
            else select_range(io, now);
        }
    }

    void encode(uint8_t *out) const {
        std::fill(out, out + PACKET_SIZE, uint8_t(0));
        out[0] = 1; out[1] = state; out[2] = error; out[3] = pose;
        put(out + 4, token, 2); out[6] = completed;
        out[7] = active() ? rows[completed].samples : 0;
        out[8] = active() ? rows[completed].range : 0;
        out[9] = restore_failed;
        for (unsigned index = 0; index < 3; ++index) {
            const auto &row = rows[index];
            uint8_t *b = out + 12 + 28 * index;
            b[0] = row.range; b[1] = row.config; b[2] = row.samples; b[3] = row.clipped;
            for (unsigned axis = 0; axis < 3; ++axis) {
                put(b + 4 + 4 * axis, uint32_t(row.sum[axis]), 4);
                put(b + 16 + 2 * axis, uint16_t(row.minimum[axis]), 2);
                put(b + 22 + 2 * axis, uint16_t(row.maximum[axis]), 2);
            }
        }
    }

private:
    uint64_t next_sample_ = 0;
    static void put(uint8_t *out, uint32_t value, unsigned bytes) {
        for (unsigned i = 0; i < bytes; ++i) out[i] = uint8_t(value >> (8 * i));
    }
    template<class IO>
    bool check_config(IO &io, Row &row) {
        return io.config(row.config) && row.config == uint8_t((completed << 4) | 3);
    }
    template<class IO>
    void select_range(IO &io, uint64_t now) {
        auto &row = rows[completed];
        row.range = uint8_t(2 << completed);
        if (!io.configure(uint8_t((completed << 4) | 3)) || !check_config(io, row)) {
            finish(io, ERROR, CONFIG_FAILED);
            return;
        }
        next_sample_ = now + 250000;
    }
    template<class IO>
    void finish(IO &io, State final_state, Error final_error) {
        uint8_t actual = 0;
        restore_failed = !io.configure(0x13) || !io.config(actual) || actual != 0x13;
        state = restore_failed ? ERROR : final_state;
        error = restore_failed ? RESTORE_FAILED : final_error;
    }
};
}  // namespace rick
#endif
