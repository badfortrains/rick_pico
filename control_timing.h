#ifndef RICK_CONTROL_TIMING_H
#define RICK_CONTROL_TIMING_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace rick {

// Monotonic deadlines, serviced on the main thread. Never replay a backlog
// of policy actions against one stale IMU sample.
class ControlSchedule {
public:
    explicit ControlSchedule(uint32_t period_us) : period_us_(period_us) {}

    void reset(uint64_t now_us) { next_us_ = now_us + period_us_; }

    bool take(uint64_t now_us, uint64_t &skipped) {
        if (now_us < next_us_) return false;
        skipped = (now_us - next_us_) / period_us_;
        next_us_ += (skipped + 1) * period_us_;
        return true;
    }

    uint64_t next_us() const { return next_us_; }

private:
    uint32_t period_us_;
    uint64_t next_us_ = 0;
};

class SampleClock {
public:
    void reset() { initialized_ = false; }

    float step(uint64_t now_us, float first_dt) {
        const float dt = initialized_
            ? static_cast<float>(now_us - previous_us_) * 1.0e-6f : first_dt;
        previous_us_ = now_us;
        initialized_ = true;
        return dt;
    }

private:
    bool initialized_ = false;
    uint64_t previous_us_ = 0;
};

class GaitClock {
public:
    void reset() { sample_clock_.reset(); phase_ = 0.0f; }

    float sample(uint64_t now_us, float frequency) {
        constexpr float pi = 3.14159265358979323846f;
        const float dt = sample_clock_.step(now_us, 0.0f);
        phase_ = std::fmod(phase_ + 2.0f * pi * frequency * dt + pi,
                           2.0f * pi) - pi;
        return phase_;
    }

private:
    SampleClock sample_clock_;
    float phase_ = 0.0f;
};

struct ControlTiming {
    uint64_t window_started_us = 0;
    uint64_t previous_started_us = 0;
    uint64_t samples = 0;
    uint64_t interval_samples = 0;
    uint64_t inference_total_us = 0;
    uint64_t inference_max_us = 0;
    uint64_t step_max_us = 0;
    uint64_t interval_min_us = std::numeric_limits<uint64_t>::max();
    uint64_t interval_max_us = 0;
    uint64_t missed_ticks = 0;
    uint64_t overruns = 0;

    void reset(uint64_t now_us, bool keep_previous = false) {
        const uint64_t previous = keep_previous ? previous_started_us : 0;
        *this = ControlTiming{};
        window_started_us = now_us;
        previous_started_us = previous;
    }

    void record(uint64_t start_us, uint64_t finish_us, uint64_t inference_us,
                uint64_t skipped, uint64_t next_deadline_us) {
        if (previous_started_us != 0) {
            const uint64_t interval = start_us - previous_started_us;
            interval_min_us = std::min(interval_min_us, interval);
            interval_max_us = std::max(interval_max_us, interval);
            ++interval_samples;
        }
        previous_started_us = start_us;
        ++samples;
        inference_total_us += inference_us;
        inference_max_us = std::max(inference_max_us, inference_us);
        step_max_us = std::max(step_max_us, finish_us - start_us);
        missed_ticks += skipped;
        overruns += finish_us > next_deadline_us;
    }
};

}  // namespace rick
#endif
