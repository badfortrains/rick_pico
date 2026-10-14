#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include "control_timing.h"
#include "policy_inference.h"

// The pre-optimization deployment implementation is the numerical reference.
static void reference_policy(const float *obs, float *action) {
    float normalized[POLICY_OBS_DIM];
    float a[POLICY_MAX_LAYER_DIM], b[POLICY_MAX_LAYER_DIM];
    for (int i = 0; i < POLICY_OBS_DIM; ++i) {
        normalized[i] = (obs[i] - POLICY_OBS_MEAN[i]) / POLICY_OBS_STD[i];
    }
    const float *input = normalized;
    float *output = a;
    for (int layer = 0; layer < POLICY_DENSE_LAYER_COUNT; ++layer) {
        const int width = POLICY_LAYER_OUT_DIMS[layer];
        for (int j = 0; j < width; ++j) {
            float sum = POLICY_LAYER_BIASES[layer][j];
            for (int i = 0; i < POLICY_LAYER_IN_DIMS[layer]; ++i) {
                sum += input[i] * POLICY_LAYER_KERNELS[layer][i * width + j];
            }
            output[j] = layer + 1 == POLICY_DENSE_LAYER_COUNT
                ? sum : rick::swish(sum);
        }
        input = output;
        output = output == a ? b : a;
    }
    for (int i = 0; i < POLICY_ACTION_DIM; ++i) action[i] = std::tanh(input[i]);
}

int main() {
    rick::PolicyNetwork network;
    std::mt19937 rng(17);
    std::normal_distribution<float> normal;
    float obs[POLICY_OBS_DIM], actual[POLICY_ACTION_DIM], expected[POLICY_ACTION_DIM];
    float max_error = 0;
    for (int sample = 0; sample < 128; ++sample) {
        for (int i = 0; i < POLICY_OBS_DIM; ++i) {
            obs[i] = POLICY_OBS_MEAN[i] + normal(rng) * POLICY_OBS_STD[i];
        }
        network.infer(obs, actual);
        reference_policy(obs, expected);
        for (int i = 0; i < POLICY_ACTION_DIM; ++i) {
            assert(std::isfinite(actual[i]));
            max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
        }
    }
    assert(max_error <= 1e-6f);

    // A deadline miss counts elapsed releases, without running stale catch-up steps.
    rick::ControlSchedule schedule(20000);
    uint64_t missed = 0;
    schedule.reset(1000);
    assert(!schedule.take(20999, missed));
    assert(schedule.take(21000, missed) && missed == 0);
    assert(schedule.next_us() == 41000);
    assert(schedule.take(105000, missed) && missed == 3);
    assert(schedule.next_us() == 121000);
    assert(!schedule.take(105000, missed));
    assert(schedule.take(121000, missed) && missed == 0);

    // Deadlines and IMU intervals survive the 32-bit microsecond rollover.
    const uint64_t near_wrap = (uint64_t{1} << 32) - 10000;
    schedule.reset(near_wrap);
    assert(schedule.take(near_wrap + 20000, missed) && missed == 0);
    rick::SampleClock imu;
    assert(imu.step(near_wrap, 0.02f) == 0.02f);
    assert(std::abs(imu.step(near_wrap + 100000, 0.02f) - 0.1f) < 1e-7f);
    imu.reset();
    assert(imu.step(near_wrap + 30000000, 0.02f) == 0.02f);

    // Gait phase follows real time even if four controller releases are missed.
    rick::GaitClock regular, delayed;
    assert(regular.sample(1000, 0.8f) == 0);
    assert(delayed.sample(1000, 0.8f) == 0);
    float regular_phase = 0;
    for (int i = 1; i <= 5; ++i) regular_phase = regular.sample(1000 + i * 20000, 0.8f);
    assert(std::abs(regular_phase - delayed.sample(101000, 0.8f)) < 1e-6f);
    delayed.reset();
    assert(delayed.sample(9000000, 0.8f) == 0);
    assert(std::abs(delayed.sample(10250000, 0.8f)) < 1e-6f);

    rick::ControlTiming timing;
    timing.reset(1000);
    timing.record(21000, 24000, 2500, 0, 41000);
    timing.record(41000, 71000, 29500, 0, 61000);
    timing.record(105000, 108000, 2500, 2, 121000);
    assert(timing.samples == 3 && timing.interval_samples == 2);
    assert(timing.inference_total_us == 34500 && timing.inference_max_us == 29500);
    assert(timing.step_max_us == 30000);
    assert(timing.interval_min_us == 20000 && timing.interval_max_us == 64000);
    assert(timing.missed_ticks == 2 && timing.overruns == 1);
    timing.reset(108000, true);
    timing.record(121000, 124000, 2500, 0, 141000);
    assert(timing.samples == 1 && timing.interval_min_us == 16000);
    assert(timing.missed_ticks == 0 && timing.overruns == 0);
    timing.reset(130000);
    assert(timing.previous_started_us == 0 && timing.samples == 0);
    std::printf("Policy parity: 128 observations, max action error %.9g; timing checks passed.\n", max_error);
}
