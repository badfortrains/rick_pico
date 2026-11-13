#include "gait_diagnostics.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

using Capture = rick::GaitDiagnostics;
uint32_t u32(const uint8_t *p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
float f32(const uint8_t *p) { const uint32_t bits = u32(p); float x; std::memcpy(&x, &bits, 4); return x; }
unsigned count(const uint8_t *p) { return unsigned(p[10]) | (unsigned(p[11]) << 8); }

int main(int argc, char **argv) {
    Capture capture;
    uint8_t packet[Capture::FRAME_SIZE];
    assert(capture.encode(packet) == Capture::HEADER_SIZE && packet[2] == Capture::EMPTY);
    Capture::Metadata meta;
    meta.duration_us = 40000; meta.frequency = 1.5f; meta.control_dt = .02f;
    meta.heading_reference = .2f; meta.checkpoint = 73728000; meta.obs_dim = 107;
    meta.initial_quat[0] = 1; meta.accel_counts_per_g = 4092.576f;
    meta.mounting_calibration_id = 1;
    meta.initial_action[7] = -.25f; meta.centers[7] = 1560;
    std::strcpy(meta.build, "test-build");
    Capture::Frame frame;
    frame.time_us = 980000; frame.heading = .1f; frame.flags = Capture::READ_OK;
    frame.raw[0] = -12345; frame.raw[5] = 2345;
    frame.accel[2] = 1; frame.gyro_before[2] = .3f; frame.gyro[2] = .2f;
    frame.bias[2] = .1f; frame.quat[0] = 1; frame.gravity[2] = -1;
    frame.pulse_us[7] = 1601; frame.action[7] = .5f; frame.imu_dt = .02f;
    capture.record(frame, false);
    capture.start(1000000, meta);
    capture.select_frame(1, 0);
    assert(capture.encode(packet) == 12 && packet[1] == 2); // No live download.
    frame.time_us = 1020000; frame.flags |= Capture::POLICY_APPLIED;
    frame.step = 0; frame.inference_us = 1234;
    capture.record(frame, true);
    frame.time_us = 1040000; frame.step = 1;
    capture.record(frame, true);
    capture.select_header();
    assert(capture.encode(packet) == Capture::HEADER_SIZE);
    assert(packet[2] == Capture::COMPLETE && packet[3] == Capture::DURATION);
    assert(count(packet) == 3 && packet[12] == 1 && u32(packet + 4) == 1);
    assert(u32(packet + 180) == 1);
    FILE *fixture = argc == 2 ? std::fopen(argv[1], "wb") : nullptr;
    if (argc == 2) { assert(fixture); std::fwrite(packet, 1, Capture::HEADER_SIZE, fixture); }
    for (unsigned i = 0; i < 3; ++i) {
        capture.select_frame(1, i);
        assert(capture.encode(packet) == Capture::FRAME_SIZE);
        assert(packet[1] == 1 && count(packet) == 3);
        if (i == 0) {
            assert(int32_t(u32(packet + 12)) == -20000);
            assert(std::abs(f32(packet + 124) + .1f) < 1e-6f); // New START reference.
            assert(u32(packet + 16) == UINT32_MAX);
        }
        assert(std::abs(f32(packet + 184) - .1f) < 1e-6f);
        if (fixture) std::fwrite(packet, 1, Capture::FRAME_SIZE, fixture);
    }
    if (fixture) std::fclose(fixture);
    capture.finish(Capture::STOP);
    // Completed frames survive later idle samples, Stop, and repeated reads.
    for (unsigned i = 0; i < 40; ++i) { frame.time_us += 20000; capture.record(frame, false); }
    capture.select_frame(1, 0); capture.encode(packet);
    assert(packet[2] == Capture::COMPLETE && int32_t(u32(packet + 12)) == -20000);
    capture.start(frame.time_us + 20000, meta);
    capture.select_header(); capture.encode(packet);
    assert(count(packet) == 25 && packet[12] == 25); // Ring preserves latest samples.
    capture.finish(Capture::STOP);
    capture.select_frame(1, 0);
    assert(capture.encode(packet) == 12 && packet[1] == 2); // Stale generation rejected.
    capture.select_frame(2, 0); capture.encode(packet);
    assert(int32_t(u32(packet + 12)) == -500000 && packet[2] == Capture::INTERRUPTED);
    capture.select_frame(2, 25); assert(capture.encode(packet) == 12);

    // Old pre-roll must not masquerade as current after sleep/raw tests.
    capture.start(frame.time_us + 2000000, meta);
    capture.select_header(); capture.encode(packet); assert(count(packet) == 0);
    // Failed reads still have timestamps and are retained, with sensor NaNs.
    frame.time_us += 2020000; frame.flags = 0; frame.imu_dt = NAN;
    capture.record(frame, true); capture.finish(Capture::STOP);
    capture.select_frame(3, 0); capture.encode(packet);
    assert(u32(packet + 20) == 0 && std::isnan(f32(packet + 24)));

    Capture bounded;
    meta.duration_us = 10000000;
    bounded.start(0, meta);
    for (unsigned i = 1; i <= 300; ++i) { frame.time_us = i * 20000; bounded.record(frame, true); }
    bounded.encode(packet);
    assert(packet[2] == Capture::FULL && packet[3] == Capture::CAPACITY_LIMIT && count(packet) == 256);
    for (auto reason : {Capture::STOP, Capture::RESET, Capture::SLEEP, Capture::OTA, Capture::RAW_TEST}) {
        bounded.start(0, meta); bounded.record(frame, true); bounded.finish(reason);
        bounded.select_header(); bounded.encode(packet);
        assert(packet[2] == Capture::INTERRUPTED && packet[3] == reason && count(packet) == 1);
    }
    std::puts("Gait diagnostics capture and wire-format tests passed");
}
