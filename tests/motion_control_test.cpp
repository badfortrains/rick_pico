#include <cassert>
#include <cmath>
#include <cstdio>
#include "motion_control.h"

int main(int argc, char **argv) {
    const rick::MotionCommand low{-.04f, -.6f}, high{.08f, .6f};
    rick::MotionCommand target;
    const uint8_t left_reverse[] = {6, 0xd8, 0xff, 0x58, 0x02};
    assert(rick::decode_motion_command(left_reverse, 5, low, high, target));
    assert(target.forward == -.04f && target.yaw == .6f);
    const uint8_t right_forward[] = {6, 80, 0, 0xa8, 0xfd};
    assert(rick::decode_motion_command(right_forward, 5, low, high, target));
    assert(target.forward == .08f && target.yaw == -.6f);
    const uint8_t too_fast[] = {6, 81, 0, 0, 0};
    const uint8_t too_far[] = {6, 0, 0, 0x59, 2};
    const uint8_t old_start[] = {2};
    assert(!rick::decode_motion_command(too_fast, 5, low, high, target));
    assert(!rick::decode_motion_command(too_far, 5, low, high, target));
    assert(!rick::decode_motion_command(old_start, 1, low, high, target));
    assert(!rick::decode_motion_command(left_reverse, 4, low, high, target));
    assert(!rick::decode_motion_command(left_reverse, 0, low, high, target));
    rick::MotionCommand value;
    target = {.08f, .6f};
    float heading = 0;
    FILE *fixture = argc > 1 ? std::fopen(argv[1], "w") : nullptr;
    for (int i=0; i<200; ++i) {
        if (i == 90) target = {-.04f, -.6f};
        if (i == 160) target = {};
        const auto previous = value;
        rick::step_motion_command(value, target, {.05f, 1.f}, .02f, heading);
        assert(std::abs(value.forward - previous.forward) < .001001f);
        assert(std::abs(value.yaw - previous.yaw) < .020001f);
        assert(std::isfinite(heading));
        if (fixture) std::fprintf(fixture, "%.9g,%.9g,%.9g\n", value.forward, value.yaw, heading);
    }
    if (fixture) std::fclose(fixture);
    assert(std::abs(value.forward) < 1e-7f && std::abs(value.yaw) < 1e-7f);
    std::puts("Motion protocol bounds, sign, and smoothing passed.");
}
