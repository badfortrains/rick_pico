#include "imu_raw_capture.h"
#include <cassert>
#include <cstdio>
#include <vector>

struct FakeImu {
    uint8_t reg = 0x13;
    unsigned reads = 0;
    bool fail_read = false, fail_restore = false, corrupt_config = false;
    std::vector<uint8_t> writes;
    bool configure(uint8_t value) {
        writes.push_back(value);
        if (fail_restore && value == 0x13) return false;
        reg = value;
        return true;
    }
    bool config(uint8_t &value) { value = corrupt_config ? 0xff : reg; return true; }
    bool read(int16_t values[3]) {
        ++reads;
        if (fail_read) return false;
        const int scale = 1 << (reg >> 4);
        values[0] = -16000 / scale;
        values[1] = 8000 / scale;
        values[2] = -4000 / scale;
        return true;
    }
};

int main(int argc, char **argv) {
    using Capture = rick::ImuRawCapture;
    FakeImu io;
    Capture capture;
    assert(!capture.start(io, 1, 7, 0));
    assert(io.writes.empty());
    assert(capture.start(io, 0xabcd, 3, 0));
    assert(!capture.start(io, 2, 2, 0));
    capture.tick(io, 249999);
    assert(io.reads == 0);
    // A delayed scheduler reads one sample, not a burst of duplicate samples.
    capture.tick(io, 1000000);
    capture.tick(io, 1000000);
    assert(io.reads == 1);
    for (uint64_t now = 1020000; capture.active(); now += 20000) {
        assert(now < 6000000);
        capture.tick(io, now);
    }
    assert(capture.state == Capture::DONE && capture.completed == 3);
    assert(io.reads == 150 && io.reg == 0x13);
    assert((io.writes == std::vector<uint8_t>{0x03, 0x13, 0x23, 0x13}));
    assert(capture.rows[0].sum[0] == -800000);
    assert(capture.rows[1].sum[0] == -400000);
    assert(capture.rows[2].sum[0] == -200000);
    uint8_t packet[Capture::PACKET_SIZE];
    capture.encode(packet);
    assert(packet[4] == 0xcd && packet[5] == 0xab);
    if (argc == 2) {
        FILE *file = std::fopen(argv[1], "wb"); assert(file);
        assert(std::fwrite(packet, 1, sizeof(packet), file) == sizeof(packet));
        std::fclose(file);
    }

    // Link loss/Stop/OTA all use cancel(), which must restore the normal range.
    for (uint64_t cancel_at : {0ULL, 1500000ULL, 2800000ULL}) {
        FakeImu device;
        Capture test;
        test.start(device, 1, 1, 0);
        for (uint64_t now = 0; now < cancel_at; now += 20000) test.tick(device, now);
        test.cancel(device);
        assert(test.state == Capture::CANCELLED && device.reg == 0x13);
        const unsigned reads = device.reads;
        test.tick(device, 9000000);
        assert(device.reads == reads);
    }
    FakeImu broken;
    Capture failure;
    failure.start(broken, 2, 2, 0);
    broken.fail_read = true;
    failure.tick(broken, 250000);
    assert(failure.state == Capture::ERROR && failure.error == Capture::READ_FAILED);
    assert(broken.reg == 0x13);

    FakeImu changed;
    Capture config_failure;
    config_failure.start(changed, 3, 3, 0);
    changed.reg = 0x23; // Unexpected range after settling.
    config_failure.tick(changed, 250000);
    assert(config_failure.error == Capture::CONFIG_FAILED && changed.reg == 0x13);
    assert(changed.reads == 0);

    FakeImu stuck;
    Capture restore_failure;
    restore_failure.start(stuck, 4, 4, 0);
    stuck.fail_restore = true;
    restore_failure.cancel(stuck);
    assert(restore_failure.restore_failed && restore_failure.error == Capture::RESTORE_FAILED);
    assert(!restore_failure.start(stuck, 5, 5, 1000000));
    restore_failure.encode(packet);
    assert(packet[9] == 1 && packet[1] == Capture::ERROR);
    std::puts("IMU raw capture tests passed");
}
