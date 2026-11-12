#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>
#include "imu_wake.h"

using Error = rick::ImuWakeError;
struct Sample { bool ok; float x, y, z; };
struct FakeImu {
    bool bus_ok = true;
    int fail_write = -1;
    Error config_error = Error::NONE;
    unsigned reads = 0, configured = 0, elapsed_ms = 0;
    std::vector<std::pair<unsigned, unsigned>> writes;
    std::vector<unsigned> write_times;
    std::vector<Sample> samples = {{true, 0, 0, 1}};
    bool begin() { return bus_ok; }
    bool write(unsigned reg, unsigned value) {
        writes.emplace_back(reg, value);
        write_times.push_back(elapsed_ms);
        return int(writes.size()) != fail_write;
    }
    Error configure() { ++configured; return config_error; }
    bool read_accel(float &x, float &y, float &z) {
        const auto &s = samples[std::min<size_t>(reads++, samples.size()-1)];
        x=s.x; y=s.y; z=s.z;
        return s.ok;
    }
    void wait_ms(unsigned ms) { elapsed_ms += ms; }
};

void check_failure(FakeImu io, Error expected) {
    float q[4]={1,2,3,4};
    assert(rick::resume_imu(io,q) == expected);
    // A failed wake must never publish a partially computed attitude.
    assert(q[0]==1 && q[1]==2 && q[2]==3 && q[3]==4);
    assert(io.reads <= 100);
    assert(io.elapsed_ms <= 2200);
}

int main() {
    FakeImu io;
    float q[4];
    assert(rick::resume_imu(io,q) == Error::NONE);
    assert((io.writes == std::vector<std::pair<unsigned,unsigned>>{
        {0x30,0x00}, {0xd1,0x6b}, {0xd1,0x68}}));
    assert(io.write_times[1] >= io.write_times[0]+100);
    assert(io.configured == 1 && io.reads == 3);
    assert(io.elapsed_ms >= 260);
    assert(std::abs(q[0]-1) < 1e-6f);

    // Reproduce the old false failure: startup read fails or gravity is not yet
    // usable, then the sensor settles. Invalid samples reset the valid streak.
    io = FakeImu{};
    io.samples = {{false,0,0,0}, {true,0,0,0}, {true,1,0,0},
                  {true,4,0,0}, {true,1,0,0}, {true,1,0,0}, {true,1,0,0}};
    assert(rick::resume_imu(io,q) == Error::NONE);
    assert(io.reads == 7);
    assert(std::abs(2*(q[0]*q[2]-q[1]*q[3])+1) < 1e-5f);

    io = FakeImu{}; io.bus_ok = false; check_failure(io, Error::BUS);
    for (int write=1; write<=3; ++write) {
        io = FakeImu{}; io.fail_write = write; check_failure(io, Error::RESUME);
    }
    io = FakeImu{}; io.config_error = Error::CHIP_ID; check_failure(io, Error::CHIP_ID);
    io = FakeImu{}; io.config_error = Error::CONFIG; check_failure(io, Error::CONFIG);
    io = FakeImu{}; io.samples = {{false,0,0,0}}; check_failure(io, Error::READ);
    io = FakeImu{}; io.samples = {{true,0,0,0}}; check_failure(io, Error::ACCEL);
    io = FakeImu{};
    io.samples = {{true,std::numeric_limits<float>::quiet_NaN(),0,1}};
    check_failure(io, Error::ACCEL);
    // A timeout must not poison the next explicit wake attempt.
    io.samples = {{true,0,0,-1}};
    assert(rick::resume_imu(io,q) == Error::NONE);
    assert(std::abs(std::abs(q[1])-1) < 1e-5f);
    std::puts("IMU resume sequence, settling, errors, and retry tests passed");
}
