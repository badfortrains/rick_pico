#include <cassert>
#include <cmath>
#include <limits>
#include <random>
#include "heading_control.h"

int main() {
    auto check = [](float ax, float ay, float az) {
        float q[4];
        assert(rick::attitude_from_accel(ax, ay, az, q));
        const float norm = std::sqrt(ax*ax + ay*ay + az*az);
        const float w=q[0], x=q[1], y=q[2], z=q[3];
        // Expected body-frame gravity must match the new resting orientation,
        // including side-lying and upside-down wakeups, regardless of old pose.
        assert(std::abs(2*(w*y-x*z) + ax/norm) < 1e-5f);
        assert(std::abs(-2*(w*x+y*z) + ay/norm) < 1e-5f);
        assert(std::abs(-(w*w-x*x-y*y+z*z) + az/norm) < 1e-5f);
        assert(std::abs(w*w+x*x+y*y+z*z-1) < 1e-5f);
    };
    for (int axis=0; axis<3; ++axis) for (float sign : {-1.f, 1.f}) {
        float a[3]={}; a[axis]=sign; check(a[0],a[1],a[2]);
    }
    std::mt19937 gen(42);
    std::normal_distribution<float> random(0,1);
    for (int i=0; i<1000; ++i) {
        float x=random(gen), y=random(gen), z=random(gen);
        float n=std::sqrt(x*x+y*y+z*z); check(x/n,y/n,z/n);
    }
    float q[4]={1,2,3,4};
    assert(!rick::attitude_from_accel(0,0,0,q));
    assert(!rick::attitude_from_accel(0,0,3,q));
    assert(!rick::attitude_from_accel(std::numeric_limits<float>::quiet_NaN(),0,1,q));
    assert(q[0]==1 && q[1]==2 && q[2]==3 && q[3]==4);
}
