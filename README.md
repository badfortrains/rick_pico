# RickJoystickFlatTerrain on Raspberry Pi Pico 2 W

This firmware runs the deterministic policy from the Brax PPO checkpoint. The
generated `policy_weights.h` contains both the network parameters and the
deployment constants saved with the training run.

## Export a checkpoint

Run the exporter from a Python environment containing `numpy`, `jax`, and
`orbax-checkpoint` (the `mujoco_playground` environment already includes them):

```sh
python export_policy.py \
  /Users/suzanna/Documents/code/checkpoints/RickJoystickFlatTerrain-20260818-165853/checkpoints/000068812800
```

By default this replaces `policy_weights.h` beside the script. The exporter
checks the PPO configuration, all layer shapes, the 41-value Rick observation
layout, and float32 serialization before writing the header.

The deployed observation is:

- indices 0--31: four previous 8-servo commands, oldest first
- indices 32--34: body-frame projected gravity
- indices 35--37: body-frame gyro in rad/s, scaled by 0.25
- indices 38--39: sine/cosine of the 0.8 Hz gait clock
- index 40: target forward velocity (0.06 m/s)

The network is `41 -> 128 -> 128 -> 128 -> 128 -> 16` with SiLU hidden
activations. Deterministic actions are `tanh` of the first eight outputs. The
current checkpoint contains 56,976 float32 policy parameters (about 223 KiB in
flash, before code).

## Servo wiring and calibration

The policy/XML actuator order and default GPIO mapping are:

| Action | Joint | GPIO |
|---:|---|---:|
| 0 | left roll | 12 |
| 1 | left upper | 4 |
| 2 | left lower | 3 |
| 3 | left foot | 2 |
| 4 | right roll | 13 |
| 5 | right upper | 7 |
| 6 | right lower | 5 |
| 7 | right foot | 6 |

GPIO 12 and 13 are new defaults for the two actuators absent from the previous
six-servo firmware. Update `SERVO_PINS`, `SERVO_DIRS`, and
`SERVO_CENTERS_US` in `main.cpp` to match the physical robot. The six existing
pin assignments and the calibrated 1460/1440 us foot centers were retained at
their corresponding shifted indices.

Test with the robot supported off the ground first. The policy was trained
with `action_scale = 0.35` radians, so the firmware converts a policy action of
`+/-1` to only about `+/-223 us` around each calibrated center. Servos need an
external power supply with a common ground; do not power eight servos from the
Pico.

## Bluetooth control

The Pico advertises a Bluetooth Low Energy device named **RickBot**. On boot,
the firmware moves to the calibrated default pose and remains stopped until it
receives a Start command. Pairing uses encrypted LE Secure Connections with
Just Works confirmation and stores the Android bond in flash.

Copy `bluetooth.html` to the Android device and open it in Chrome, then:

1. Tap **Pair & connect** and choose **RickBot**.
2. Approve the Android pairing request if one appears.
3. Use one of the four controls:
   - **Reset to default pos** stops the policy and moves all servos to their
     calibrated centers.
   - **Reset to full action** stops the policy and sends normalized policy
     action `+1.0` to every servo (the full trained action range around each
     calibrated center, with `SERVO_DIRS` applied).
   - **Start** resets the gait clock and IMU orientation estimate, then runs the
     PPO policy at 50 Hz.
   - **Stop** stops policy inference and holds the last commanded servo pose.

The same page includes two calibration tools:

- **IMU axis check** shows live accelerometer and gyroscope readings in the
  physical sensor frame. To verify a direction, point the named positive axis
  straight up, hold the robot still, and tap its check button. A passing axis
  reads about `+1 g` with the other two axes near zero. Complete all three to
  confirm `+X` right, `+Y` down, and `+Z` forward.
- **Servo centers** adjusts any of the eight neutral pulse widths in 1, 5, or
  10 us steps. The first adjustment stops the policy and commands every joint
  to its center. Mark each joint after mechanically centering it, then tap
  **Finish & print values**. The page displays a complete
  `SERVO_CENTERS_US` declaration and the firmware prints the same declaration
  over USB serial.

Servo calibration changes live RAM only. Copy the generated declaration into
`main.cpp` and rebuild/flash the firmware to preserve it across a reboot.

Disconnecting Bluetooth also stops the policy and holds the last pose. Always
test and calibrate with the robot supported off the ground first.

Web Bluetooth is supported by Chrome on Android and is restricted to secure
contexts. The controller is a completely self-contained file; if Chrome does
not expose Bluetooth when it is opened from local storage, serve the unchanged
file from an HTTPS static host.

## Build

The project now targets `pico2_w` in `CMakeLists.txt` and uses the Pico SDK
BTstack/CYW43 libraries. Build and flash it through the Raspberry Pi Pico VS
Code extension as before. Ensure the SDK's `btstack` and `cyw43-driver`
submodules are installed; the Pico extension normally manages these.
