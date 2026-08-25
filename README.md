# RickJoystickFlatTerrain on Raspberry Pi Pico 2

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

## Build

The project targets `pico2` in `CMakeLists.txt`. Build and flash it through the
Raspberry Pi Pico VS Code extension as before. On boot, the firmware centers
all eight servos for ten seconds before starting the 50 Hz controller.
