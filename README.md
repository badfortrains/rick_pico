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
receives a Start command. Robot-changing commands use encrypted LE Secure
Connections with Just Works confirmation and store the Android bond in flash.
State and telemetry reads do not require pairing, so Android can finish GATT
discovery before it prompts for security on the first command.

Copy `bluetooth.html` to the Android device and open it in Chrome, then:

1. Tap **Pair & connect** and choose **RickBot**.
2. Tap a control and approve the Android pairing request if one appears.
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
contexts. The controller can still be opened directly for development, but its
install and offline features require `bluetooth.html`, `manifest.webmanifest`,
`service-worker.js`, and the `icons` directory to be served together from an
HTTPS static host.

The controller is also an installable Progressive Web App when served over
HTTPS. Open it in Chrome on Android, then use the in-page **Install app** button
when it appears, or choose **Add to Home screen** from Chrome's menu. The app
shell is cached so the controller can open without an internet connection;
Bluetooth still needs to be enabled on the phone.

## Wireless firmware updates

The Pico 2 W firmware and Rick Control page support encrypted BLE firmware
updates. The resident RP2350 partition table occupies the first 8 KiB of
flash, followed by two 2036 KiB application slots. An update is always written
to the inactive slot, so disconnecting Bluetooth or losing power during an
upload leaves the currently running firmware intact. The final 16 KiB of flash
are kept outside both slots for BTstack's bond store and the RP2350-E10
reserved sector.

The build is hashed and marked **Try Before You Buy**. After an update, the
RP2350 boot ROM starts the new slot under its rollback watchdog. The firmware
accepts the new slot only after the robot hardware and Bluetooth stack have
initialized; otherwise the boot ROM returns to the previous slot.

### One-time wired setup

The partition table and application are deliberately separate. This matches
the RP2350 A/B update flow and keeps the later wireless-update UF2 free of
partition-table metadata.

For a Pico 2 W with empty/unpartitioned flash:

1. Build the project, enter BOOTSEL mode, and copy
   `build/ota_partition.uf2` to the Pico.
2. Let it reboot, then create a one-time bootstrap build with
   `-DRICK_BOOTSTRAP_IMAGE=ON`. Enter BOOTSEL a second time and load that
   build's `rick_v2_pico.uf2` with Picotool's `-x` option. The bootstrap image
   is not TBYB because neither slot contains a bought fallback yet.
3. Build normally afterward. The regular `build/rick_v2_pico.uf2` remains
   hashed and TBYB-enabled for all later Bluetooth updates.

Installing `ota_partition.uf2` is a one-time operation. Do not send that file
through Rick Control. The earlier experimental OTA-enabled build embedded the
table in replaceable slot A; that is not equivalent to installing the resident
table, so a Pico flashed with that build must also complete both wired steps
above once. Keep the robot supported and keep servo power safely isolated
during the first application boot.

### Later updates over Bluetooth

1. Rebuild the project to create a new `build/rick_v2_pico.uf2`.
2. Open the HTTPS-hosted Rick Control page in Chrome on Android and connect to
   RickBot.
3. In **Firmware update**, choose `rick_v2_pico.uf2` and tap **Install
   firmware**.
4. Keep the Pico powered while the page uploads and verifies the image. RickBot
   disconnects when it reboots into the verified slot. Reconnect after a few
   seconds; the page then reports whether the new slot was confirmed or the
   boot ROM rolled back to the previous slot.

The page accepts only complete RP2350 RISC-V Rick firmware UF2s, hashes the
transfer before it starts, and shows both upload and device-written progress.
The firmware independently validates every UF2 block, its target range and
order, the full SHA-256 transfer digest, and the programmed slot with the
RP2350 boot ROM before enabling the reboot. The update GATT characteristics
require the same encrypted BLE connection as robot-changing commands. USB
BOOTSEL remains the recovery path if both firmware slots are ever damaged.

## Build

The project targets `pico2_w` in `CMakeLists.txt` and requires Pico SDK 2.3.0
and Picotool 2.3.0 for the RP2350 A/B update flow. Build and flash it through
the Raspberry Pi Pico VS Code extension as before. Ensure the SDK's `btstack`
and `cyw43-driver` submodules are installed; the Pico extension normally
manages these.
