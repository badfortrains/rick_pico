# RickJoystickFlatTerrain on Waveshare ESP32-S3-Zero

This firmware runs Rick's deterministic Brax PPO policy on a Waveshare
ESP32-S3-Zero. It drives eight hobby servos at 50 Hz, reads a Tokmas
ICM-42688-P over I2C, exposes encrypted Bluetooth controls, and supports
dual-slot BLE firmware updates.

The target is the standard ESP32-S3-Zero with 4 MB flash and 2 MB QSPI PSRAM.
PlatformIO uses Waveshare's recommended `esp32-s3-devkitm-1` board profile with
the flash, PSRAM, USB CDC, and partition layout overridden in `platformio.ini`
and `sdkconfig.defaults`. Arduino 2.0.17 is compiled as an ESP-IDF 4.4.7 component
so Bluetooth modem sleep, power management, and tickless idle are enabled in
the SDK. The precompiled Arduino-only SDK cannot provide this mode.
`firmware/main.cpp` includes the existing root `main.cpp`; the two CMake files
provide the ESP-IDF build entry points.

## Wiring

| Device | Connector/signal | GPIO |
|---|---|---:|
| Right roll servo | J4 / PWM | 6 |
| Right upper servo | J5 / PWM | 7 |
| Right lower servo | J6 / PWM | 8 |
| Right foot servo | J7 / PWM | 9 |
| Left roll servo | J8 / PWM | 10 |
| Left upper servo | J9 / PWM | 11 |
| Left lower servo | J10 / PWM | 12 |
| Left foot servo | J11 / PWM | 13 |
| Tokmas ICM-42688-P | SDA | 2 |
| Tokmas ICM-42688-P | SCL | 4 |
| Tokmas ICM-42688-P | INT1 | 5 |
| Battery sense | J2 pin 4 / BAT_ADC | 1 |

Policy outputs and servo-center calibration use right roll/upper/lower/foot
first (GPIO6–9), then left (GPIO10–13). The right leg is the XML's `Mirror`
chain at negative X: with forward at -Y and up at +Z, the model's +X is Rick's
left. Keep each center calibration attached to its physical servo when
changing the pin mapping. Update the Bluetooth controller together with the
firmware so its calibration labels match this order.

In that order, `SERVO_DIRS` is `{+1, -1, -1, -1, +1, -1, -1, -1}`.
The six pitch signs restore the measured knees-forward default stance after
correcting the leg banks. Roll signs retain their existing values and need
an individual motion check; the default roll targets are zero.

Battery sensing measures the 2S pack through R15 + R16 (200 kOhm total)
over R17 (100 kOhm), with C23 (100 nF) filtering. At a fully charged 8.4 V,
GPIO1 sees 2.8 V. Use `ADC_11db` in the pinned Arduino core (about 3.1 V
input range on ESP32-S3); `ADC_6db` clips and can produce a fixed reading
around 5.5 V after the 3:1 scaling. The existing 250 ms startup delay exceeds
the PCB's 50 ms settling requirement. After flashing, compare the displayed
pack voltage with a multimeter; GPIO1 should measure about one third of it.

The IMU is JLCPCB/LCSC part `C54308212`, uses the 7-bit I2C address `0x36`,
and runs from 3.3 V. It is not register-compatible with the TDK part carrying
the same ICM-42688-P name. Firmware follows the Tokmas quick-start sequence and
checks its `0xA1` chip ID before enabling any servo outputs. INT1 is configured
as an input but the 50 Hz controller currently polls the 110 Hz sensor stream.
The axis transform matches the installed PCB with its component side facing the
robot's left, its servo bank facing forward, S1 above S8, and the power plug at
the back/top. Bluetooth telemetry is right/down/forward; attitude estimation
and policy observations are converted again to MuJoCo's left/back/up frame.

`imu_calibration.h` applies the offset and sensitivity measured on this assembled
Rick's accelerometer before both live telemetry and attitude estimation. At the
normal +/-4 g setting, its effective sensitivity is about 4093 counts/g, with
sensor-frame zero offsets around -7369/-6731/-7105 counts. These values are
specific to this IMU, not general Tokmas defaults. The raw-count test bypasses
this correction. See [the measurement and validation notes](IMU_RAW_TEST.md).

`imu_mounting.h` additionally rotates both acceleration and gyro into the level
chassis frame, before telemetry, attitude estimation, and policy observations.
This fixed mounting correction was measured with the chassis level on
2026-11-12 (about 4.02 degrees pitch and 0.81 degrees roll in policy axes).
It preserves acceleration magnitude and leaves startup gyro-bias calibration
enabled. Repeat the level capture if the PCB mounting changes. Raw-count tests
bypass this rotation as well as the sensor calibration.

GPIO21 is not used because it drives the ESP32-S3-Zero's onboard WS2812. The
USB D-/D+ pins, GPIO19 and GPIO20, are also left free. The eight servos need an
external supply with a common ground; do not power them from the board's 3.3 V
rail or USB regulator.

Verify `SERVO_PINS`, `SERVO_DIRS`, and `SERVO_CENTERS_US` in `main.cpp` before
applying servo power. Test with the robot supported off the ground.

## Build and first flash

Install PlatformIO, then run:

```sh
pio run
```

The flashable application image is:

```text
.pio/build/waveshare_esp32_s3_zero/firmware.bin
```

For the first install, hold the ESP32-S3-Zero **BOOT** button while connecting
USB, release it after the board appears, and run:

```sh
pio run --target upload
pio device monitor
```

The first wired upload installs the bootloader, two-slot partition table, and
application. Later wired uploads can normally use the same command without
holding BOOT; if native USB is unavailable, enter download mode again.

The custom 4 MB partition table provides two 1.94 MB application slots. The
application is approximately 1.2 MB. PlatformIO fails the build if the image grows
beyond one slot.

## Policy and control loop

`policy_weights.h` contains the network parameters and deployment constants
exported with the training run. The controller maintains command history,
projected gravity, body-frame gyro, gait clock, commanded forward speed and
turn rate, and heading error relative to the integrated turn command. Policy inference and servo updates run at the
exported 50 Hz control rate.

The firmware continuously settles its Madgwick attitude estimate while
stopped. Heading-aware policies capture a fresh relative heading when Start is
pressed. Command policies integrate the smoothed turn rate into that reference,
so steering holds the new heading when returned to zero. If gyro-bias calibration is enabled in the exported header, keep Rick
still for the reported calibration interval after boot.

To export a different checkpoint, use the Python environment containing
`numpy`, `jax`, and `orbax-checkpoint`:

```sh
python export_policy.py /path/to/checkpoint/step
pio run
```

The exporter validates layer shapes, observation layout, and float32
serialization before replacing `policy_weights.h`. The generated
`POLICY_DEFAULT_POSE_RAD` comes from the model's `basic` keyframe when that
metadata is available.

## Bluetooth control

The board advertises as **RickBot** using the same service and characteristic
UUIDs as the Pico firmware. Open `bluetooth.html` from an HTTPS host in Chrome
on Android and choose **Pair & connect**. Robot-changing characteristics require
an encrypted bonded connection; telemetry and state reads remain available
during GATT discovery.

The controller provides:

- reset to the exported default pose;
- reset every servo to its raw calibrated center pulse;
- start and stop policy inference;
- forward/reverse speed and left/right steering, plus Straighten and Stand;
- live physical-frame accelerometer and gyro checks;
- buffered first-step IMU/attitude/servo diagnostics, downloadable after Stop;
- live battery-voltage monitoring from GPIO1 / ADC1 through the 3:1 divider;
- live servo-center calibration;
- encrypted wireless firmware updates.

Disconnecting Bluetooth stops the policy and holds the last servo command
when PWM is enabled. Sleeping or released servos stay released.
Servo-center changes are kept only in RAM; copy the generated
`SERVO_CENTERS_US` declaration into `main.cpp` and rebuild to preserve them.

For a first-step drag or pivot, use **First-step diagnostics → Fetch last Start**
after stopping, then **Save JSON**. Each Start automatically records pre-motion
samples and the first three gait cycles. See [capture instructions and field
definitions](GAIT_DIAGNOSTICS.md).

## Training speed and steering

`RickJoystickFlatTerrain` now defaults to recovering straight walking at
**50 mm/s**, then widens commands through explicit stages:
`walk`, `speed`, `transitions`, `turns`, `full`. Select a stage with
`--rick_command_stage` in `learning/train_jax_ppo.py`. Restore the successful
checkpoint from the preceding stage before advancing. Compatible 107-input,
heading-aware walking checkpoints can now warm-start the 108-input policy;
conversion preserves their initial walking output and seeds command normalization.
See [the training guide](../mujoco_playground/learning/RICK_COMMAND_TRAINING.md)
for the recovery command, stage ranges, reward changes, and rollout checks.

The policy observation has 108 inputs: 96 servo-command history values,
3 gravity, 3 gyro, 2 gait clock, forward speed (m/s), yaw rate (rad/s), and 2
heading-error values. Positive speed is forward; positive yaw turns left.
Lateral strafing is not a command.

The `full` stage targets **-40 to +80 mm/s** and **-0.60 to +0.60 rad/s**.
These are training targets, not measured hardware capability. Early stages use
narrower ranges; exported limits include zero for startup. Speed ramps at
50 mm/s² and yaw rate at 1 rad/s², with the same limits exported to firmware.
Walking cadence stays fixed. Speed reward emphasizes cycle-average tracking,
and swing progress requires a lifted foot with support from the other foot.
At zero speed **and** zero turn rate the reward blends to supported standing.
Velocity tracking uses the body heading; heading and yaw penalties follow the
turn command.

After training, export the checkpoint with `export_policy.py`, rebuild with
`pio run`, install the firmware, and reload Rick Control. Speed and steering
ranges are read from the firmware's exported policy limits. **Start** ramps
from rest to the selected setpoint. Slider changes apply during walking;
**Straighten** zeros steering, and **Stand** zeros both commands while keeping
the policy active. **Stop** immediately stops inference, holds the current
servo pose, and clears the selected command. Bluetooth disconnection also
stops inference. Legacy walking policies report command version zero and keep
the new controls disabled; command policies enable controls within their
exported stage limits.

The control characteristic accepts opcode `6`, followed by signed little-endian
int16 forward speed in mm/s and yaw rate in mrad/s (5 bytes total). Out-of-range
packets are rejected. Its read status preserves the first three legacy bytes
(state, sleep capability, wake error), followed by command version at byte 3;
version 1 appends int16 speed min/max, yaw min/max, selected speed, and selected
yaw at offsets 4, 6, 8, 10, 12, and 14. Limits must fit these integer units.
Legacy policies report command version zero and continue using their trained
fixed speed.

## Bluetooth standby

With Rick laid down or supported, press **Sleep**. The firmware stops policy
inference, cancels raw capture, drives all eight servo signals low, pauses their
PWM timer, powers down the Tokmas IMU, turns off the RGB LED, and ends USB serial.
The web controller waits for confirmation, stops telemetry polling, and
disconnects. The servo supply remains on with this PCB; motor release and total
battery current must be checked with your actual servos.

RickBot then advertises connectably every 1–2 seconds. The control task blocks
on its command queue while BLE modem sleep and automatic light sleep handle
radio activity. Discovery may take several seconds. Reconnect and press
**Wake** to send the encrypted command over the existing bonded connection.
Connecting, disconnecting, Stop, calibration, OTA, and ordinary motion commands
cannot wake it. A client that stays connected is offered longer connection
intervals and peripheral latency; the phone decides whether to accept them.
For the lowest consumption, disconnect while asleep as the web controller does.

Wake restores USB serial, restarts the I2C driver, and uses the Tokmas resume
sequence without repeating the cold-start reset. It verifies the chip ID and
configuration, waits for three consecutive usable accelerometer samples (up to
two seconds of sample polling), resets tilt from their average, then restores
the LED and attitude estimation. Hold Rick
still during wake. **Servo pulses remain off** until an explicit pose,
calibration, or Start command. Motion Start is inhibited for one second after
wake. An IMU failure leaves the robot asleep with outputs off and a sensor-fault
state; retry Wake after checking the sensor. An IMU power-down failure also
reports a fault and may consume more power than normal standby.

Sleep is a RAM state, not a persistent boot preference: reset or power cycling
still performs the existing default-pose startup. USB serial is unavailable
while asleep; wake over BLE or use BOOT plus reset for wired recovery. Measure
standby current at the battery with USB unplugged, after the BLE client has
disconnected. Sleep current and battery lifetime have not been measured.

The control characteristic remains compatible with older clients: its first
read byte is the state; a new second byte is a capability bitmask (bit 0 means
Sleep/Wake support). Commands 0–3 retain their meanings; **4 = Sleep**, **5 =
Wake**. New states are **5 = Sleeping**, **6 = Awake with PWM off**, and **7 =
Sleeping with sensor fault**, and **8 = Waking**. A third read byte reports the
last sleep/wake error: 0 = none, 1 = power-down write, 2 = I2C restart, 3 = resume
sequence, 4 = chip ID, 5 = configuration, 6 = no readable samples, 7 = unusable
accelerometer samples. The page accepts old one/two-byte responses and displays
the detailed reason when provided. A timeout is shown separately from a reported
sensor fault. All writes retain encrypted GATT permissions.
The new UI disables Sleep/Wake when talking to older firmware. IMU readings are
zeroed during sleep; battery reads return the last value sampled on sleep entry.
Wake before starting an OTA update. If the installed firmware cannot wake,
support Rick and power-cycle it to return to normal startup, then update over
Bluetooth before sleeping again. Reset re-enables the default servo pose.

On a fresh checkout, `pio run` downloads ESP-IDF and builds its libraries once.
If changing SDK options later, remove the generated `sdkconfig.<environment>`
file before rebuilding so `sdkconfig.defaults` is reapplied. Generated SDK
configurations are ignored by Git. Do not substitute `-DCONFIG_PM_ENABLE` for
rebuilding the SDK; defining the macro alone does not enable power management.

Hardware acceptance checks (support the robot throughout):

1. Verify normal boot, IMU readings, motion control, calibration, and OTA.
2. Sleep and verify the LED goes dark, USB disconnects, and all servo signals
   stay low. Confirm the servos release instead of holding torque.
3. Reconnect after at least a minute; verify it still reports Sleeping, then
   disconnect again. Repeat after leaving it asleep overnight.
4. Wake and verify live IMU readings return while servo pulses remain off;
   choose a pose explicitly before testing Start and Stop.
5. Repeat several sleep/wake cycles, including waking from a different resting
   orientation. Test failed sensor communication with servo power disconnected.
6. Measure battery current both asleep/disconnected and asleep/connected.

## Wireless firmware updates

After one wired install, select
`.pio/build/waveshare_esp32_s3_zero/firmware.bin` in Rick Control. Protocol 3
streams the raw ESP32 image to the inactive OTA slot, checks the full SHA-256
digest, asks ESP-IDF to validate the image, and only then marks that slot for
boot and restarts.

An interrupted or rejected transfer does not change the active boot partition.
The stock Arduino framework used here does not enable ESP-IDF's runtime
rollback watchdog, so an image that passes structural validation but fails only
after boot must be recovered over USB. Keep a known-good wired build available.

Do not select the old `rick_v2_pico.uf2`: RP2350 UF2 files are intentionally
rejected by the ESP32 protocol. The web controller retains its older UF2 parser
only so it can still service robots running the prior Pico firmware.

## Verification

For accelerometer offsets or suspicious sensitivity, use the
[IMU raw-count test](IMU_RAW_TEST.md) in Rick Control. The normal firmware
captures six poses at +/-2, +/-4, and +/-8 g over Bluetooth, then restores
the normal sensor range. A separate USB-only diagnostic build is also available.

Build the target and run the host regressions:

```sh
pio run
c++ -std=c++17 -O2 -I. tests/control_test.cpp -o /tmp/rick-control-test
/tmp/rick-control-test
python -m unittest tests/export_policy_test.py
node tests/uf2_test.mjs .pio/build/waveshare_esp32_s3_zero/firmware.bin
node tests/sleep_web_test.mjs
c++ -std=c++17 -O2 -I. tests/sleep_attitude_test.cpp -o /tmp/rick-sleep-attitude-test
/tmp/rick-sleep-attitude-test
c++ -std=c++17 -O2 -I. tests/imu_wake_test.cpp -o /tmp/rick-imu-wake-test
/tmp/rick-imu-wake-test
```

On hardware, verify in this order with the robot supported:

1. USB serial reports the ESP32-S3 startup and BLE advertising messages.
2. The IMU axis check reports +X right, +Y down, and +Z forward.
3. Each servo calibration control moves only its named joint.
4. Stop and Bluetooth disconnect both hold the current pose.
5. Policy inference remains near 50 Hz without timing overruns.
6. A BLE update reboots into the other OTA slot and advertises again.

## PWA hosting

Serve `bluetooth.html`, `manifest.webmanifest`, `service-worker.js`, and the
`icons` directory together from an HTTPS static host. The cached app shell can
open offline, but Bluetooth must still be enabled on the Android device.

Command-control regression checks:

```sh
node tests/motion_controls_web_test.mjs
c++ -std=c++17 -Wall -Wextra -Werror -I. tests/motion_control_test.cpp -o /tmp/rick-motion-test
/tmp/rick-motion-test
# In mujoco_playground, using its Python environment:
python -m unittest mujoco_playground._src.locomotion.rick.motion_command_test
```
