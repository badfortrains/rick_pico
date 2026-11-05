# RickJoystickFlatTerrain on Waveshare ESP32-S3-Zero

This firmware runs Rick's deterministic Brax PPO policy on a Waveshare
ESP32-S3-Zero. It drives eight hobby servos at 50 Hz, reads a Tokmas
ICM-42688-P over I2C, exposes encrypted Bluetooth controls, and supports
dual-slot BLE firmware updates.

The target is the standard ESP32-S3-Zero with 4 MB flash and 2 MB QSPI PSRAM.
PlatformIO uses Waveshare's recommended `esp32-s3-devkitm-1` board profile with
the flash, PSRAM, USB CDC, and partition layout overridden in `platformio.ini`.

## Wiring

| Device | Connector/signal | GPIO |
|---|---|---:|
| Left roll servo | J4 / PWM | 6 |
| Left upper servo | J5 / PWM | 7 |
| Left lower servo | J6 / PWM | 8 |
| Left foot servo | J7 / PWM | 9 |
| Right roll servo | J8 / PWM | 10 |
| Right upper servo | J9 / PWM | 11 |
| Right lower servo | J10 / PWM | 12 |
| Right foot servo | J11 / PWM | 13 |
| Tokmas ICM-42688-P | SDA | 2 |
| Tokmas ICM-42688-P | SCL | 4 |
| Tokmas ICM-42688-P | INT1 | 5 |

The IMU is JLCPCB/LCSC part `C54308212`, uses the 7-bit I2C address `0x36`,
and runs from 3.3 V. It is not register-compatible with the TDK part carrying
the same ICM-42688-P name. Firmware follows the Tokmas quick-start sequence and
checks its `0xA1` chip ID before enabling any servo outputs. INT1 is configured
as an input but the 50 Hz controller currently polls the 110 Hz sensor stream.
The sensor-to-robot axis transform is unchanged; verify it with the controller's
IMU axis check because the assembled package orientation cannot be inferred
from the connector pinout alone.

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
current build is about 1.20 MB. PlatformIO fails the build if the image grows
beyond one slot.

## Policy and control loop

`policy_weights.h` contains the network parameters and deployment constants
exported with the training run. The controller maintains command history,
projected gravity, body-frame gyro, gait clock, target velocity, and optional
relative-heading observations. Policy inference and servo updates run at the
exported 50 Hz control rate.

The firmware continuously settles its Madgwick attitude estimate while
stopped. Heading-aware policies capture a fresh relative heading when Start is
pressed. If gyro-bias calibration is enabled in the exported header, keep Rick
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
- reset all normalized policy actions to `+1.0`;
- start and stop policy inference;
- live physical-frame accelerometer and gyro checks;
- live servo-center calibration;
- encrypted wireless firmware updates.

Disconnecting Bluetooth stops the policy and holds the last servo command.
Servo-center changes are kept only in RAM; copy the generated
`SERVO_CENTERS_US` declaration into `main.cpp` and rebuild to preserve them.

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

Build the target and run the host regressions:

```sh
pio run
c++ -std=c++17 -O2 -I. tests/control_test.cpp -o /tmp/rick-control-test
/tmp/rick-control-test
python -m unittest tests/export_policy_test.py
node tests/uf2_test.mjs .pio/build/waveshare_esp32_s3_zero/firmware.bin
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
