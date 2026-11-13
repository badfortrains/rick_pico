# First-step Bluetooth diagnostics

Every accepted **Start** now records the most recent 0.5 seconds of idle IMU
samples plus the first three gait cycles. Capture runs at the normal control
rate (currently 50 Hz), with no extra sensor reads or Bluetooth transfers.
At the current cadence this is about 2.2 seconds after Start. Slow cadences are
limited to 4.5 seconds and all captures are bounded to 256 samples.

## Collect a log

1. Install `build/rick-joint-directions.bin` using the controller's existing
   **Firmware update** uploader. This is the production walking firmware,
   including the currently exported `policy_weights.h` and the corrected
   right-then-left servo mapping and negative pulse polarity for the six pitch
   joints. GPIO6–9 drive Rick's right leg; GPIO10–13 drive his left. Each center
   calibration follows its physical servo. Roll polarity is unchanged and
   still needs an individual motion check.
2. Reload the updated controller and reconnect. Deploy `gait-diagnostics.js`
   alongside `bluetooth.html` and the updated `service-worker.js` if hosting the
   controller. The new section is **First-step diagnostics**. If the firmware
   has been updated but support is still missing, forget RickBot in Bluetooth
   settings and pair again to clear the cached GATT services.
3. Put Rick in its normal starting pose and let it stand still for at least
   two seconds, allowing the usual gyro calibration to settle.
4. Press **Start**, observe a few steps, and press **Stop**. An early Stop is
   fine: it preserves an explicitly marked partial capture. Capture itself
   does not automatically stop the robot or change its gait.
5. Choose **Fetch last Start**, then **Save JSON**. CSV is also available.
   JSON includes the settings and calibration needed to interpret the data.
   Fetching requires the policy to be stopped and does not send motion commands.

Capture data survives Stop, reset-to-pose, sleep, and Bluetooth disconnection.
The next accepted Start replaces it, and a firmware reboot loses it. Downloaded
results remain in the page across reconnects, but save them before reloading.
If the link drops mid-download, reconnect and fetch again. Capture IDs and
sample indexes prevent a partial download from mixing different Starts.

## What is recorded

Each sample includes:

- Signed raw accelerometer and gyro counts in the sensor's XYZ axes, from the
  **same 12-byte I2C read** that supplied that controller update.
- Calibrated acceleration in g, gyro before bias correction in rad/s, gyro
  after correction, and the bias actually subtracted on that sample.
- The Madgwick quaternion (wxyz, policy-to-world), projected gravity, estimated
  heading relative to Start, and gait phase used by that policy step.
- Eight normalized policy commands and requested servo pulse widths in µs.
  These are commands; the servos do not report actual joint positions.
- Microseconds relative to Start, zero-based policy step, actual IMU filter
  `dt`, skipped scheduler ticks, cumulative I2C errors, inference time, and
  controller processing time before appending the log.

Negative timestamps are pre-roll. Samples precede motion when `time_us < 0`;
`policy_step=4294967295` means no policy inference occurred on that sample.
For a successful walking sample, sensor acquisition and filter update precede
inference; actions/pulses are the resulting commands. Failed reads are explicit
samples with `read_ok=0`, zero raw counts, and NaN sensor floats (JSON `null`,
CSV empty). Attitude/action fields on such a sample retain the last state;
they are not a new filter or policy update.

Flags are `1=read_ok`, `2=policy_applied`, `4=bias_ready`, `8=raw_clipped`,
`16=pwm_enabled`. A zero PWM flag means recorded pulse widths are the previous
requested widths, not currently active pulses. Clipping tests raw ADC limits
(±32760 counts); it does not establish the physical calibration range.

The metadata records firmware build time, checkpoint step, policy dimensions,
cadence, control interval, heading reference, initial quaternion/actions,
servo centers, accelerometer calibration, gyro scaling, and filter settings.
Header bytes 180-183 identify the mounting calibration (0 for older firmware,
1 for the level-chassis rotation in `imu_mounting.h`). Acceleration and gyro
vectors include that rotation; raw sensor counts do not.
Register configuration bytes describe the configuration verified at IMU setup
or wake, not an additional register read on every sample.

Policy-frame vectors use the model's left/back/up convention and the
current mapping `[-sensor_z, -sensor_y, -sensor_x]`. This differs from the
live UI's right/down/forward display. All pre-roll headings are rebased to the
new Start reference when downloaded. Pre-roll gyro biases remain those that
were actually used at acquisition, even if calibration subsequently changed.

## Interpreting the first capture

The page shows basic statistics; it does not label the IMU healthy or faulty.
For a still, upright starting pose, acceleration magnitude should be close to
1 g, corrected gyro close to zero, and the projected gravity direction stable.
Inspect the initial quaternion and bias to check whether the filter was settled
before the first action. During walking, acceleration includes body motion and
impacts, so deviations from 1 g alone do not indicate a fault.

Check timestamps and failed/clipped samples before interpreting spikes. Compare
raw counts → calibrated values → corrected gyro → attitude/heading, then align
those changes with gait phase and left/right servo commands. This can show
whether an asymmetric command preceded the pivot or followed a reported body
rotation. A video of the same Start helps check the signs of the estimated turn;
the onboard IMU alone cannot establish true yaw or foot contact.

No sensor calibration, axis mapping, attitude-filter settings, policy weights,
or gait timing were changed for this diagnostic capture.

## Transport and tests

GATT characteristic `7e57a00a-4c91-4d8e-8f2a-7dca6d5a1000` supports encrypted
read/write. A write of `[0]` selects the 184-byte metadata response. A write of
`[1, capture_id:u32le, sample_index:u16le]` selects a 188-byte sample. Reads are
idempotent; an invalid index/generation or a still-recording sample selection
returns a 12-byte error header. All responses start with version, packet type
(0 metadata, 1 sample, 2 invalid), capture state, reason, generation:u32le,
index:u16le, and count:u16le. The browser validates every frame and rereads the
header after downloading. Exact field offsets are in `gait_diagnostics.h` and
`gait-diagnostics.js`.

The control task and BLE callbacks share a short critical section around bounded
copies/encoding; neither holds it during I2C, inference, or GATT calls. The buffer
uses about 54 KB of static RAM. The full production build currently uses about
87 KB static RAM and 1.19 MB flash. Control timing reports include log append
overhead; each sample's `control_before_log_us` excludes its own append.

Run the capture, wire-format, download-flow, and decoder checks together:

```sh
node tests/gait_diagnostics_web_test.mjs
```

This compiles and runs the C++ capture tests and checks the actual encoded
packets in JavaScript. It covers bounded capture, chronological pre-roll,
heading rebasing, failed reads, early Stop, all interruption reasons, stale
capture rejection, stopped-only fetching, and retaining downloads on disconnect.
