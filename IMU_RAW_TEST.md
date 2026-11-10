# IMU raw-count test

## Calibration measured on this board

The production firmware applies `imu_calibration.h` before converting sensor
acceleration into the robot and policy frames. A sphere fit to the first 18
stationary measurements (six poses, three ranges) gives a nominal-g center of
`[-0.89949914, -0.82169417, -0.86726939]` and a radius of `0.49958199`.
This allows the poses to be slightly tilted rather than treating every hold
as perfectly aligned with an axis.

At the normal +/-4 g range, the correction is:

```text
sensor_x_g = (raw_x + 7368.697) / 4092.576
sensor_y_g = (raw_y + 6731.319) / 4092.576
sensor_z_g = (raw_z + 7104.671) / 4092.576
```

The first set has corrected magnitudes of 0.9900-1.0084 g. A separate repeat of
upright, upside-down, and both side poses (12 more samples) has magnitudes of
1.0019-1.0142 g using the original fit. The repeat upright +/-4 g row predicts
robot acceleration `[-0.0067, -1.0003, +0.0639]` g. This validates the correction
on a second set of holds, but does not establish long-term or temperature
stability. The root cause of the large offset and half sensitivity remains
unconfirmed. Recalibrate if this sensor is replaced.

The repeat's left/right labels were from the observer facing Rick. Thus
`left_down` in that repeat means Rick's right side down, and `right_down` means
Rick's left side down. The existing axis mapping is consistent with that
clarification. New captures should use **Rick's own left and right** as labeled
on the updated page.

Raw diagnostic captures remain uncorrected so they can detect later changes.

## Bluetooth and web page (no USB needed)

The normal `waveshare_esp32_s3_zero` firmware now includes the raw-count test
and retains Bluetooth control and OTA updates.

1. Reload the local `bluetooth.html` in Chrome and connect to RickBot.
2. Use **Firmware update** to upload
   `.pio/build/waveshare_esp32_s3_zero/firmware.bin` (or the identical
   `build/rick-imu-bluetooth.bin` convenience copy). Reconnect after reboot.
   **Do not upload `.pio/build/imu_raw_test/firmware.bin` over Bluetooth:** that
   USB-only diagnostic build has no Bluetooth service.
3. Find **IMU raw-count test** below the live IMU values. Select a pose,
   support Rick in that orientation, and press **Capture pose**. Turn the
   entire body/PCB, not just the legs. Hold still for about four seconds.
4. Repeat for all six poses. The next unrecorded pose is selected automatically;
   recapturing a pose replaces its previous three rows. Press **Copy results**
   and paste the CSV into the chat. Results survive a Bluetooth disconnect, but
   copy them before reloading or closing the page.

The test stops the policy and holds the current servo pose. It samples each of
the three ranges without updating attitude or gyro calibration from those test
samples. It restores and verifies the normal +/-4 g configuration on completion,
cancel, Stop, disconnect, or the start of an OTA update. The policy stays stopped.
If restoration fails, motion commands are blocked until Rick restarts; Bluetooth
and OTA remain available. A failed or interrupted capture does not replace a
previous successful result.

If the test remains unavailable after updating and reconnecting, the computer
may have cached the old Bluetooth services: forget RickBot in Bluetooth settings
and pair again. The updated page also works with old firmware for uploading.

## USB-only alternative

The separate `imu_raw_test` build uses the same Tokmas initialization as the normal
firmware. It holds all servo signal pins low and does not start Bluetooth or
the policy. It reports signed, averaged ADC counts before calibration or g
conversion, along with the actual range-register readback.

### Run over USB

1. Disconnect the battery and remove JP1 before connecting USB, as required by
   the carrier board. Support the robot by hand or on a stable surface.
2. From this directory, upload and open the USB serial monitor:

   ```sh
   ~/.platformio/penv/bin/pio run -e imu_raw_test -t upload
   ~/.platformio/penv/bin/pio device monitor -e imu_raw_test
   ```

   If PlatformIO cannot choose the port, use `pio device list`, then append
   `--upload-port /dev/cu.usbmodem...` to the upload command or
   `--port /dev/cu.usbmodem...` to the monitor command, using the actual device.
   If necessary, enter download mode with BOOT while connecting USB, then
   release BOOT. Exit the monitor with Ctrl+C.
3. Type `?` to display instructions if you missed the boot messages. Commands
   are single lowercase letters; Enter is optional.
4. Turn the **whole body/PCB**, not just the legs, into each pose below. Hold it
   still, send the letter, and keep holding until `END` appears (about four
   seconds). Each command measures all three ranges automatically.

   | Command | Pose | Main robot-count axis |
   | --- | --- | --- |
   | `u` | Upright, feet down | Y negative |
   | `d` | Upside-down, feet up | Y positive |
   | `l` | Robot's left side facing down | X positive |
   | `r` | Robot's right side facing down | X negative |
   | `f` | Front facing down | Z negative |
   | `b` | Back facing down | Z positive |

   Align the body axes as closely as possible to vertical/horizontal. These
   signs assume the installed PCB orientation documented in README.md.
5. Save the full serial output: six poses produce 18 measurement rows. For a
   cold-start comparison, unplug USB with the battery still disconnected,
   reconnect, reopen the monitor, and repeat.

## Interpret

`sensor_x/y/z` are averages of the actual signed sensor counts. `robot_x/y/z`
only reorder/change signs into right/down/forward coordinates, retaining count
units. `span_x/y/z` are the sensor-axis maximum minus minimum over 50 samples
spaced 20 ms apart; large spans can indicate movement or noise. `clipped=1`
flags samples near an int16 limit; those measurements are unsuitable for a
linear calibration. Do not treat a register/read failure as a valid capture.

For each range, subtract opposite-pose means on the corresponding robot axis:

- X: `left-down robot_x - right-down robot_x`
- Y: `upside-down robot_y - upright robot_y`
- Z: `back-down robot_z - front-down robot_z`

| Range | Register 0x21 | Expected counts per g | Expected opposite-pose difference |
| --- | --- | --- | --- |
| +/-2 g | 0x03 | 16384 | 32768 |
| +/-4 g | 0x13 | 8192 | 16384 |
| +/-8 g | 0x23 | 4096 | 8192 |

Compute differences in a wider integer or floating-point type, not int16.
The midpoint `(positive + negative) / 2` estimates the zero-g offset in
counts. Half the difference estimates the measured counts per g, assuming
accurate opposite poses. Bias cancels in the difference.

If differences are consistently half these values, the actual sensitivity
does not match the selected range's documented conversion. If differences do
not halve as the range doubles, investigate range configuration. If differences
match but midpoints are large, the remaining problem is primarily offset.
A sign reversal points to orientation/sign mapping. This test identifies the
behavior; it cannot alone distinguish a chip defect from incorrect vendor
documentation or initialization requirements.

## Restore normal firmware after the USB-only alternative

The diagnostic build has no Bluetooth or OTA service. Restore over USB:

```sh
~/.platformio/penv/bin/pio run -e waveshare_esp32_s3_zero -t upload
```

Do not use the diagnostic image as normal robot firmware. The normal build
remains the default for `pio run`.
