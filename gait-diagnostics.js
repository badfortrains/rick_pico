"use strict";

// Shared by the browser controller and the cross-language protocol tests.
globalThis.RickGaitDiagnostics = (() => {
  const UUID = "7e57a00a-4c91-4d8e-8f2a-7dca6d5a1000";
  const floats = (v, offset, count) => Array.from({length: count}, (_, i) => v.getFloat32(offset + 4*i, true));
  const integers = (v, offset, count, signed = false) => Array.from({length: count}, (_, i) =>
    signed ? v.getInt16(offset + 2*i, true) : v.getUint16(offset + 2*i, true));
  function common(v, type, length) {
    if (v.byteLength !== length || v.getUint8(0) !== 1 || v.getUint8(1) !== type)
      throw new Error("Incomplete, stale, or unsupported gait log. Fetch again after Stop.");
    return {capture_id: v.getUint32(4, true), state: v.getUint8(2), reason: v.getUint8(3),
      index: v.getUint16(8, true), count: v.getUint16(10, true)};
  }
  function decodeHeader(v) {
    const h = common(v, 0, 184);
    Object.assign(h, {
      pre_roll_count: v.getUint16(12, true), capacity: v.getUint16(14, true),
      checkpoint_step: v.getUint32(16, true), control_dt_s: v.getFloat32(20, true),
      frequency_hz: v.getFloat32(24, true), target_velocity_m_s: v.getFloat32(28, true),
      action_scale_rad: v.getFloat32(32, true), heading_reference_rad: v.getFloat32(36, true),
      gyro_rad_s_per_count: v.getFloat32(40, true), madgwick_beta: v.getFloat32(44, true),
      start_gyro_bias_rad_s: floats(v, 48, 3), initial_action: floats(v, 60, 8),
      servo_centers_us: integers(v, 92, 8), observation_size: v.getUint16(108, true),
      action_size: v.getUint16(110, true), requested_duration_us: v.getUint32(112, true),
      heading_enabled: Boolean(v.getUint8(116)), calibration_enabled: Boolean(v.getUint8(117)),
      accel_config: v.getUint8(118), gyro_config: v.getUint8(119),
      accel_zero_sensor_counts: floats(v, 120, 3), accel_counts_per_g: v.getFloat32(132, true),
      gyro_observation_scale: v.getFloat32(136, true), initial_quaternion_wxyz: floats(v, 140, 4),
      mounting_calibration_id: v.getUint32(180, true),
      firmware_build: Array.from({length: 24}, (_, i) => v.getUint8(156 + i))
        .map(c => String.fromCharCode(c)).join("").split("\0")[0],
    });
    if (h.state > 4 || h.reason > 7 || h.capacity !== 256 || h.count > h.capacity ||
        h.pre_roll_count > h.count || h.action_size !== 8 || h.index !== 65535)
      throw new Error("Invalid gait log header.");
    return h;
  }
  function decodeFrame(v, header, index) {
    const f = common(v, 1, 188);
    if (f.capture_id !== header.capture_id || f.count !== header.count || f.index !== index ||
        f.state !== header.state || f.reason !== header.reason)
      throw new Error("The gait log changed during download. Stop and fetch again.");
    return {
      index, time_us: v.getInt32(12, true), policy_step: v.getUint32(16, true),
      flags: v.getUint32(20, true), imu_dt_s: v.getFloat32(24, true),
      skipped_ticks: v.getUint32(28, true), total_read_errors: v.getUint32(32, true),
      inference_us: v.getUint32(36, true), control_before_log_us: v.getUint32(40, true),
      raw_sensor_counts: integers(v, 44, 6, true), accel_policy_g: floats(v, 56, 3),
      gyro_before_bias_rad_s: floats(v, 68, 3), gyro_policy_rad_s: floats(v, 80, 3),
      quaternion_wxyz: floats(v, 92, 4), gravity_policy: floats(v, 108, 3),
      gait_phase_rad: v.getFloat32(120, true), heading_relative_rad: v.getFloat32(124, true),
      action: floats(v, 128, 8), servo_pulse_us: integers(v, 160, 8),
      gyro_bias_rad_s: floats(v, 176, 3),
    };
  }
  async function fetchCapture(exchange, progress = () => {}) {
    const header = decodeHeader(await exchange(Uint8Array.of(0)));
    if (!header.capture_id || header.state === 0) throw new Error("No Start has been recorded yet.");
    if (header.state === 1) throw new Error("Still recording. Press Stop, then fetch the log.");
    const frames = [];
    for (let index = 0; index < header.count; ++index) {
      const packet = new Uint8Array(7), view = new DataView(packet.buffer);
      packet[0] = 1; view.setUint32(1, header.capture_id, true); view.setUint16(5, index, true);
      frames.push(decodeFrame(await exchange(packet), header, index));
      if (index && frames[index].time_us <= frames[index - 1].time_us)
        throw new Error("Gait log timestamps are out of order.");
      progress(index + 1, header.count);
    }
    const end = decodeHeader(await exchange(Uint8Array.of(0)));
    if (end.capture_id !== header.capture_id || end.count !== header.count || end.state !== header.state)
      throw new Error("A new Start replaced the gait log during download. Fetch again.");
    return {format: "rick-start-gait-v1", downloaded_at: new Date().toISOString(),
      axes: {raw: "sensor XYZ; acceleration then gyro", policy: "+X left, +Y back, +Z up",
        sensor_to_policy: header.mounting_calibration_id
          ? `nominal sensor-to-robot mapping, mounting calibration ${header.mounting_calibration_id}, then robot-to-policy; see imu_mounting.h for rotation`
          : "[-sensor_z, -sensor_y, -sensor_x]", quaternion: "policy-to-world wxyz"},
      flags: {read_ok: 1, policy_applied: 2, bias_ready: 4, raw_clipped: 8, pwm_enabled: 16},
      notes: "Negative times precede START. Sensor failures have read_ok=0 and null sensor floats in JSON. Servo pulses are commands, not measured positions. Pre-roll heading uses the new START reference.",
      header, frames};
  }
  function captureCsv(capture) {
    const axes = ["x", "y", "z"];
    const vectors = [
      ["raw_sensor_counts", ["ax", "ay", "az", "gx", "gy", "gz"]],
      ["accel_policy_g", axes], ["gyro_before_bias_rad_s", axes], ["gyro_policy_rad_s", axes],
      ["quaternion_wxyz", ["w", "x", "y", "z"]], ["gravity_policy", axes],
      ["action", Array.from({length: 8}, (_, i) => i)],
      ["servo_pulse_us", Array.from({length: 8}, (_, i) => i)], ["gyro_bias_rad_s", axes],
    ];
    const scalar = ["index", "time_us", "policy_step", "flags", "imu_dt_s", "skipped_ticks",
      "total_read_errors", "inference_us", "control_before_log_us", "gait_phase_rad", "heading_relative_rad"];
    const columns = [...scalar, ...vectors.flatMap(([name, labels]) => labels.map(axis => `${name}_${axis}`))];
    const rows = capture.frames.map(f => [...scalar.map(k => f[k]), ...vectors.flatMap(([name]) => f[name])]
      .map(x => Number.isFinite(x) ? String(x) : "").join(","));
    return [columns.join(","), ...rows].join("\n");
  }
  function summarize(capture) {
    const good = capture.frames.filter(f => f.flags & 1);
    const pre = good.filter(f => f.time_us < 0);
    const walking = good.filter(f => f.flags & 2);
    const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : NaN;
    const format = (v, digits = 3) => Number.isFinite(v) ? v.toFixed(digits) : "n/a";
    const peakHeading = walking.length ? Math.max(...walking.map(f => Math.abs(f.heading_relative_rad))) * 180 / Math.PI : NaN;
    const last = walking.at(-1);
    const state = ["empty", "recording", "complete", "stopped early", "buffer full"][capture.header.state];
    return `${capture.frames.length} samples (${pre.length} valid before Start); ${state}.\n` +
      `Before Start: mean |accel| ${format(mean(pre.map(f => Math.hypot(...f.accel_policy_g))))} g; ` +
      `mean corrected gyro XYZ [${[0,1,2].map(i => format(mean(pre.map(f => f.gyro_policy_rad_s[i])))).join(", ")}] rad/s.\n` +
      `Walking: peak |heading error| ${format(peakHeading, 1)}°; last heading ${format(last ? last.heading_relative_rad * 180 / Math.PI : NaN, 1)}°.\n` +
      `Read failures ${capture.frames.filter(f => !(f.flags & 1)).length}; raw clipping ${capture.frames.filter(f => f.flags & 8).length}; ` +
      `skipped control ticks ${capture.frames.reduce((sum, f) => sum + f.skipped_ticks, 0)}.\n` +
      `At rest, |accel| should be near 1 g and corrected gyro near zero. Walking acceleration can differ from 1 g; this summary does not establish that the IMU axes or scale are correct.`;
  }

  class Controller {
    constructor(app) {
      this.app = app; this.characteristic = null; this.epoch = 0; this.capture = null;
      this.status = document.getElementById("gaitStatus");
      this.fetchButton = document.getElementById("gaitFetch");
      this.jsonButton = document.getElementById("gaitJson");
      this.csvButton = document.getElementById("gaitCsv");
      this.summary = document.getElementById("gaitSummary");
      this.fetchButton.addEventListener("click", () => this.fetch());
      this.jsonButton.addEventListener("click", () => this.save("json"));
      this.csvButton.addEventListener("click", () => this.save("csv"));
    }
    updateButtons() {
      this.fetchButton.disabled = !this.app.connected() || !this.characteristic || this.app.busy() || this.app.sleeping();
      this.jsonButton.disabled = this.csvButton.disabled = !this.capture;
    }
    async connect(service) {
      const epoch = ++this.epoch;
      this.characteristic = null;
      try {
        const characteristic = await this.app.queue(() => service.getCharacteristic(UUID));
        if (epoch !== this.epoch) return;
        this.characteristic = characteristic;
        this.status.textContent = "Ready. Start records automatically; Stop before fetching.";
      } catch (error) {
        if (epoch === this.epoch)
          this.status.textContent = "Gait logging requires updated firmware. Install rick-gait-diagnostics.bin, then reconnect. If needed, forget RickBot and pair again.";
      }
      this.updateButtons();
    }
    disconnect() {
      ++this.epoch; this.characteristic = null;
      this.status.textContent = "Disconnected. Downloaded results are kept; the robot retains its latest log until the next Start or reboot.";
      this.updateButtons();
    }
    async fetch() {
      if (!this.app.connected() || this.app.busy() || !this.characteristic) return;
      const epoch = this.epoch, characteristic = this.characteristic;
      this.app.setBusy(true); this.app.stopPolling();
      this.status.textContent = "Checking that the policy is stopped…";
      const check = () => {
        if (epoch !== this.epoch || !this.app.connected()) throw new Error("Disconnected during download. Reconnect and fetch again.");
      };
      try {
        const state = await this.app.readState(); check();
        if (state === 2 || state === 8) throw new Error("Press Stop before fetching the gait log.");
        const exchange = payload => this.app.queue(async () => {
          check();
          if (characteristic.writeValueWithResponse) await characteristic.writeValueWithResponse(payload);
          else await characteristic.writeValue(payload);
          check(); const value = await characteristic.readValue(); check(); return value;
        });
        const capture = await fetchCapture(exchange, (count, total) => {
          this.status.textContent = `Fetching ${count} / ${total} samples…`;
        });
        check(); this.capture = capture;
        this.summary.textContent = summarize(capture);
        this.status.textContent = "Log ready. Save JSON for analysis; CSV is also available.";
      } catch (error) {
        if (epoch === this.epoch) this.status.textContent = error.message || String(error);
      } finally {
        if (epoch === this.epoch) {
          this.app.setBusy(false);
          if (this.app.connected()) this.app.startPolling();
          this.updateButtons();
        }
      }
    }
    save(format) {
      if (!this.capture) return;
      const content = format === "json" ? JSON.stringify(this.capture, null, 2) : captureCsv(this.capture);
      const blob = new Blob([content], {type: format === "json" ? "application/json" : "text/csv"});
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = `rick-start-${this.capture.header.capture_id}.${format}`;
      document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  }
  return {Controller, decodeHeader, decodeFrame, fetchCapture, captureCsv, summarize};
})();
