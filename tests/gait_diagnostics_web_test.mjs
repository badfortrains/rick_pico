import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {execFileSync} from "node:child_process";
import os from "node:os";
import path from "node:path";

const root = new URL("../", import.meta.url);
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rick-gait-test-"));
try {
  const executable = path.join(dir, "capture"), fixture = path.join(dir, "capture.bin");
  execFileSync("c++", ["-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", root.pathname,
    new URL("gait_diagnostics_test.cpp", import.meta.url).pathname, "-o", executable]);
  execFileSync(executable, [fixture], {stdio: "inherit"});
  const bytes = fs.readFileSync(fixture);
  const view = (start, size) => new DataView(bytes.buffer, bytes.byteOffset + start, size);
  const context = vm.createContext({console});
  vm.runInContext(fs.readFileSync(new URL("gait-diagnostics.js", root), "utf8"), context);
  const api = context.RickGaitDiagnostics;
  const header = api.decodeHeader(view(0, 184));
  assert.equal(header.capture_id, 1); assert.equal(header.count, 3);
  assert.equal(header.pre_roll_count, 1); assert.equal(header.firmware_build, "test-build");
  assert.equal(header.mounting_calibration_id, 1);
  assert.equal(header.initial_action[7], -.25); assert.equal(header.servo_centers_us[7], 1560);
  const frame = api.decodeFrame(view(184, 188), header, 0);
  assert.equal(frame.time_us, -20000); assert.equal(frame.raw_sensor_counts[0], -12345);
  assert.equal(frame.raw_sensor_counts[5], 2345); assert.equal(frame.servo_pulse_us[7], 1601);
  assert.ok(Math.abs(frame.heading_relative_rad + .1) < 1e-6);
  assert.ok(Math.abs(frame.gyro_bias_rad_s[2] - .1) < 1e-6);
  const requests = [];
  async function exchange(payload) {
    requests.push(Array.from(payload));
    if (payload[0] === 0) return view(0, 184);
    const p = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);
    assert.equal(p.getUint32(1, true), 1);
    return view(184 + 188 * p.getUint16(5, true), 188);
  }
  const capture = await api.fetchCapture(exchange);
  assert.equal(capture.frames.length, 3);
  assert.match(capture.axes.sensor_to_policy, /mounting calibration 1/);
  const legacyHeader = new Uint8Array(bytes.subarray(0,184));
  new DataView(legacyHeader.buffer).setUint32(180, 0, true);
  assert.equal(api.decodeHeader(new DataView(legacyHeader.buffer)).mounting_calibration_id, 0);
  assert.deepEqual(requests, [[0], [1,1,0,0,0,0,0], [1,1,0,0,0,1,0], [1,1,0,0,0,2,0], [0]]);
  const csv = api.captureCsv(capture);
  assert.equal(csv.split("\n").length, 4);
  assert.ok(csv.includes("gyro_policy_rad_s_z")); assert.ok(csv.includes("-12345"));
  assert.match(api.summarize(capture), /3 samples/);
  assert.throws(() => api.decodeFrame(view(184,188), {...header, capture_id: 2}, 0), /changed/);
  assert.throws(() => api.decodeFrame(view(184,188), header, 1), /changed/);
  assert.throws(() => api.decodeHeader(view(0,12)), /unsupported/);
  const copy = new Uint8Array(bytes.subarray(0,184)); copy[2] = 1;
  await assert.rejects(() => api.fetchCapture(async () => new DataView(copy.buffer)), /Still recording/);
  copy[2] = 2; new DataView(copy.buffer).setUint32(4, 2, true);
  let headerReads = 0;
  await assert.rejects(() => api.fetchCapture(async payload => {
    if (payload[0] === 0 && ++headerReads === 2) return new DataView(copy.buffer);
    return exchange(payload);
  }), /replaced/);
  await assert.rejects(() => api.fetchCapture(async payload => {
    if (payload[0] === 1) throw new Error("Disconnected");
    return exchange(payload);
  }), /Disconnected/);
  console.log("Gait web decoding, CSV, stale capture, and interrupted download tests passed");

  // Exercise the real UI controller: never fetch while walking, never issue a
  // motion command, retain completed downloads across link loss/old firmware.
  const elements = Object.fromEntries(["gaitStatus", "gaitFetch", "gaitJson", "gaitCsv", "gaitSummary"]
    .map(id => [id, {textContent: "", addEventListener() {}}]));
  context.document = {getElementById: id => elements[id]};
  let connected = true, busy = false, state = 2, polls = 0, selected, writes = 0;
  let drop = false, controller;
  const characteristic = {
    async writeValueWithResponse(payload) { selected = payload; ++writes; },
    async readValue() {
      if (drop && selected[0] === 1) {
        connected = false; busy = false; controller.disconnect();
        throw new Error("Link lost");
      }
      return exchange(selected);
    },
  };
  const app = {connected: () => connected, busy: () => busy, sleeping: () => false,
    setBusy: value => { busy = value; controller.updateButtons(); },
    queue: operation => Promise.resolve().then(operation), readState: async () => state,
    stopPolling() {}, startPolling() { ++polls; }};
  controller = new api.Controller(app);
  await controller.connect({getCharacteristic: async () => { throw new Error("old firmware"); }});
  assert.equal(elements.gaitFetch.disabled, true);
  await controller.connect({getCharacteristic: async () => characteristic});
  assert.equal(elements.gaitFetch.disabled, false);
  await controller.fetch();
  assert.equal(writes, 0); assert.equal(busy, false);
  assert.match(elements.gaitStatus.textContent, /Press Stop/);
  state = 3;
  await controller.fetch();
  assert.equal(writes, 5); assert.equal(busy, false); assert.equal(polls, 2);
  assert.equal(elements.gaitJson.disabled, false);
  const previous = controller.capture;
  drop = true;
  await controller.fetch();
  assert.equal(controller.capture, previous);
  assert.equal(elements.gaitFetch.disabled, true);
  assert.equal(elements.gaitJson.disabled, false);
  assert.match(elements.gaitStatus.textContent, /Disconnected/);
  const html = fs.readFileSync(new URL("bluetooth.html", root), "utf8");
  assert.ok(html.includes('<script src="gait-diagnostics.js"></script>'));
  new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
  assert.ok(fs.readFileSync(new URL("service-worker.js", root), "utf8").includes('"./gait-diagnostics.js"'));
  console.log("Gait UI discovery, stopped-only download, and disconnect retention tests passed");
} finally {
  fs.rmSync(dir, {recursive: true, force: true});
}
