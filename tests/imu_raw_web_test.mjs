import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const html = await readFile(new URL("../bluetooth.html", import.meta.url), "utf8");
// Syntax-check the entire page, then test the actual decoder with C++ output.
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
const source = html.slice(html.indexOf("    function decodeRawCapture("),
                          html.indexOf("    function renderRawResults("));
const context = vm.createContext({});
vm.runInContext(source, context);
const bytes = await readFile(process.argv[2]);
const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
const result = context.decodeRawCapture(view);
assert.equal(result.token, 0xabcd);
assert.equal(result.pose, 3);
assert.equal(result.state, 2);
assert.equal(result.rows.length, 3);
assert.equal(result.rows[0].sensor[0], -16000);
assert.equal(result.rows[1].sensor[0], -8000);
assert.equal(result.rows[2].sensor[0], -4000);
assert.deepEqual(Array.from(result.rows[0].robot), [4000, -16000, -8000]);
assert.deepEqual(Array.from(result.rows[0].span), [0, 0, 0]);
const csv = context.rawCaptureCsv(new Map([[3, result.rows]]));
assert.equal(csv.split("\n").length, 4);
assert.match(csv, /left_down,2,0x03,50,-16000\.0,8000\.0,-4000\.0,4000\.0,-16000\.0,-8000\.0,0,0,0,0/);
assert.throws(() => context.decodeRawCapture(new DataView(new ArrayBuffer(12))), /Unsupported/);
const mutated = new Uint8Array(bytes);
const bad = new DataView(mutated.buffer);
bad.setUint8(6, 2);
assert.throws(() => context.decodeRawCapture(bad), /did not finish/);
bad.setUint8(6, 3); bad.setUint8(13, 0x23);
assert.throws(() => context.decodeRawCapture(bad), /range readback/);
bad.setUint8(13, 0x03); bad.setUint8(14, 0);
assert.throws(() => context.decodeRawCapture(bad), /Incomplete/);
console.log("IMU raw web decoding and CSV tests passed");

const flowSource = html.slice(html.indexOf("    function decodeRawCapture("),
                              html.indexOf("    async function readImuTelemetry("));
async function testCaptureFlow(terminalState) {
  const writes = [];
  let reads = 0, pollsRestarted = 0;
  function response(state, token, pose) {
    const data = new Uint8Array(bytes);
    const view = new DataView(data.buffer);
    view.setUint8(1, state);
    view.setUint16(4, token, true);
    view.setUint8(3, pose);
    if (state !== 2) view.setUint8(6, 0);
    if (state === 1) { view.setUint8(8, 2); view.setUint8(7, 20); }
    return view;
  }
  const fake = {
    rawCharacteristic: { readValue: async () => {
      ++reads;
      if (reads === 1) return response(0, 0, 0); // Preflight.
      if (reads === 2) return response(2, 99, 3); // Old result must be ignored.
      if (reads === 3) return response(1, 101, 3);
      return response(terminalState, 101, 3);
    } },
    rawRun: 0, rawToken: 100, rawInProgress: false, rawCancelRequested: false, busy: false,
    rawPose: { value: "3", selectedIndex: 0, options: [{ text: "Left side down" }] },
    rawStatus: {}, rawProgress: {}, rawOutput: {}, imuStatus: {}, rawResults: new Map(),
    rawCancelButton: {},
    isConnected: () => true, stopImuPolling: () => {}, updateButtons: () => {},
    startImuPolling: () => { ++pollsRestarted; }, showRobotState: state => assert.equal(state, 3),
    queueGattOperation: operation => operation(),
    writeWithResponse: async (_, data) => writes.push(Array.from(data)),
    setTimeout: callback => queueMicrotask(callback),
  };
  // A previous valid pose must survive a cancelled replacement.
  if (terminalState === 4) fake.rawResults.set(3, result.rows);
  const flow = vm.createContext(fake);
  vm.runInContext(flowSource, flow);
  await flow.captureRawPose();
  assert.equal(reads, 4);
  assert.deepEqual(writes, [[1, 101, 0, 3]]);
  assert.equal(fake.busy, false);
  assert.equal(fake.rawInProgress, false);
  assert.equal(pollsRestarted, 1);
  assert.equal(fake.rawResults.size, 1);
  if (terminalState === 2) {
    assert.match(fake.rawOutput.value, /left_down/);
    assert.match(fake.rawStatus.textContent, /Normal range restored/);
    assert.equal(fake.rawPose.value, "1");
  } else {
    assert.equal(fake.rawResults.get(3), result.rows);
    assert.match(fake.rawStatus.textContent, /cancelled/);
  }
}
await testCaptureFlow(2);
await testCaptureFlow(4);
console.log("IMU raw web capture, stale-result, and cancellation tests passed");

// Missing optional services must not disable OTA or prevent a discovery retry.
let connected = false;
const supportFake = {
  isConnected: () => connected, busy: false, commandButtons: [{}],
  document: { querySelectorAll: () => [] }, centeredServos: new Set(),
  servoNames: Array(8), bluetoothSupported: true, otaIsAvailable: () => connected,
  otaProtocolVersion: 3, selectedFirmware: {}, otaUpdating: false,
  rawCharacteristic: null, rawInProgress: false, rawResults: new Map(),
  rawStatus: {}, IMU_RAW_UUID: "raw-test", queueGattOperation: operation => operation(),
};
for (const name of ["finishCalibration", "copyCenters", "connectButton", "firmwareFileInput",
  "installFirmware", "abortFirmware", "rawCaptureButton", "rawCheckSupportButton", "rawPose",
  "rawCancelButton", "rawCopyButton", "rawClearButton"]) supportFake[name] = {};
const support = vm.createContext(supportFake);
vm.runInContext(
  html.slice(html.indexOf("    function updateButtons("), html.indexOf("    function showConnection(")) +
  html.slice(html.indexOf("    async function discoverRawSupport("), html.indexOf("    function decodeRawCapture(")), support);
support.updateButtons();
assert.equal(supportFake.rawCheckSupportButton.disabled, true);
connected = true;
support.updateButtons();
assert.equal(supportFake.rawCaptureButton.disabled, true);
assert.equal(supportFake.rawCheckSupportButton.disabled, false);
assert.equal(supportFake.installFirmware.disabled, false);
assert.equal(supportFake.commandButtons[0].disabled, false);
const missingService = { getCharacteristic: async () => {
  const error = new Error("No matching characteristic"); error.name = "NotFoundError"; throw error;
} };
assert.equal(await support.discoverRawSupport(missingService), false);
assert.match(supportFake.rawStatus.textContent, /NotFoundError: No matching characteristic/);
const rawHandle = {};
assert.equal(await support.discoverRawSupport({ getCharacteristic: async () => rawHandle }), true);
support.updateButtons();
assert.equal(supportFake.rawCaptureButton.disabled, false);
assert.equal(supportFake.rawPose.disabled, false);
assert.equal(supportFake.rawCancelButton.disabled, true);
assert.equal(supportFake.rawCopyButton.disabled, true);
supportFake.busy = true;
support.updateButtons();
assert.equal(supportFake.rawCaptureButton.disabled, true);
assert.equal(supportFake.rawCheckSupportButton.disabled, true);
console.log("IMU raw optional-service discovery and button-state tests passed");
