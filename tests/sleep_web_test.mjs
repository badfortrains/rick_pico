import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import vm from "node:vm";

const html = await readFile(new URL("../bluetooth.html", import.meta.url), "utf8");
new vm.Script(html.match(/<script>([\s\S]*?)<\/script>/)[1]);
const commandSource = html.slice(html.indexOf("    function wakeFailureMessage("),
                                 html.indexOf("    async function adjustServo("));
async function commandTest(command, responses, initialState = 3, supported = true, wakeError = 0) {
  let clock = 0, connected = true;
  const writes = [], messages = [];
  let disconnected = 0, polls = 0, reads = 0;
  const fake = {
    busy: false, sleepSupported: supported, currentRobotState: initialState, lastWakeError: wakeError,
    isConnected: () => connected,
    robotIsSleeping: () => [5, 7, 8].includes(fake.currentRobotState),
    window: { confirm: () => true },
    commandMessages: ["default", "centers", "start", "stop", "sleep", "wake"],
    controlCharacteristic: {},
    showConnection: () => {}, updateButtons: () => {}, stopImuPolling: () => {},
    setMessage: (message, error = false) => messages.push({ message, error }),
    writeWithResponse: async (_, data) => writes.push(Array.from(data)),
    readRobotState: async () => {
      const state = responses[Math.min(reads++, responses.length - 1)];
      fake.currentRobotState = state;
      return state;
    },
    startImuPolling: () => { if (!fake.robotIsSleeping()) ++polls; },
    handleDisconnect: () => {},
    device: { gatt: { get connected() { return connected; }, disconnect() { connected = false; ++disconnected; } } },
    Date: { now: () => clock },
    setTimeout: (callback, delay) => { clock += delay; queueMicrotask(callback); },
  };
  vm.runInContext(commandSource, vm.createContext(fake));
  await fake.sendCommand(command);
  assert.equal(fake.busy, false);
  return { writes, messages, disconnected, polls, reads, clock };
}

// Acknowledging the write must not count as entering sleep. Wait for firmware.
let result = await commandTest(4, [3, 3, 5]);
assert.deepEqual(result.writes, [[4]]);
assert.equal(result.reads, 3);
assert.equal(result.disconnected, 1);
assert.equal(result.polls, 0);
assert.equal(result.messages.at(-1).message, "sleep");

// Rejected commands never fabricate sleep or disconnect the user.
result = await commandTest(4, [3]);
assert.equal(result.disconnected, 0);
assert.equal(result.messages.at(-1).error, true);
result = await commandTest(4, [7]);
assert.equal(result.disconnected, 0);
assert.match(result.messages.at(-1).message, /shutdown failed/);

// Wake restores polling only after the firmware reports awake and settles.
result = await commandTest(5, [5, 5, 6], 5);
assert.deepEqual(result.writes, [[5]]);
assert.equal(result.polls, 1);
assert.equal(result.disconnected, 0);
assert.ok(result.clock >= 1400);
result = await commandTest(5, [7], 5);
assert.equal(result.polls, 0);
assert.match(result.messages.at(-1).message, /Wake failed/);

// Transitional or missing confirmation is distinct from a sensor fault.
result = await commandTest(5, [8, 8, 6], 5);
assert.equal(result.messages.at(-1).message, "wake");
result = await commandTest(5, [8], 5);
assert.match(result.messages.at(-1).message, /not confirmed.*Waking/);
assert.doesNotMatch(result.messages.at(-1).message, /IMU may need attention|Reconnect/);
result = await commandTest(5, [7], 5, true, 7);
assert.match(result.messages.at(-1).message, /accelerometer readings did not settle/);
result = await commandTest(5, [7], 5, true, 3);
assert.match(result.messages.at(-1).message, /resume sequence/);
result = await commandTest(5, [7], 5, true, 6);
assert.match(result.messages.at(-1).message, /no readable samples/);

// Legacy firmware has no sleep capability; ordinary commands cannot wake.
assert.deepEqual((await commandTest(4, [3], 3, false)).writes, []);
assert.deepEqual((await commandTest(2, [5], 5)).writes, []);
assert.deepEqual((await commandTest(0, [7], 7)).writes, []);

// Polling must stay off even when reconnecting to an already sleeping robot.
const polling = vm.createContext({
  stopImuPolling() {}, robotIsSleeping: () => true,
  readImuTelemetry() { assert.fail("IMU polled while sleeping"); },
  readBatteryVoltage() { assert.fail("battery polled while sleeping"); },
  window: { setInterval() { assert.fail("poll timer started while sleeping"); } },
});
vm.runInContext(html.slice(html.indexOf("    function startImuPolling("),
                           html.indexOf("    async function writeWithResponse(")), polling);
polling.startImuPolling();
console.log("Sleep/wake confirmation, rejection, compatibility, and polling tests passed");

// While asleep only Wake and disconnect are available; old firmware cannot
// accidentally receive the newly assigned sleep command numbers.
let sleeping = false;
const buttons = Array.from({ length: 6 }, (_, command) => ({ dataset: { command: String(command) } }));
const controls = [{}];
const buttonFake = {
  isConnected: () => true, busy: false, sleepSupported: false,
  robotIsSleeping: () => sleeping, commandButtons: buttons,
  document: { querySelectorAll: () => controls }, centeredServos: new Set(),
  servoNames: Array(8), bluetoothSupported: true, otaIsAvailable: () => true,
  otaProtocolVersion: 3, selectedFirmware: {}, otaUpdating: false,
  rawCharacteristic: {}, rawInProgress: false, rawResults: new Map(),
};
for (const name of ["finishCalibration", "copyCenters", "connectButton", "firmwareFileInput",
  "installFirmware", "abortFirmware", "rawCaptureButton", "rawCheckSupportButton", "rawPose",
  "rawCancelButton", "rawCopyButton", "rawClearButton"]) buttonFake[name] = {};
const buttonContext = vm.createContext(buttonFake);
vm.runInContext(html.slice(html.indexOf("    function updateButtons("),
                           html.indexOf("    function showConnection(")), buttonContext);
buttonContext.updateButtons();
assert.equal(buttons[4].disabled, true);
assert.equal(buttons[5].disabled, true);
buttonFake.sleepSupported = true;
buttonContext.updateButtons();
assert.equal(buttons[4].disabled, false);
assert.equal(buttons[5].disabled, true);
sleeping = true;
buttonContext.updateButtons();
assert.deepEqual(buttons.map(button => button.disabled), [true,true,true,true,true,false]);
assert.equal(controls[0].disabled, true);
assert.equal(buttonFake.installFirmware.disabled, true);
assert.equal(buttonFake.rawCaptureButton.disabled, true);
assert.equal(buttonFake.connectButton.disabled, false);
console.log("Sleep/wake button availability tests passed");

// Decode optional diagnostics without breaking older one- or two-byte replies.
const readerFake = {
  controlCharacteristic: {}, queueGattOperation: operation => operation(),
  sleepSupported: false, lastWakeError: 0, showRobotState() {},
};
const reader = vm.createContext(readerFake);
vm.runInContext(html.slice(html.indexOf("    async function readRobotState("),
                           html.indexOf("    async function readServoCenters(")), reader);
for (const [bytes, supported, error] of [
  [[3], false, 0], [[5,1], true, 0], [[7,1,6], true, 6], [[6,1,0], true, 0],
]) {
  readerFake.controlCharacteristic.readValue = async () => new DataView(Uint8Array.from(bytes).buffer);
  assert.equal(await reader.readRobotState(), bytes[0]);
  assert.equal(readerFake.sleepSupported, supported);
  assert.equal(readerFake.lastWakeError, error);
}
readerFake.controlCharacteristic.readValue = async () => new DataView(new ArrayBuffer(0));
await assert.rejects(reader.readRobotState(), /empty control status/);
console.log("Wake diagnostics, timeout, and legacy status decoding tests passed");
