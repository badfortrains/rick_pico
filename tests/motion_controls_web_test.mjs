import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import {execFileSync} from "node:child_process";
import os from "node:os";
import path from "node:path";

const source = fs.readFileSync(new URL("../motion-controls.js", import.meta.url), "utf8");
const context = vm.createContext({module: {exports: {}}, setTimeout, clearTimeout});
vm.runInContext(source, context);
const {RickMotionControls} = context.module.exports;
const element = () => ({value: "0", min: "0", max: "0", disabled: true,
  listeners: {}, addEventListener(name, callback) { this.listeners[name] = callback; }});
const elements = Object.fromEntries(["speed", "turn", "speedValue", "turnValue", "straight", "stand", "status"]
  .map(name => [name, element()]));
let connected = true, locked = false, delayWrite = false;
const writes = [], errors = [], delayed = [];
const io = {
  connected: () => connected, locked: () => locked || !connected,
  error: text => errors.push(text),
  async write(bytes, valid) {
    if (delayWrite) await new Promise(resolve => delayed.push(resolve));
    if (valid()) writes.push(Array.from(bytes));
  },
};
const controls = new RickMotionControls(elements, io);
const status = new DataView(new ArrayBuffer(16));
status.setUint8(3, 1);
[-40,80,-600,600,50,0].forEach((x,i) => status.setInt16(4+i*2, x, true));
controls.readStatus(status);
assert.equal(elements.speed.min, -40); assert.equal(elements.speed.max, 80);
assert.equal(elements.speed.value, 50); assert.equal(elements.speed.disabled, false);
assert.equal(elements.turn.min, -600);
elements.speed.value = "-40"; elements.turn.value = "600";
const browserPacket = Array.from(controls.packet());
assert.deepEqual(browserPacket, [6,216,255,88,2]);
controls.renderValues();
assert.match(elements.speedValue.textContent, /reverse/);
assert.match(elements.turnValue.textContent, /Left 0.60/);
await controls.prepareStart();
assert.deepEqual(writes.at(-1), [6,216,255,88,2]);
elements.speed.value = "81";
assert.throws(() => controls.packet(), /outside/);
elements.speed.value = "0";
// Continuous input coalesces into the most recent setpoint.
for (const speed of [10,20,30,40]) {
  elements.speed.value = String(speed); elements.speed.listeners.input();
}
await new Promise(resolve => setTimeout(resolve, 130));
assert.equal(writes.length, 2);
assert.deepEqual(writes.at(-1), [6,40,0,88,2]);
// Cancel queued traffic on Stop, disconnect, sleep, or a new command.
delayWrite = true;
elements.speed.value = "60"; controls.schedule();
clearTimeout(controls.timer); controls.timer = null;
const inFlight = controls.flush();
controls.cancel(); delayed.shift()(); await inFlight;
assert.equal(writes.length, 2);
delayWrite = false;
elements.stand.listeners.click();
await new Promise(resolve => setTimeout(resolve, 130));
assert.deepEqual(writes.at(-1), [6,0,0,0,0]);
locked = true; controls.updateButtons();
assert.equal(elements.stand.disabled, true);
const before = writes.length;
controls.schedule(); await controls.flush(); assert.equal(writes.length, before);
locked = false;
controls.readStatus(new DataView(Uint8Array.of(3,1,0).buffer));
assert.equal(controls.supported, false); assert.equal(elements.speed.disabled, true);
assert.match(elements.status.textContent, /newly trained/);
controls.readStatus(status);
connected = false; controls.disconnect();
assert.equal(controls.pending, null); assert.equal(elements.turn.disabled, true);
assert.equal(errors.length, 0);

// Feed actual browser bytes into the firmware decoder, rather than a JS copy.
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "rick-motion-web-"));
try {
  const root = new URL("../", import.meta.url).pathname;
  const cpp = path.join(dir,"decode.cpp"), exe = path.join(dir,"decode");
  fs.writeFileSync(cpp, `#include <cassert>\n#include "motion_control.h"\nint main(){
    uint8_t packet[]={${browserPacket.join(",")}}; rick::MotionCommand out;
    assert(rick::decode_motion_command(packet,5,{-.04f,-.6f},{.08f,.6f},out));
    assert(out.forward==-.04f && out.yaw==.6f); }`);
  execFileSync("c++",["-std=c++17","-I",root,cpp,"-o",exe]); execFileSync(exe);
} finally { fs.rmSync(dir,{recursive:true,force:true}); }
console.log("Motion UI limits, signed wire format, coalescing, cancellation, legacy policy, and standing passed.");
