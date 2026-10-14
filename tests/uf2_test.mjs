import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { webcrypto } from "node:crypto";
import vm from "node:vm";

const html = await readFile(new URL("../bluetooth.html", import.meta.url), "utf8");
const constants = html.match(/    const UF2_\w+ = [^;]+;/g).join("\n");
const parser = html.slice(html.indexOf("    async function parseUf2Firmware("),
                          html.indexOf("    async function chooseFirmware("));
const familyCheck = html.slice(html.indexOf("    function checkFirmwareFamily("),
                               html.indexOf("    async function installSelectedFirmware("));
const protocolConstants = html.match(/    const OTA_PROTOCOL_VERSION_\w+ = [^;]+;/g).join("\n");
const statusReader = html.slice(html.indexOf("    async function readOtaStatus("),
                                html.indexOf("    async function waitForOtaState("));
const context = vm.createContext({
  crypto: webcrypto, Uint8Array, DataView, queueGattOperation: (operation) => operation()
});
vm.runInContext(constants + protocolConstants + parser + familyCheck + statusReader, context);
const arm = 0xe48bff59, riscv = 0xe48bff5a;
for (const [length, protocol, family] of [[16, 1, riscv], [28, 2, riscv],
                                        [32, 2, arm], [32, 2, riscv]]) {
  const status = new DataView(new ArrayBuffer(length));
  status.setUint8(0, protocol);
  if (length >= 32) status.setUint32(28, family, true);
  context.otaStatusCharacteristic = { readValue: async () => status };
  const decoded = await context.readOtaStatus();
  assert.equal(decoded.firmwareFamily, family);
  assert.equal(decoded.protocolVersion, protocol);
}
function uf2(families) {
  const bytes = new Uint8Array(families.length * 512);
  families.forEach((family, index) => {
    const block = new DataView(bytes.buffer, index * 512, 512);
    for (const [offset, value] of [[0, 0x0a324655], [4, 0x9e5d5157],
      [8, 0x2000], [12, 0x10000000 + index * 256], [16, 256],
      [20, index], [24, families.length], [28, family], [508, 0x0ab16f30]]) {
      block.setUint32(offset, value, true);
    }
    bytes.fill(index + 7, index * 512 + 32, index * 512 + 288);
  });
  return bytes;
}
async function parse(bytes) {
  return context.parseUf2Firmware({ name: "test.uf2", arrayBuffer: async () => bytes.buffer });
}
for (const family of [arm, riscv]) {
  const result = await parse(uf2([family, family]));
  assert.equal(result.family, family);
  assert.equal(result.codeSize, 512);
  assert.equal(result.blockCount, 2);
  assert.equal(result.compactStream[0], 7);
  assert.equal(result.compactStream[256], 8);
  assert.equal(result.compactDigest.length, 32);
  context.checkFirmwareFamily(result, { firmwareFamily: family });
  assert.throws(() => context.checkFirmwareFamily(result,
    { firmwareFamily: family === arm ? riscv : arm }), /USB firmware install/);
}
await assert.rejects(parse(uf2([arm, riscv])), /mixes CPU architectures/);
await assert.rejects(parse(uf2([0xe48bff5b])), /Choose the ARM or RISC-V/);
await assert.rejects(parse(new Uint8Array()), /not a complete UF2/);
const missing = uf2([arm, arm]).slice(0, 512);
await assert.rejects(parse(missing), /does not contain a complete/);
const badFlags = uf2([arm]);
new DataView(badFlags.buffer).setUint32(8, 0, true);
await assert.rejects(parse(badFlags), /unsupported firmware block/);
for (const path of process.argv.slice(2)) {
  const bytes = Uint8Array.from(await readFile(path));
  const result = await parse(bytes);
  console.log(`${path}: family=0x${result.family.toString(16)}, ${result.codeSize} bytes`);
}
console.log("UF2 architecture, payload, and rejection checks passed.");
