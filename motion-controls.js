"use strict";

// Bluetooth units are signed mm/s and mrad/s. Firmware advertises the limits
// exported with its policy; the page never invents a wider operating range.
class RickMotionControls {
  constructor(elements, io) {
    this.elements = elements;
    this.io = io;
    this.supported = false;
    this.generation = 0;
    this.pending = null;
    this.timer = null;
    this.sending = false;
    for (const input of [elements.speed, elements.turn]) {
      input.addEventListener("input", () => { this.renderValues(); this.schedule(); });
    }
    elements.straight.addEventListener("click", () => {
      elements.turn.value = "0"; this.renderValues(); this.schedule();
    });
    elements.stand.addEventListener("click", () => {
      elements.speed.value = elements.turn.value = "0";
      this.renderValues(); this.schedule();
    });
    this.updateButtons();
  }

  values() {
    return [Number(this.elements.speed.value), Number(this.elements.turn.value)];
  }

  renderValues() {
    const [speed, turn] = this.values();
    this.elements.speedValue.textContent = `${Math.abs(speed)} mm/s${speed < 0 ? " reverse" : speed > 0 ? " forward" : ""}`;
    this.elements.turnValue.textContent = turn === 0 ? "Straight" :
      `${turn > 0 ? "Left" : "Right"} ${(Math.abs(turn) / 1000).toFixed(2)} rad/s`;
  }

  readStatus(value) {
    const supported = value.byteLength >= 16 && value.getUint8(3) === 1;
    const speed = this.elements.speed, turn = this.elements.turn;
    this.supported = supported;
    if (supported) {
      speed.min = value.getInt16(4, true); speed.max = value.getInt16(6, true);
      turn.min = value.getInt16(8, true); turn.max = value.getInt16(10, true);
      if (!(Number(speed.min) <= 0 && Number(speed.max) >= 0 &&
            Number(turn.min) <= 0 && Number(turn.max) >= 0)) this.supported = false;
      if (!this.sending && this.pending === null && this.timer === null) {
        speed.value = value.getInt16(12, true); turn.value = value.getInt16(14, true);
      }
    }
    this.renderValues();
    this.updateButtons();
  }

  updateButtons() {
    const locked = !this.supported || this.io.locked();
    for (const control of [this.elements.speed, this.elements.turn,
      this.elements.straight, this.elements.stand]) control.disabled = locked;
    this.elements.status.textContent = !this.io.connected() ? "Connect to set speed and steering." :
      !this.supported ? "Speed and steering require firmware with a newly trained command policy." :
      "Changes apply while walking. Zero speed and turn rate request standing; Stop holds the current pose.";
  }

  cancel() {
    ++this.generation;
    clearTimeout(this.timer); this.timer = null; this.pending = null;
  }

  disconnect() {
    this.cancel(); this.supported = false;
    this.elements.speed.value = this.elements.turn.value = "0";
    this.renderValues(); this.updateButtons();
  }

  packet(values = this.values()) {
    const packet = new Uint8Array(5), view = new DataView(packet.buffer);
    packet[0] = 6;
    const inputs = [this.elements.speed, this.elements.turn];
    values.forEach((value, index) => {
      if (!Number.isInteger(value) || value < Number(inputs[index].min) ||
          value > Number(inputs[index].max)) throw new Error("Motion command is outside the trained range.");
      view.setInt16(1 + index * 2, value, true);
    });
    return packet;
  }

  schedule() {
    if (!this.supported || this.io.locked()) return;
    this.pending = this.values();
    if (this.timer !== null || this.sending) return;
    this.timer = setTimeout(() => { this.timer = null; this.flush(); }, 100);
  }

  async flush() {
    if (this.sending || this.pending === null || !this.supported || this.io.locked()) return;
    const generation = this.generation, values = this.pending;
    this.pending = null; this.sending = true;
    try {
      await this.io.write(this.packet(values), () => generation === this.generation);
    } catch (error) {
      if (generation === this.generation) {
        this.cancel(); this.io.error(error.message || error.name);
      }
    } finally {
      this.sending = false;
      if (this.pending !== null) this.schedule();
    }
  }

  // START sends the selected setpoint before enabling the policy. This also
  // handles a slider edit whose debounce timer has not fired yet.
  async prepareStart() {
    this.cancel();
    if (this.supported) await this.io.write(this.packet(), () => true);
  }
}

if (typeof module !== "undefined") module.exports = { RickMotionControls };
