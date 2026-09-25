"""
serial_bridge.py - talks to the Arduino UNO conveyor controller over USB serial.

Replaces MQTT for the conveyor (motor + encoder) link: the ESP32's H-bridge wiring wasn't driving
the motor reliably, so the conveyor controller moved back to an Arduino UNO (see
firmware/arduino_uno_conveyor/). The camera/vision pipeline is untouched - it still talks HTTP to
the ESP32-CAM and still publishes detections over MQTT (see vision.py / mqtt_bridge.py).

Protocol (must match firmware/arduino_uno_conveyor/arduino_uno_conveyor.ino):
  Arduino -> PC   one JSON line every 100 ms:
                  {"rpm":-12.0,"setpoint":100.0,"output":120,"dir":"F","distance_cm":58.5,"obj_speed_cm_s":0.0}
  PC -> Arduino   one line per command:
                  F | R | S                        direction
                  V<0..100>                         speed setpoint, % of MAX_RPM
                  P<float> | I<float> | D<float>    Kp / Ki / Kd

Exposes the same interface app.py/controls.py already use for the MQTT bridge (start(), publish()),
so CONTROLS in config.py (topics "conveyor/cmd/...") did not need to change - only the transport did.
"""
import json
import threading
import time

import serial
import serial.tools.list_ports

import config as cfg
from state import is_num, state

S = cfg.SERIAL
_ARDUINO_HINTS = ("arduino", "ch340", "wch", "usb serial", "usb-serial", "usb2.0-serial", "usbmodem", "usbserial")
# USB-serial chip vendor ids commonly found on Arduino (clone) boards, matched against p.hwid
# ("...VID:PID=XXXX:YYYY...") - more reliable than the description text, which is often just a
# generic "USB Serial" with no vendor name (that's what a genuine CH340 reports on Linux).
_ARDUINO_VIDS = ("1a86",    # QinHeng CH340/CH341 - most common on UNO clones
                 "0403",    # FTDI
                 "10c4",    # Silicon Labs CP210x
                 "2341", "2a03")  # Arduino LLC / Arduino SA (genuine boards)


class SerialBridge:
    def __init__(self):
        # topic -> list of variable dicts that read from it (mirrors MqttBridge's by_topic)
        self.by_topic = {}
        for v in cfg.VARIABLES:
            if v.get("source") == "serial":
                self.by_topic.setdefault(v["topic"], []).append(v)

        self._ser = None
        self._lock = threading.Lock()
        self._last_dir_line = "S"
        self._last_err = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        threading.Thread(target=self._reader_loop, daemon=True, name="serial-reader").start()
        threading.Thread(target=self._heartbeat_loop, daemon=True, name="serial-heartbeat").start()

    @staticmethod
    def _find_port():
        configured = S.get("port", "auto")
        if configured and configured != "auto":
            return configured
        candidates = list(serial.tools.list_ports.comports())
        for p in candidates:
            desc = f"{p.description} {p.manufacturer or ''}".lower()
            if any(hint in desc for hint in _ARDUINO_HINTS) or any(vid in (p.hwid or "").lower() for vid in _ARDUINO_VIDS):
                return p.device
        return None

    def _reader_loop(self):
        while True:
            port = self._find_port()
            if not port:
                state.device_status = "offline"
                time.sleep(S["reconnect_s"])
                continue
            try:
                with serial.Serial(port, S["baud"], timeout=1) as ser:
                    with self._lock:
                        self._ser = ser
                    state.log(f"Serial connected to {port}")
                    time.sleep(2.0)                      # let the Arduino finish its auto-reset boot
                    for raw in ser:
                        line = raw.decode("utf-8", errors="replace").strip()
                        if line:
                            self._handle_line(line)
            except (serial.SerialException, OSError) as e:
                state.log(f"Serial error on {port}: {e}", "warn")
            with self._lock:
                self._ser = None
            state.device_status = "offline"
            time.sleep(S["reconnect_s"])

    def _heartbeat_loop(self):
        """Re-sends the last direction even if unchanged, so the Arduino's link-loss failsafe sees traffic."""
        while True:
            time.sleep(S["heartbeat_s"])
            self._write(self._last_dir_line, quiet=True)

    # ── incoming ───────────────────────────────────────────────────────────
    def _handle_line(self, line):
        now = time.time()
        state.raw[cfg.TOPIC_TELEMETRY] = (now, line)
        state.device_status = "online"
        try:
            data = json.loads(line)
        except ValueError:
            self._err(f"Bad line from Arduino: {line[:60]}")
            return
        for v in self.by_topic.get(cfg.TOPIC_TELEMETRY, []):
            key = v.get("key")
            val = data.get(key) if key else None
            if val is None:
                continue
            if is_num(val):
                val = float(val)
            state.set_value(v["id"], val, now)

    def _err(self, msg):
        if time.time() - self._last_err > 5:            # avoid flooding the log
            self._last_err = time.time()
            state.log(msg, "warn")

    # ── outgoing ───────────────────────────────────────────────────────────
    def _write(self, line, quiet=False):
        with self._lock:
            ser = self._ser
        if ser is None:
            if not quiet:
                state.log(f"Not sent (no serial link): {line}", "warn")
            return False
        try:
            ser.write((line + "\n").encode("ascii"))
            return True
        except (serial.SerialException, OSError) as e:
            state.log(f"Serial write failed: {e}", "warn")
            return False

    def publish(self, topic, payload, retain=False, qos=1, quiet=False):
        """Same signature as MqttBridge.publish() so controls.py doesn't need to know the transport."""
        prefix = f"{cfg.TOPIC_PREFIX}/cmd/"
        if not topic.startswith(prefix):
            return False                                  # not a conveyor command - nothing to do here
        key = topic[len(prefix):]
        line = {"direction": lambda p: p, "speed": lambda p: f"V{p}",
                "kp": lambda p: f"P{p}", "ki": lambda p: f"I{p}", "kd": lambda p: f"D{p}"}.get(key)
        if line is None:
            return False
        line = line(payload)
        if key == "direction":
            self._last_dir_line = line
        return self._write(line, quiet=quiet)
