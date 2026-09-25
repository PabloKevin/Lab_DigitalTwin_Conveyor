"""
sim_conveyor.py - pretends to be the Arduino UNO conveyor controller so you can test the twin
without hardware. Uses a virtual serial port (a pty pair) - Linux/macOS only, needs no extra deps.

    python sim_conveyor.py                 # normal conveyor
    python sim_conveyor.py --loss 25       # 'real' conveyor loses 25 % speed (friction) -> the twin raises a divergence

Speaks exactly the protocol of firmware/arduino_uno_conveyor/ (see serial_bridge.py):
  -> PC   one JSON line every 100 ms: {"rpm":..,"setpoint":..,"output":..,"dir":"F"}
  <- PC   F | R | S  (direction),  V<0-100> (speed %),  P<f> | I<f> | D<f> (PID gains)

On startup this prints the pty device path (e.g. /dev/pts/4) - copy it into config.py's
SERIAL["port"] (replacing "auto") before starting app.py, since "auto" only looks for real USB
devices. Restart both scripts if you restart this one - the path changes each run.
"""
import argparse
import json
import os
import pty
import random
import time

import config as cfg

ap = argparse.ArgumentParser()
ap.add_argument("--loss", type=float, default=0.0, help="percent speed lost by the 'real' conveyor")
args = ap.parse_args()

st = dict(direction="S", speed=0.0, loss=args.loss, rpm=0.0, kp=2.0, ki=0.5, kd=0.1)

master, slave = pty.openpty()
port_name = os.ttyname(slave)
os.set_blocking(master, False)
print(f"Simulated conveyor on {port_name}")
print(f"  -> set SERIAL[\"port\"] = \"{port_name}\" in config.py, then run app.py   (Ctrl+C to stop)")

buf = b""
dt = 0.1
try:
    while True:
        # ---- read any pending commands (non-blocking) ----
        try:
            buf += os.read(master, 256)
        except BlockingIOError:
            pass
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.decode(errors="replace").strip()
            if not line or line == "H":                   # H = keep-alive, nothing to simulate
                continue
            cmd, rest = line[0], line[1:]
            try:
                if cmd in ("F", "R", "S"):
                    st["direction"] = cmd
                elif cmd == "V":
                    st["speed"] = float(rest)
                elif cmd in ("P", "I", "D"):
                    st[{"P": "kp", "I": "ki", "D": "kd"}[cmd]] = float(rest)
                print(f"cmd {line}")
            except ValueError:
                pass

        # ---- fake physics (same dynamics as the twin model: tau = 0.5 s) ----
        sign = {"F": 1, "R": -1}.get(st["direction"], 0)
        target = sign * cfg.MAX_RPM * st["speed"] / 100.0 * (1 - st["loss"] / 100.0)
        st["rpm"] += (target - st["rpm"]) * dt / 0.5
        rpm = st["rpm"] + random.gauss(0, 1.2)
        pwm = min(255, abs(target) / cfg.MAX_RPM * 255 * 1.05)
        line = json.dumps(dict(rpm=round(rpm, 1), setpoint=round(sign * cfg.MAX_RPM * st["speed"] / 100.0, 1),
                                output=round(pwm), dir=st["direction"], speed_pct=st["speed"])) + "\n"
        os.write(master, line.encode())
        time.sleep(dt)
except KeyboardInterrupt:
    pass
finally:
    os.close(master)
