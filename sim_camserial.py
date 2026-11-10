"""
sim_camserial.py - fake ESP32-CAM on a virtual serial port (Linux/macOS pty), speaking the same packet protocol as
firmware/esp32_cam_serial/ (see camserial.py). Streams a synthetic 640x160 belt band with a cube sliding across.

    python sim_camserial.py                 # prints the port, e.g. /dev/pts/5
    -> config.VISION["camera_url"] = "serial:/dev/pts/5"   or   python -m classifier.capture --url serial:/dev/pts/5
    python sim_camserial.py --junk 0.1      # also inject garbage bytes / corrupted packets (resync test)
"""
import argparse
import os
import random
import threading
import time
import tty

import cv2
import numpy as np

import camserial


class FakeCam:
    def __init__(self, fps=12.0, junk=0.0, size=(640, 160)):
        self.master, slave = os.openpty()
        tty.setraw(self.master)
        tty.setraw(slave)
        self.port = os.ttyname(slave)
        self._slave = slave                          # keep it open so the pty stays alive
        self.fps, self.junk, self.size = fps, junk, size
        self.streaming, self.settings, self.seq = False, {}, 0
        self.t0 = time.monotonic()
        self.commands = []
        self._wlock = threading.Lock()
        self._stop = False

    def start(self):
        threading.Thread(target=self._rx, daemon=True).start()
        threading.Thread(target=self._tx, daemon=True).start()
        self._send(b"L", b"ready (simulated ESP32-CAM)")
        return self

    def stop(self):
        self._stop = True

    def _ts(self):
        return int((time.monotonic() - self.t0) * 1e6)

    def _send(self, ptype, payload, ts=None, corrupt=False):
        pkt = bytearray(camserial.build_packet(ptype, self.seq, self._ts() if ts is None else ts, payload))
        self.seq += 1
        if corrupt:
            pkt[len(pkt) // 2] ^= 0xFF
        with self._wlock:
            if self.junk and random.random() < self.junk:
                os.write(self.master, bytes(random.getrandbits(8) for _ in range(random.randint(1, 40))))
            os.write(self.master, bytes(pkt))

    def frame(self, t):
        w, h = self.size
        if not self.settings.get("raw", 0):      # like the firmware: hardware-JPEG mode sends the full frame
            h = w * 3 // 4
        img = np.full((h, w, 3), (70, 75, 80), np.uint8)
        cv2.putText(img, f"sim {t:6.2f}s  sat {self.settings.get('saturation', 0)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        x = int((t * 80) % (w + 120)) - 60
        y0 = (h - 60) // 2
        cv2.rectangle(img, (x, y0), (x + 60, y0 + 60), (30, 120, 230), -1)
        q = 90 - 2 * self.settings.get("quality", 12)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, max(10, q)])[1].tobytes()

    def _tx(self):
        while not self._stop:
            if self.streaming:
                ts = self._ts()
                self._send(b"J", self.frame(ts / 1e6), ts=ts, corrupt=bool(self.junk) and random.random() < self.junk)
            time.sleep(1.0 / self.fps if self.fps > 0 else 0.01)

    def _rx(self):
        buf = b""
        while not self._stop:
            try:
                buf += os.read(self.master, 256)
            except OSError:
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self._cmd(line.decode().strip())

    def _cmd(self, line):
        self.commands.append(line)
        parts = line.split()
        ans = "err unknown"
        if parts[:1] == ["set"] and len(parts) == 3:
            self.settings[parts[1]] = int(parts[2])
            ans = f"ok {line}"
        elif parts[:1] == ["stream"] and len(parts) == 2:
            self.streaming = parts[1] == "1"
            ans = f"ok {line}"
        elif parts[:1] == ["fps"] and len(parts) == 2:
            self.fps = float(parts[1])
            ans = f"ok {line}"
        elif parts == ["snap"]:
            self._send(b"J", self.frame(self._ts() / 1e6))
            ans = "ok snap"
        elif parts == ["ping"]:
            ans = "ok pong"
        self._send(b"A", ans.encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--junk", type=float, default=0.0, help="probability of garbage / corrupted packet per frame")
    a = ap.parse_args()
    cam = FakeCam(a.fps, a.junk).start()
    print(f"Fake ESP32-CAM on {cam.port}   ->   camera_url = \"serial:{cam.port}\"   (Ctrl-C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
