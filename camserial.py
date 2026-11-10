"""
camserial.py - ESP32-CAM over USB serial (no WiFi). PC side of firmware/esp32_cam_serial/.

Why: WiFi in the lab is too slow/unreliable for the image stream; the ESP32-CAM-MB programmer board already gives
a USB-serial link (CH340, up to 2 Mbaud ~ 200 KB/s ~ 15+ fps of the 640x160 band).

Packet format (ESP32 -> PC, little-endian), must match the firmware:
    magic   4 B   A5 5A 43 4D               ("\\xa5\\x5aCM")
    type    1 B   'J' JPEG frame | 'L' log text | 'A' command answer | 'R' result JSON | 'F' 64x64 RGB face
                  ('R' and 'F' come from the classifier firmware, firmware/esp32_cam_classifier)
    seq     2 B   uint16, increments per packet
    ts_us   8 B   uint64, capture time in microseconds since the ESP32 booted (frames), send time otherwise
    len     4 B   uint32 payload length
    payload len B
    check   2 B   Fletcher-16 of type..payload (everything after the magic)
Anything that is not a valid packet (ROM boot messages, a corrupted frame) is skipped by re-syncing on the magic.

Commands (PC -> ESP32): one text line each, answered with an 'A' packet ("ok ..." / "err ...")
    set <var> <int>   same vars as the HTTP /control of the WiFi firmware: crop quality brightness contrast saturation
                      vflip hmirror exposure_ctrl aec_value gain_ctrl agc_gain framesize
    fps <n>           max frames per second sent (0 = as fast as possible)
    stream <0|1>      start / stop sending frames
    snap              send one frame now (also when the stream is stopped)
    ping              -> "ok pong"

Opening a CH340 port can pull DTR/RTS, which on the ESP32-CAM-MB are wired to EN / IO0 (reset / bootloader). The port
is opened with both released; if no frame arrives, SerialCam pulses a clean reset (EN low with IO0 high).

Usage:
    cam = camserial.SerialCam("/dev/ttyUSB0")          # starts the reader thread
    cam.set("saturation", 2)
    frame = cam.wait_frame(timeout=2)                   # Frame(jpeg, ts_cam_s, seq, arrival)  or None
    img = frame.decode()                                # BGR numpy image
URL form used by config.VISION["camera_url"] / capture.py:  "serial:/dev/ttyUSB0"  or  "serial:/dev/ttyUSB0@921600"

Diagnostics (close app.py / capture.py first - one program per port):
    python camserial.py /dev/ttyUSB0                       # fps, KB/s, frame size, bad packets, firmware timing
    python camserial.py /dev/ttyUSB0 --set raw=1 quality=20 --save /tmp/frame.jpg
"""
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import serial

MAGIC = b"\xa5\x5aCM"
HEAD = struct.Struct("<cHQI")              # type, seq, ts_us, len    (after the magic)
MAX_PAYLOAD = 512 * 1024
DEFAULT_BAUD = 2_000_000


def fletcher16_reference(data):
    """Plain Fletcher-16 (mod 255), exactly what the firmware computes byte by byte."""
    a = b = 0
    for x in data:
        a = (a + x) % 255
        b = (b + a) % 255
    return (b << 8) | a


def fletcher16(data):
    """Same result as fletcher16_reference, vectorised: a = sum(x), b = sum of prefix sums = sum(x_i * (n - i))."""
    x = np.frombuffer(bytes(data), np.uint8).astype(np.int64)
    a = int(x.sum() % 255)
    b = int((x * np.arange(x.size, 0, -1, dtype=np.int64)).sum() % 255)
    return (b << 8) | a


def build_packet(ptype, seq, ts_us, payload):
    """Used by the tests / simulator; the firmware builds the same bytes in C."""
    body = HEAD.pack(ptype if isinstance(ptype, bytes) else ptype.encode(), seq & 0xFFFF, ts_us, len(payload)) + payload
    return MAGIC + body + struct.pack("<H", fletcher16(body))


def middle_third(img):
    """The belt band of a full frame (rows h/3 .. 2h/3) - the crop the raw-mode firmware does on the ESP32.
    Frames that are already a band (wider than 2:1) are returned unchanged."""
    h, w = img.shape[:2]
    return img if w >= 2 * h else img[h // 3: 2 * h // 3]


def parse_url(url):
    """'serial:/dev/ttyUSB0@921600' -> ('/dev/ttyUSB0', 921600)."""
    rest = url[len("serial:"):] if url.startswith("serial:") else url
    port, _, baud = rest.partition("@")
    return port, int(baud) if baud else DEFAULT_BAUD


@dataclass
class Frame:
    jpeg: bytes
    ts_cam: float          # seconds since the ESP32 booted (capture time)
    seq: int
    arrival: float         # PC time.time() when the last byte arrived

    def decode(self):
        import cv2
        import numpy as np
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)


class SerialCam:
    def __init__(self, port, baud=DEFAULT_BAUD, settings=None, on_log=None, fps=None, auto_reset_s=4.0,
                 stream=True, on_result=None):
        self.port, self.baud = port, baud
        self.settings = dict(settings or {})       # re-sent after every (re)connect / reboot
        self.fps = fps
        self.on_log = on_log or (lambda s: print(f"[esp32-cam] {s}"))
        self.auto_reset_s = auto_reset_s
        self.stream = stream                       # send "stream 1" on connect (preview frames)
        self.on_result = on_result                 # callback(dict) for every 'R' packet
        self.result = None                         # latest 'R' (dict, + "arrival")
        self.face = None                           # latest 'F' (64x64x3 RGB numpy)
        self.frame = None
        self.status = "connecting"
        self.stats = dict(frames=0, bad=0, bytes=0, fps=0.0, kbps=0.0)
        self._ser = None
        self._wlock = threading.Lock()
        self._cv = threading.Condition()
        self._answers = deque(maxlen=50)
        self._n_answers = 0
        self._stamps = deque(maxlen=30)
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camserial-{port}")
        self._thread.start()

    # ── public API ──────────────────────────────────────────────────────────────────────────
    def command(self, line, wait=1.0):
        """Send one command line; returns the answer text (or None if none arrived within `wait` s)."""
        with self._cv:
            n0 = self._n_answers
        if not self._write((line.strip() + "\n").encode()):
            return None
        t_end = time.time() + wait
        with self._cv:
            while self._n_answers == n0:
                left = t_end - time.time()
                if left <= 0:
                    return None
                self._cv.wait(timeout=left)
            return self._answers[-1]

    def set(self, var, val, wait=1.0):
        """Camera setting (same names as the HTTP /control). Remembered and re-applied after a reboot."""
        self.settings[var] = int(val)
        ans = self.command(f"set {var} {int(val)}", wait)
        if ans is None or not ans.startswith("ok"):
            self.on_log(f"set {var}={val}: {ans or 'no answer'}")
        return ans

    def wait_frame(self, timeout=2.0):
        """Block until a frame newer than the current one arrives; returns it, or None on timeout."""
        t_end = time.time() + timeout
        with self._cv:
            n0 = self.stats["frames"]
            while self.stats["frames"] == n0:
                left = t_end - time.time()
                if left <= 0:
                    return None
                self._cv.wait(timeout=left)
            return self.frame

    def reset(self):
        """Clean reboot of the ESP32 (EN pulse with IO0 released) through the MB board's RTS/DTR."""
        s = self._ser
        if s is None:
            return
        try:
            s.dtr = False            # IO0 high -> normal boot, not the bootloader
            s.rts = True             # EN low
            time.sleep(0.12)
            s.rts = False
        except (OSError, serial.SerialException):
            pass

    def close(self):
        self._stop = True
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    # ── internals ───────────────────────────────────────────────────────────────────────────
    def _write(self, data):
        s = self._ser
        if s is None:
            return False
        try:
            with self._wlock:
                s.write(data)
            return True
        except (OSError, serial.SerialException):
            return False

    def _open(self):
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = self.port, self.baud, 0.05
        try:
            s.dtr = False                            # do not touch EN / IO0 on open
            s.rts = False
        except (OSError, ValueError):
            pass
        s.open()
        return s

    def _configure(self):
        for var, val in self.settings.items():
            self._write(f"set {var} {int(val)}\n".encode())
        if self.fps is not None:
            self._write(f"fps {int(self.fps)}\n".encode())
        if self.stream:
            self._write(b"stream 1\n")

    def _run(self):
        while not self._stop:
            try:
                self._ser = self._open()
                self.status = "waiting for frames"
                self._configure()
                self._read_loop()
            except Exception as e:                   # noqa: BLE001 - port unplugged, closed under us, ...
                if self._stop:
                    return                           # close() was called: exit quietly
                self.status = f"serial error: {type(e).__name__}"
            finally:
                if self._ser:
                    try:
                        self._ser.close()
                    except Exception:
                        pass
                self._ser = None
            time.sleep(1.0)

    def _read_loop(self):
        buf = bytearray()
        last_rx_frame = time.time()
        resets = 0
        while not self._stop:
            chunk = self._ser.read(max(1, min(65536, self._ser.in_waiting or 1)))
            if chunk:
                buf += chunk
                self.stats["bytes"] += len(chunk)
                got = self._parse(buf)
                if got:
                    last_rx_frame = time.time()
                    resets = 0
            # no frames for a while: maybe stuck in the bootloader after the port was opened -> clean reset
            if self.auto_reset_s and time.time() - last_rx_frame > self.auto_reset_s * (1 + resets):
                self.status = "no frames - resetting ESP32"
                self.reset()
                resets += 1
                last_rx_frame = time.time()
                time.sleep(1.5)                      # boot time, then send the settings again
                self._configure()

    def _parse(self, buf):
        """Consume complete packets from buf. Returns True if at least one frame was received."""
        got_frame = False
        while True:
            i = buf.find(MAGIC)
            if i < 0:
                if len(buf) > 3:
                    del buf[:-3]                     # keep a possible partial magic
                return got_frame
            if i > 0:
                del buf[:i]
            if len(buf) < 4 + HEAD.size:
                return got_frame
            ptype, seq, ts_us, n = HEAD.unpack_from(buf, 4)
            if n > MAX_PAYLOAD or ptype not in (b"J", b"L", b"A", b"R", b"F"):
                self.stats["bad"] += 1
                del buf[:1]                          # false magic: resync
                continue
            end = 4 + HEAD.size + n + 2
            if len(buf) < end:
                return got_frame
            body = bytes(buf[4:end - 2])
            (chk,) = struct.unpack_from("<H", buf, end - 2)
            if fletcher16(body) != chk:
                self.stats["bad"] += 1
                del buf[:1]
                continue
            payload = body[HEAD.size:]
            del buf[:end]
            self._handle(ptype, seq, ts_us, payload)
            got_frame |= ptype in (b"J", b"R")         # results also mean "the camera is alive"

    def _handle(self, ptype, seq, ts_us, payload):
        now = time.time()
        if ptype == b"J":
            with self._cv:
                self.frame = Frame(payload, ts_us / 1e6, seq, now)
                self.stats["frames"] += 1
                self._cv.notify_all()
            self._stamps.append((now, len(payload)))
            if len(self._stamps) >= 2:
                dt = self._stamps[-1][0] - self._stamps[0][0]
                if dt > 0:
                    self.stats["fps"] = (len(self._stamps) - 1) / dt
                    self.stats["kbps"] = sum(b for _, b in list(self._stamps)[1:]) / dt / 1024
            self.status = "streaming"
        elif ptype == b"A":
            with self._cv:
                self._answers.append(payload.decode("latin-1"))
                self._n_answers += 1
                self._cv.notify_all()
        elif ptype == b"L":
            txt = payload.decode("latin-1").rstrip()
            if txt.startswith("ready"):              # (re)booted: it forgot the settings
                self._configure()
            self.on_log(txt)
        elif ptype == b"R":
            import json
            try:
                res = json.loads(payload.decode("latin-1"))
            except ValueError:
                self.on_log("bad result JSON: " + payload[:80].decode("latin-1"))
                return
            res["arrival"] = now
            with self._cv:
                self.result = res
                self._cv.notify_all()
            if self.on_result:
                self.on_result(res)
        elif ptype == b"F":
            side = int(round((len(payload) // 3) ** 0.5))
            if side * side * 3 == len(payload):
                self.face = np.frombuffer(payload, np.uint8).reshape(side, side, 3).copy()


# One SerialCam per port, shared by vision.py and controls.py (only one process/thread may own the port)
_shared = {}
_shared_lock = threading.Lock()


def shared(url, **kw):
    port, baud = parse_url(url)
    with _shared_lock:
        if port not in _shared:
            _shared[port] = SerialCam(port, baud, **kw)
        return _shared[port]


def main():
    import argparse
    ap = argparse.ArgumentParser(description="ESP32-CAM serial link diagnostics.")
    ap.add_argument("port", help="e.g. /dev/ttyUSB0 or serial:/dev/ttyUSB0@921600")
    ap.add_argument("--baud", type=int)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--set", nargs="*", default=[], metavar="VAR=VAL", help="camera settings, e.g. raw=0 quality=15")
    ap.add_argument("--fps", type=int, help="max fps asked to the firmware (default: firmware's)")
    ap.add_argument("--save", help="save the last frame here (JPEG as received) to judge the colours")
    a = ap.parse_args()
    port, baud = parse_url(a.port)
    settings = {k: int(v) for k, v in (x.split("=", 1) for x in a.set)}
    cam = SerialCam(port, a.baud or baud, settings=settings, fps=a.fps,
                    on_log=lambda m: print(f"  [esp32] {m}"))
    t0, last = time.time(), 0
    while time.time() - t0 < a.seconds:
        time.sleep(1.0)
        st, f = cam.stats, cam.frame
        res = ""
        if f is not None:
            img = f.decode()
            res = "decode FAILED" if img is None else f"{img.shape[1]}x{img.shape[0]}"
        n = st["frames"] - last
        last = st["frames"]
        print(f"{time.time() - t0:5.1f}s  {cam.status:<22} frames/s {n:3d}  KB/s {st['kbps']:6.1f}  "
              f"frame {len(f.jpeg) / 1024 if f else 0:5.1f} KB {res:>9}  bad packets {st['bad']}")
    if a.save and cam.frame is not None:
        with open(a.save, "wb") as fh:
            fh.write(cam.frame.jpeg)
        print(f"saved {a.save}")
    print("Reading it: 'cap' = sensor capture, 'enc' = software JPEG (raw mode only), 'send' = serial transfer.\n"
          "  KB/s near 190 -> the cable is the limit (raise quality number / smaller frame / raw mode);\n"
          "  KB/s low but 'send' high -> link problems (try @921600); bad packets > 0 -> noisy link / wrong baud;\n"
          "  high fps here but slow on the twin page -> the bottleneck is after the camera (PC CPU, or the browser\n"
          "  watching the page over the lab WiFi).")
    cam.close()


if __name__ == "__main__":
    main()
