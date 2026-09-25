"""
vision.py - camera -> objects on the belt (id, position in cm, speed in cm/s).

Modes (config.VISION["mode"]):
  "bgsub" read the MJPEG stream of the ESP32-CAM, detect moving objects with background
          subtraction (OpenCV MOG2) + contours, track them with a small centroid tracker.
          No GPU/ML runtime needed - deliberately light enough for a Pi-class board (no CUDA,
          no PyTorch). It only tells you THAT something moved and roughly its size, not what it
          is (no class label) - fine for this twin, which only needs position/speed.
  "sim"   no camera: fake objects that ride on the measured belt speed (to test the twin)
  "off"   vision disabled

All modes feed the same pipeline (VisionBase.process), which
  * converts pixels to belt centimetres using the calibrated ROI,
  * keeps a track per object and fits its speed,
  * predicts where the object should be from the ENCODER travel (used by the divergence rule),
  * writes cam_* variables, state.objects and an annotated JPEG for the browser,
  * publishes the result on MQTT (config.VISION["publish_topic"]).

To add a new kind of measurement, edit VisionBase.process().
"""
import json
import re
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests

import config as cfg
from state import state

V = cfg.VISION
L = cfg.BELT_LENGTH_CM


# ── calibration helpers (read live from the "Camera calibration" controls) ──────────────
def roi():
    p, d = state.params, V["belt_roi_px"]
    return (p.get("roi_x1", d[0]), p.get("roi_y1", d[1]), p.get("roi_x2", d[2]), p.get("roi_y2", d[3]))


def flipped():
    return bool(state.params.get("flip_x", V["flip_x"]))


def px_to_cm(cx):
    x1, _, x2, _ = roi()
    cm = (cx - x1) / max(1.0, (x2 - x1)) * L
    return L - cm if flipped() else cm


def cm_to_px(cm):
    x1, _, x2, _ = roi()
    frac = (L - cm if flipped() else cm) / L
    return x1 + frac * (x2 - x1)


# ── placeholder + overlay ───────────────────────────────────────────────────────────────
def placeholder_jpeg(text):
    img = np.full((360, 640, 3), (236, 239, 243), np.uint8)
    cv2.putText(img, text, (30, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (90, 100, 115), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def draw_overlay(img, objs):
    x1, y1, x2, y2 = [int(v) for v in roi()]
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 140, 20), 1)
    for cm in range(0, int(L) + 1, 10):
        px = int(cm_to_px(cm))
        cv2.line(img, (px, y1), (px, y1 + 12), (200, 140, 20), 1)
        cv2.putText(img, str(cm), (px + 2, y1 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 140, 20), 1, cv2.LINE_AA)

    # ground-truth calibration mark (physical reference glued on the belt, see config.CALIBRATION_MARK):
    # drawn where the ROI says it SHOULD be - line the ROI sliders up until this lands on the real mark.
    mk = cfg.CALIBRATION_MARK
    mx1, mx2 = int(cm_to_px(mk["x_start_cm"])), int(cm_to_px(mk["x_start_cm"] + mk["width_cm"]))
    my = y1 - 10
    cv2.rectangle(img, (min(mx1, mx2), my - 8), (max(mx1, mx2), my), (255, 210, 0), 1, cv2.LINE_AA)
    for px in (mx1, mx2):
        cv2.line(img, (px, y1), (px, y2), (255, 210, 0), 1, cv2.LINE_4)
    cv2.putText(img, "ref mark", (min(mx1, mx2), my - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 210, 0), 1, cv2.LINE_AA)

    for o in objs:
        bx1, by1, bx2, by2 = [int(v) for v in o["box"]]
        col = (50, 50, 220) if o.get("diverging") else (90, 190, 60)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), col, 2)
        spd = "" if o["speed_cm_s"] is None else f"  {o['speed_cm_s']:+.1f}cm/s"
        cv2.putText(img, f"#{o['id']} {o['label']} {o['x_cm']:.1f}cm{spd}", (bx1, max(14, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    return img


# ── tracking ───────────────────────────────────────────────────────────────────────────
class TrackBook:
    """Keeps a history per object id and derives speed + encoder-based expected position."""

    def __init__(self):
        self.tracks = {}

    def update(self, dets, t):
        travel = state.travel_at(t)          # encoder travel at the moment the frame was captured
        for d in dets:
            tr = self.tracks.get(d["tid"])
            if tr is None:
                tr = dict(id=d["tid"], x0=d["x_cm"], travel0=travel, t0=t, hist=deque())
                self.tracks[d["tid"]] = tr
            tr.update(label=d["label"], x=d["x_cm"], conf=d.get("conf", 1.0), box=d["box"], last=t)
            tr["hist"].append((t, d["x_cm"]))
            while tr["hist"] and t - tr["hist"][0][0] > V["speed_window_s"]:
                tr["hist"].popleft()
        for k in [k for k, tr in self.tracks.items() if t - tr["last"] > V["lost_after_s"]]:
            del self.tracks[k]

        out = []
        for tr in self.tracks.values():
            if tr["last"] != t:                       # not visible in this frame
                continue
            speed = None
            if len(tr["hist"]) >= 3 and tr["hist"][-1][0] - tr["hist"][0][0] >= 0.4:
                ts = np.array([h[0] for h in tr["hist"]]) - tr["hist"][0][0]
                xs = np.array([h[1] for h in tr["hist"]])
                speed = float(np.polyfit(ts, xs, 1)[0])
            age = t - tr["t0"]
            expected = tr["x0"] + (travel - tr["travel0"])
            err = (tr["x"] - expected) if (age >= 0.5 and 0 <= expected <= L) else None
            out.append(dict(id=tr["id"], label=tr["label"], x_cm=tr["x"], speed_cm_s=speed, expected_cm=expected,
                            err_cm=err, conf=tr["conf"], age_s=age, box=tr["box"], diverging=False))
        return out


class VisionBase(threading.Thread):
    def __init__(self, bridge):
        super().__init__(daemon=True, name="vision")
        self.bridge = bridge
        self.book = TrackBook()
        self._stamps = deque(maxlen=15)
        self._last_pub = 0.0

    def run(self):
        try:
            self.loop()
        except Exception as e:                       # never die silently
            import traceback
            traceback.print_exc()
            state.vision_status = f"error: {e}"
            state.log(f"Vision stopped: {e}", "alert")

    def process(self, dets, t):
        objs = self.book.update(dets, t)
        state.objects = objs
        state.set_value("cam_objects", len(objs))

        speeds = [o["speed_cm_s"] for o in objs if o["speed_cm_s"] is not None and o["age_s"] >= 0.6]
        if speeds:
            state.set_value("cam_belt_speed_cm_s", float(np.median(speeds)))

        state.set_value("cam_lag_ms", max(0.0, (time.time() - t) * 1000.0))
        self._stamps.append(t)
        if len(self._stamps) >= 2 and self._stamps[-1] > self._stamps[0]:
            state.set_value("cam_fps", (len(self._stamps) - 1) / (self._stamps[-1] - self._stamps[0]))

        if self.bridge and t - self._last_pub >= 1.0 / V["publish_hz"]:
            self._last_pub = t
            payload = dict(ts=t, belt_speed_cm_s=state.get("cam_belt_speed_cm_s"), objects=[
                dict(id=o["id"], label=o["label"], x_cm=round(o["x_cm"], 2),
                     speed_cm_s=None if o["speed_cm_s"] is None else round(o["speed_cm_s"], 2)) for o in objs])
            self.bridge.publish(V["publish_topic"], json.dumps(payload), qos=0, quiet=True)
        return objs

    @staticmethod
    def show(frame, objs):
        frame = draw_overlay(frame, objs)
        if frame.shape[1] > 800:
            s = 800.0 / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        state.jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


# ── real camera + background subtraction ─────────────────────────────────────────────────
def mjpeg_frames(resp):
    """Parse a multipart/x-mixed-replace stream by hand so the per-frame X-Timestamp header is available.
    Yields (jpeg_bytes, camera_timestamp_seconds_or_None)."""
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=2048):   # small: must not wait to fill a big buffer
        buf += chunk
        while True:
            h = buf.find(b"\r\n\r\n")
            if h < 0:
                break
            head = bytes(buf[:h]).decode("latin-1")
            m = re.search(r"Content-Length:\s*(\d+)", head, re.I)
            if not m:
                del buf[:h + 4]
                continue
            n = int(m.group(1))
            if len(buf) < h + 4 + n:
                break                                   # frame not complete yet
            jpg = bytes(buf[h + 4:h + 4 + n])
            del buf[:h + 4 + n]
            ts = re.search(r"X-Timestamp:\s*([\d.]+)", head, re.I)
            yield jpg, (float(ts.group(1)) if ts else None)


class ClockSync:
    """Maps the ESP32-CAM clock to the PC clock.
    The camera timestamp is only meaningful as a difference (time since boot), so we estimate the offset as the
    smallest (arrival - camera_ts) seen recently, i.e. the frames with the least network delay.
    Result: frame times without WiFi jitter, at the moment of capture (+ the minimum transport delay)."""

    def __init__(self):
        self.samples = deque(maxlen=300)
        self.last_ts = None

    def to_pc(self, ts_cam, arrival):
        if self.last_ts is not None and ts_cam < self.last_ts - 1.0:      # camera rebooted -> new timebase
            self.samples.clear()
        self.last_ts = ts_cam
        self.samples.append(arrival - ts_cam)
        return ts_cam + min(self.samples)


class CameraReader(threading.Thread):
    """Keeps only the newest frame so slow inference never adds latency.
    http(s) URLs use the hand-written MJPEG parser (uses X-Timestamp); anything else (file, 0 = webcam) uses OpenCV."""

    def __init__(self, url):
        super().__init__(daemon=True, name="camera-reader")
        self.url = int(url) if str(url).isdigit() else url
        self.frame, self.t, self.seq = None, 0.0, 0

    def _push(self, frame, t):
        self.frame, self.t, self.seq = frame, t, self.seq + 1

    def run(self):
        http = isinstance(self.url, str) and self.url.startswith("http")
        while True:
            try:
                self._run_http() if http else self._run_cv()
            except Exception as e:
                state.vision_status = f"camera error: {type(e).__name__}"
            time.sleep(1.5)

    def _run_http(self):
        sync = ClockSync()
        with requests.get(self.url, stream=True, timeout=(4, 5)) as resp:
            resp.raise_for_status()
            for jpg, ts in mjpeg_frames(resp):
                arrival = time.time()
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:                       # ESP32-CAM sometimes sends a damaged JPEG
                    continue
                self._push(frame, sync.to_pc(ts, arrival) if ts is not None else arrival)

    def _run_cv(self):
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            state.vision_status = f"cannot open {self.url}"
            return
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self._push(frame, time.time())
        cap.release()


class CentroidTracker:
    """Assigns persistent ids to bounding boxes across frames by nearest-centroid matching (a
    minimal version of the classic pyimagesearch centroid tracker). No GPU/ML involved: this is
    what turns per-frame background-subtraction blobs into the same "track an id over time" shape
    the rest of the pipeline (TrackBook) already expects from any detector."""

    def __init__(self, max_missed, max_dist_px):
        self.max_missed = max_missed
        self.max_dist_px = max_dist_px
        self.next_id = 1
        self.objects = {}      # id -> box (bx1, by1, bx2, by2)
        self.missed = {}       # id -> consecutive frames without a match

    @staticmethod
    def _centroid(box):
        bx1, by1, bx2, by2 = box
        return (bx1 + bx2) / 2, (by1 + by2) / 2

    def _drop_stale(self, ids):
        for tid in ids:
            self.missed[tid] += 1
            if self.missed[tid] > self.max_missed:
                del self.objects[tid]
                del self.missed[tid]

    def update(self, boxes):
        if not boxes:
            self._drop_stale(list(self.missed))
            return dict(self.objects)
        if not self.objects:
            for box in boxes:
                self.objects[self.next_id], self.missed[self.next_id] = box, 0
                self.next_id += 1
            return dict(self.objects)

        ids = list(self.objects)
        prev_c = np.array([self._centroid(self.objects[i]) for i in ids])
        new_c = np.array([self._centroid(b) for b in boxes])
        dist = np.linalg.norm(prev_c[:, None, :] - new_c[None, :, :], axis=2)     # [prev, new]

        used_rows, used_cols = set(), set()
        for r, c in np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]:
            if r in used_rows or c in used_cols or dist[r, c] > self.max_dist_px:
                continue                                    # closest pairs first, greedy matching
            tid = ids[r]
            self.objects[tid], self.missed[tid] = boxes[c], 0
            used_rows.add(r)
            used_cols.add(c)

        self._drop_stale([ids[r] for r in range(len(ids)) if r not in used_rows])
        for c, box in enumerate(boxes):
            if c not in used_cols:
                self.objects[self.next_id], self.missed[self.next_id] = box, 0
                self.next_id += 1
        return dict(self.objects)


class BgSubVision(VisionBase):
    def loop(self):
        reader = CameraReader(V["camera_url"])
        reader.start()
        state.vision_status = f"connecting to {V['camera_url']}"

        backsub = cv2.createBackgroundSubtractorMOG2(
            history=V["bg_history"], varThreshold=V["bg_var_threshold"], detectShadows=True)
        tracker = CentroidTracker(max_missed=V["track_max_missed"], max_dist_px=V["track_max_dist_px"])
        kernel = np.ones((5, 5), np.uint8)

        last_seq, last_frame_t = 0, time.time()
        objs = []
        while True:
            if reader.seq == last_seq:
                if time.time() - last_frame_t > 2.0 and reader.seq > 0:
                    state.vision_status = "no frames"
                    state.objects = []
                time.sleep(0.005)
                continue
            frame, t, last_seq = reader.frame.copy(), reader.t, reader.seq
            last_frame_t = time.time()
            state.vision_status = "streaming"

            dets = self._detect(frame, backsub, tracker, kernel)
            objs = self.process(dets, t)
            self.show(frame, objs)

    @staticmethod
    def _detect(frame, backsub, tracker, kernel):
        x1, y1, x2, y2 = [int(v) for v in roi()]
        belt = frame[y1:y2, x1:x2]
        if belt.size == 0:
            return []

        mask = backsub.apply(belt, learningRate=V["bg_learning_rate"])
        mask[mask == 127] = 0                              # drop MOG2's "shadow" pixels, keep solid foreground only
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)    # remove speckle noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)   # fill holes inside blobs

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if V["min_area_px"] <= area <= V["max_area_px"]:
                bx, by, bw, bh = cv2.boundingRect(c)
                boxes.append((bx + x1, by + y1, bx + x1 + bw, by + y1 + bh))    # back to full-frame coords

        dets = []
        for tid, (bx1, by1, bx2, by2) in tracker.update(boxes).items():
            dets.append(dict(tid=tid, label="object", x_cm=float(px_to_cm((bx1 + bx2) / 2)),
                              conf=1.0, box=(bx1, by1, bx2, by2)))
        return dets


# ── simulated vision (no camera needed) ────────────────────────────────────────────────
class SimVision(VisionBase):
    def loop(self):
        state.vision_status = "simulated"
        objs, next_id, last_spawn, prev = [], 1, 0.0, time.time()
        while True:
            time.sleep(0.05)
            now = time.time()
            dt, prev = now - prev, now
            v = state.get("belt_speed_cm_s") or 0.0
            for o in objs:
                o["x"] += v * V["sim_slip"] * dt
            objs = [o for o in objs if 0 <= o["x"] <= L]
            if now - last_spawn > V["sim_spawn_every_s"] and len(objs) < 3 and abs(v) > 1.0:
                objs.append(dict(id=next_id, x=2.0 if v > 0 else L - 2.0))
                next_id, last_spawn = next_id + 1, now

            _, y1, _, y2 = roi()
            cy = (y1 + y2) / 2
            dets = [dict(tid=o["id"], label="box", x_cm=o["x"], conf=1.0,
                         box=(cm_to_px(o["x"]) - 25, cy - 25, cm_to_px(o["x"]) + 25, cy + 25)) for o in objs]
            out = self.process(dets, now)

            img = np.full((480, 640, 3), (236, 239, 243), np.uint8)
            rx1, ry1, rx2, ry2 = [int(x) for x in roi()]
            cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (95, 105, 120), -1)
            for o in out:
                b = [int(x) for x in o["box"]]
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (60, 130, 220), -1)
            self.show(img, out)


def start(bridge):
    mode = V["mode"]
    state.jpeg = placeholder_jpeg("No camera signal" if mode != "off" else "Vision disabled")
    if mode == "off":
        state.vision_status = "off"
        return
    worker = SimVision(bridge) if mode == "sim" else BgSubVision(bridge)
    worker.start()
