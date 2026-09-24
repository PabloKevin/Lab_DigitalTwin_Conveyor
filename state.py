"""
state.py - the single shared object that every module reads/writes (thread-safe).

    from state import state
    state.set_value("rpm", 120.0)
    state.get("rpm")            # latest value, or None if older than STALE_AFTER_S
"""
import threading
import time
from collections import deque

import config as cfg


def is_num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class TwinState:
    def __init__(self):
        self.lock = threading.RLock()
        self.started = time.time()

        # variables
        self.values = {}                       # id -> latest value
        self.stamps = {}                       # id -> time of last update
        n = int(cfg.HISTORY_SECONDS / cfg.HISTORY_DT) + 20
        self.history = {v["id"]: deque(maxlen=n) for v in cfg.VARIABLES}
        self._hist_t = {}

        # controls / twin parameters
        self.controls = {}                     # control id -> typed value (float / str / bool)
        self.params = {}                       # twin parameters written by controls (model_var)
        self.sync = True                       # True = commands go to the real conveyor
        self.last_command = "—"

        # links
        self.mqtt_connected = False
        self.device_status = "unknown"         # online / offline / unknown (from STATUS_TOPIC)
        self.vision_status = "starting"

        # twin + vision
        self.model = {"rpm": 0.0}
        self.travel_real = 0.0                 # cm, integrated from encoder speed
        self.travel_model = 0.0                # cm, integrated from model speed
        self.objects = []                      # list of dicts published by vision.py
        self.jpeg = None                       # last annotated camera frame (bytes)

        # diagnostics
        self.divergences = {}
        self.events = deque(maxlen=200)
        self.raw = {}                          # topic -> (time, payload text)

    # ── variables ──────────────────────────────────────────────────────────
    def set_value(self, vid, value, t=None):
        t = t or time.time()
        with self.lock:
            self.values[vid] = value
            self.stamps[vid] = t
            if is_num(value):
                h = self.history.setdefault(vid, deque(maxlen=int(cfg.HISTORY_SECONDS / cfg.HISTORY_DT) + 20))
                if t - self._hist_t.get(vid, 0) >= cfg.HISTORY_DT:
                    h.append((t, float(value)))
                    self._hist_t[vid] = t

    def age(self, vid):
        t = self.stamps.get(vid)
        return None if t is None else time.time() - t

    def get(self, vid, fresh=True):
        """Latest value. With fresh=True returns None when the value is older than STALE_AFTER_S."""
        v = self.values.get(vid)
        if v is None:
            return None
        if fresh and time.time() - self.stamps[vid] > cfg.STALE_AFTER_S:
            return None
        return v

    def snapshot(self):
        """dict of all fresh values {id: value}"""
        now = time.time()
        with self.lock:
            return {k: v for k, v in self.values.items() if now - self.stamps[k] <= cfg.STALE_AFTER_S}

    def series(self, vid):
        with self.lock:
            return list(self.history.get(vid, []))

    # ── events ─────────────────────────────────────────────────────────────
    def log(self, msg, level="info"):
        self.events.appendleft((time.time(), level, msg))
        print(f"[{level}] {msg}")


state = TwinState()
