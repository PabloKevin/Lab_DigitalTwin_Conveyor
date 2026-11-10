"""
orchestrator.py - the automatic sorting cycle (lab 03 state machine), run by the twin app.

    IDLE --Start--> RUN --cube seen before the stop line--> OBJECT_DETECTED
      --cube centre reaches the stop line (image centre)--> stop belt --> CLASSIFYING
   (Where the cube is comes from the camera or from the HC-SR04 ultrasonic sensor: config.CYCLE["position_source"],
    switchable on the page. With the sensor, the stops are sensor readings: CYCLE["ultrasonic"]["classify_cm"] and
    each class's "us_cm"; the camera is then only used to classify.)
      --N frames (params.json "decision" n_frames) -> class with the highest mean probability
        (CYCLE["uncertain"] = True brings back the "dudoso" rule of classifier.live.Voter)-->
          class in config.CYCLE["push"] --> TO_ARM: belt runs that class's cm (encoder) --> stop
                --> PUSHING: MQTT routine ("PUSH1"/"PUSH2"/"PUSH3") to the arm, wait for "IDLE",
                    then "HOME" (ARM["home_after_push"]), wait for "IDLE"
                --> VERIFY: is the cube still at the push point (camera or ultrasonic)?  no --> RUN
                    yes (push failed) --> CORRECTING: belt moves the cube back onto the push point --> PUSHING again
                    failed again (CYCLE["verify"]["attempts"]) --> REVERSE for failed_reverse_s --> RUN
          class in CYCLE["reverse_classes"] (other, dudoso) --> REVERSE: belt runs backwards until the cube has left
                the belt at the origin (reverse_cm of travel or reverse_s, whichever first) --> forward again, RUN
          any other class --> TO_END: belt runs clear_cm --> RUN (the cube rides to the end basket)
    any fault (telemetry or camera lost, belt not stopping / stalling, arm error / timeout / offline) or E-stop
      --> SAFE_STOP: belt stopped, arm STOP; stays there until "Reset" on the page.

Every cube gets a record (times, class, confidence, routine, latencies, deadline checks) appended to
config.CYCLE["log_csv"] and published as JSON on config.CYCLE["publish_topic"].

The logic only talks to a Plant (belt / arm / encoder / camera), so it runs the same against the real machine
(Live sync), against the twin model (Sandbox: belt commands only move the model, the arm is simulated) and in the
unit tests (tests/test_orchestrator.py, fake plant).
"""
import csv
import json
import os
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np

import config as cfg
from state import state

C, A = cfg.CYCLE, cfg.ARM
ROOT = os.path.dirname(os.path.abspath(__file__))
STATES = ["IDLE", "RUN", "OBJECT_DETECTED", "CLASSIFYING", "TO_ARM", "PUSHING", "VERIFY", "CORRECTING", "REVERSE",
          "TO_END", "SAFE_STOP"]


class Plant:
    """Real machine (Live sync) or twin model (Sandbox), as seen by the orchestrator."""

    def __init__(self, bridge, mqtt):
        self.bridge, self.mqtt = bridge, mqtt

    def live(self):
        return state.sync

    def belt(self, direction, speed_pct=None, force=False):
        import controls
        if speed_pct is not None:
            controls.apply(controls.BY_ID["speed"], float(speed_pct), self.bridge, force=force)
        controls.apply(controls.BY_ID["direction"], direction, self.bridge, force=force)

    def travel(self):
        return state.travel_real if state.sync else state.travel_model

    def speed(self):
        return state.get("belt_speed_cm_s" if state.sync else "model_belt_speed_cm_s")

    def speed_lag(self):
        """Lag (s) of the reported speed behind the real belt: the Arduino's EMA; the Sandbox model has none."""
        return C["speed_lag_s"] if state.sync else 0.0

    def telemetry_age(self):
        return state.age("rpm")

    def frame(self):
        return state.cam_frame

    def position_source(self):
        return state.params.get("cycle_source", C["position_source"])

    def us_sample(self):
        """(time of the reading, distance cm or -1) of the latest ultrasonic telemetry, or None."""
        t = state.stamps.get("us_distance_cm")
        return None if t is None else (t, state.values.get("us_distance_cm"))

    def px_to_cm(self, cx):
        import vision
        return vision.px_to_cm(cx)

    def geometry(self):
        """(px per cm along the belt, +1 if forward moves to +x in the image else -1) from the camera calibration."""
        import vision
        x1, _, x2, _ = vision.roi()
        return max(1.0, x2 - x1) / cfg.BELT_LENGTH_CM, (-1 if vision.flipped() else 1)

    def send_arm(self, cmd):
        return self.mqtt.publish(A["cmd_topic"], cmd, qos=1) if state.sync else True

    def publish(self, payload):
        self.mqtt.publish(C["publish_topic"], json.dumps(payload), quiet=True)


class Orchestrator:
    def __init__(self, plant, model=None, params=None, clock=time.time):
        from classifier import preprocess as pp
        from classifier.live import Voter
        self.pp = pp
        self.p = params or pp.load_params()
        self.classes = self.p["model"]["classes"]
        if model is None:
            from classifier.npmodel import NpModel
            model = NpModel(os.path.join(ROOT, C["model"]))
        self.model = model
        d = self.p["decision"]
        self.new_voter = lambda: Voter(self.classes, d["n_frames"], d["min_agree"], d["tau"],
                                       uncertain=C.get("uncertain", False))
        self._seen_probs = []                      # every face classified for the current cube (timeout fallback)
        self.plant, self.clock = plant, clock
        self.lock = threading.RLock()
        self.state, self.since, self.msg = "IDLE", clock(), "press Start"
        self.counts = Counter()
        self.misses = Counter()                    # deadline misses per deadline
        self.cubes = deque(maxlen=30)
        self.cube = None                           # record of the cube being handled
        self.next_id = 1
        self.arm_online, self.arm_last, self._arm_q = None, "", deque()
        self.last_seq, self.last_frame_wall, self.last_det = None, clock(), None
        self.face = self.probs = self.vote = None
        self.t_moving = clock()
        self.us = None                             # filtered ultrasonic distance (cm), None = nothing / no data
        self._us_hist, self._us_t = deque(maxlen=C["ultrasonic"]["median_n"]), None
        self._dir, self._last_travel, self.free_x = "S", None, None   # for the cube position on the twin belt
        self.odo, self._move_dir, self._t_stop = 0.0, "F", -1e9        # belt odometer (see _odometer)
        self._odo_n, self._was_moving, self._last_lead = 0.0, False, 0.0
        self._t_run_start = -1e9
        self.calib = deque(self._load_calib(), maxlen=10)   # camera/encoder speed measurements (see _calib_finish)
        self.last_blobs, self.last_x_stop, self._ver = [], None, []
        self._thread = None

    # ── commands from the page (thread-safe) ───────────────────────────────────────────────
    def start(self):
        with self.lock:
            if self.state != "IDLE":
                return f"cannot start from {self.state}" + (" (press Reset)" if self.state == "SAFE_STOP" else "")
            now = self.clock()
            self.last_frame_wall = self.t_moving = now
            self._belt("F", C["run_speed_pct"])
            self._goto("RUN", "cycle started")
            return "started"

    def stop(self):
        with self.lock:
            if self.state in ("IDLE", "SAFE_STOP"):
                return "already stopped"
            self._belt("S")
            if self.cube:
                self._finish("stopped by operator")
            self._goto("IDLE", "stopped by operator")
            return "stopped"

    def reset(self):
        with self.lock:
            if self.state != "SAFE_STOP":
                return "nothing to reset"
            self._goto("IDLE", "reset by operator")
            return "reset"

    def estop(self):
        with self.lock:
            if self.state != "SAFE_STOP":
                self._fault("E-stop")

    def on_arm_status(self, text):                # MQTT thread
        self._arm_q.append((self.clock(), text.strip()))

    def on_arm_online(self, text):
        self.arm_online = text.strip() == "1"

    # ── main loop ───────────────────────────────────────────────────────────────────────────
    def run_forever(self, period=0.05):
        while True:
            try:
                self.step()
            except Exception as e:                 # noqa: BLE001 - never die silently: stop the machine
                import traceback
                traceback.print_exc()
                with self.lock:
                    self._fault(f"orchestrator error: {e}")
            time.sleep(period)

    def start_thread(self):
        self._thread = threading.Thread(target=self.run_forever, daemon=True, name="orchestrator")
        self._thread.start()

    def step(self):
        with self.lock:
            now = self.clock()
            det = self._new_detection(now)
            self._us_update(now)
            sp = self.plant.speed()
            self._track_position(det, self._odometer(now))
            if sp is not None and abs(sp) >= C["stopped_cm_s"]:
                self.t_moving = now
            self._arm_events(now)
            self._calib_sample(now, det, sp)
            if self.state not in ("IDLE", "SAFE_STOP"):
                age = self.plant.telemetry_age()
                if self.plant.live() and (age is None or age > C["telemetry_timeout_s"]):
                    return self._fault("conveyor telemetry lost")
                if self.state in ("RUN", "OBJECT_DETECTED", "CLASSIFYING") and \
                        now - self.last_frame_wall > C["camera_timeout_s"]:
                    return self._fault("no camera frames")
            getattr(self, "_s_" + self.state.lower())(now, det, sp)

    # ── perception ──────────────────────────────────────────────────────────────────────────
    def _new_detection(self, now):
        """Newest frame -> the cube to follow: among all squares in view (complete or entering), the one closest to
        the stop line on the upstream side (cubes already more than overshoot_px past the line are ignored - e.g. the
        one just classified, riding away). Its upright face crop is computed (re-running the detector in a window
        around it if the detector's own best square is a different cube)."""
        fr = self.plant.frame()
        if fr is None or fr[2] == self.last_seq:
            return None
        frame, t_cap, self.last_seq = fr
        self.last_frame_wall = now
        d = self.pp.process(frame, self.p)
        ppcm, sign = self.plant.geometry()
        H, W = frame.shape[:2]
        x_stop = W * C["stop_x_frac"]
        self.last_x_stop = x_stop
        self.last_blobs = [((b["bbox"][0] + b["bbox"][2]) / 2, (b["bbox"][1] + b["bbox"][3]) / 2)
                           for b in d.blobs if b["why"] in ("", "edge", "rect", "corners")]   # for VERIFY
        cands = []
        for b in d.blobs:
            if b["why"] in ("", "edge"):
                x1, y1, x2, y2 = b["bbox"]
                sv = sign * ((x1 + x2) / 2 - x_stop)
                if sv <= C["overshoot_px"]:
                    cands.append((sv, b))
        if not cands:
            self.last_det = None
            return dict(status="none", t=t_cap)
        sv, b = max(cands, key=lambda c: c[0])
        x1, y1, x2, y2 = b["bbox"]
        crop = corners = None
        if b["why"] == "":
            if d.status == "ok" and tuple(d.bbox) == tuple(b["bbox"]):
                crop, corners = d.crop, d.corners
            else:                                   # another cube is the detector's "best": look only at ours
                m = 24
                q = dict(self.p, image=dict(self.p["image"], window=[max(0, x1 - m), max(0, y1 - m), min(W, x2 + m),
                                                                       min(H, y2 + m)]))
                dd = self.pp.process(frame, q)
                if dd.status == "ok":
                    crop, corners = dd.crop, dd.corners
        side_px = None
        if corners is not None:
            cc = np.asarray(corners, float)
            side_px = float(np.mean([np.linalg.norm(cc[i] - cc[(i + 1) % 4]) for i in range(4)]))
        self.last_det = dict(status="ok" if crop is not None else "partial", t=t_cap, cx=(x1 + x2) / 2, side_px=side_px,
                             cy=(y1 + y2) / 2, s=sv, x_stop=x_stop, ppcm=ppcm, crop=crop, corners=corners, bbox=b["bbox"])
        return self.last_det

    def lead_px(self, sp):
        ppcm = self.plant.geometry()[0]
        return abs(sp or 0.0) * C["stop_latency_s"] * ppcm

    def _us_update(self, now):
        """Median of the last readings; None when the sensor sees nothing or its data is stale."""
        smp = self.plant.us_sample()
        if smp is None or smp[1] is None or now - smp[0] > C["telemetry_timeout_s"]:
            self._us_hist.clear()
            self.us = None
            return
        if smp[0] != self._us_t:
            self._us_t = smp[0]
            if smp[1] < 0:                          # firmware: -1 = no echo (already debounced over 3 readings)
                self._us_hist.clear()
            else:
                self._us_hist.append(float(smp[1]))
        self.us = float(np.median(self._us_hist)) if self._us_hist else None

    def us_to_go(self, target_cm):
        """Belt travel (cm, + = still to come when moving forward) until the sensor reads target_cm."""
        if self.us is None:
            return None
        return -cfg.ULTRASONIC["facing"] * (self.us - target_cm)

    def _belt(self, direction, speed_pct=None, force=False):
        if direction in ("F", "R"):
            if self._dir != direction:
                self._t_run_start = self.clock()
            self._move_dir = direction
        elif self._dir != "S":
            self._t_stop = self.clock()
        self._dir = direction
        self.plant.belt(direction, speed_pct, force=force)

    def moving(self, now=None):
        """Belt commanded to run, or stopped less than coast_s ago (motor still stopping)."""
        now = self.clock() if now is None else now
        return self._dir != "S" or now - self._t_stop < C["coast_s"]

    def _odometer(self, now):
        """Signed belt travel (cm, + = forward) estimated from the encoder.
        The Arduino's speed is filtered (EMA, lag ~speed_lag_s; held 400 ms after the last pulse), so its integral
        lags the real belt while running and keeps growing for ~1 s after a real stop - that made the twin's cube (and
        the arm distances) run on after every stop. Here: filtered travel is counted only while the belt runs or
        coasts (coast_s after a stop command), plus lag * speed while moving (first-order lag compensation), frozen
        when it stops."""
        travel = self.plant.travel()
        moved = 0.0 if self._last_travel is None else abs(travel - self._last_travel)
        self._last_travel = travel
        sgn = 1.0 if self._move_dir == "F" else -1.0
        mv = self.moving(now)
        lead = sgn * self.plant.speed_lag() * abs(self.plant.speed() or 0.0)
        if mv:
            self._odo_n += sgn * moved
        elif self._was_moving:
            self._odo_n += self._last_lead           # bake the lag compensation in at the stop
        self._was_moving = mv
        self._last_lead = lead if mv else 0.0
        old, self.odo = self.odo, self._odo_n + self._last_lead
        return self.odo - old

    def _track_position(self, det, d):
        """Belt position (cm) of the cube being handled, for the twin's belt drawing: measured while the camera /
        sensor sees it, otherwise dead-reckoned with the odometer (which does not move while the belt is stopped)."""
        meas = None
        if self.plant.position_source() == "ultrasonic" or (self.cube and self.cube.get("src") == "ultrasonic"):
            if self.us is not None:
                meas = cfg.us_position_cm(self.us)
        elif det is not None:
            meas = self.plant.px_to_cm(det["cx"]) if det["status"] != "none" else None
            if self.cube and self.state not in ("OBJECT_DETECTED", "CLASSIFYING"):
                meas = None                         # after the decision the camera may see the NEXT cube
        if self.cube is not None:
            if meas is not None:
                self.cube["x_cm"] = meas
            elif self.cube.get("x_cm") is not None:
                self.cube["x_cm"] += d * self.speed_scale()
            self.free_x = None
        elif det is not None or self.plant.position_source() == "ultrasonic":
            self.free_x = meas

    def objects(self):
        """Cube(s) for the twin's belt drawing / objects table (same fields as vision's objects)."""
        with self.lock:
            sp = self.plant.speed()
            v = None if sp is None else ({"F": 1, "R": -1}.get(self._dir, 0) * abs(sp) * self.speed_scale()
                                         if self.moving() else 0.0)
            mk = lambda oid, label, x: dict(id=oid, label=label, x_cm=x, speed_cm_s=v, expected_cm=None, err_cm=None,
                                            diverging=False)
            if self.cube is not None and self.cube.get("x_cm") is not None:
                c = self.cube
                return [mk(c["id"], c.get("cls") or "cube", c["x_cm"])]
            if self.free_x is not None:
                return [mk("-", "cube", self.free_x)]
            return []

    # ── speed calibration: camera vs encoder ────────────────────────────────────────────────
    def _calib_sample(self, now, det, sp):
        """While a cube rides to the stop line at a steady speed, keep (capture time, x px, side px, encoder speed)."""
        if self.state != "OBJECT_DETECTED" or self.cube is None or det is None or det.get("side_px") is None:
            return
        if self._dir != "F" or now - self._t_run_start < C["calib_settle_s"] or not sp or abs(sp) < C["stopped_cm_s"]:
            return
        self.cube.setdefault("track", []).append((det["t"], det["cx"], det["side_px"], abs(sp)))

    def _calib_finish(self):
        """Cube reached the stop line: camera speed = slope of x px vs time / px per cm of the belt calibration.
        The cube's size in cm (with that calibration) is kept as a check of the calibration itself."""
        c = self.cube
        tr = np.asarray(c.pop("track", []), float)
        if len(tr) < 4 or tr[-1, 0] - tr[0, 0] < 0.8:
            return
        ppcm = self.plant.geometry()[0]
        cube_cm = float(np.median(tr[:, 2])) / ppcm
        slope = abs(np.polyfit(tr[:, 0] - tr[0, 0], tr[:, 1], 1)[0])      # px/s
        cam_v, enc_v = slope / ppcm, float(np.median(tr[:, 3]))
        if enc_v <= 0 or not (0.3 < cam_v / enc_v < 3.0):
            state.log(f"Cycle: speed calibration ignored (camera {cam_v:.1f} vs encoder {enc_v:.1f} cm/s)", "warn")
            return
        m = dict(t=time.time(), cam_v=round(cam_v, 2), enc_v=round(enc_v, 2), ratio=round(cam_v / enc_v, 3),
                 cube_cm=round(cube_cm, 2), n=len(tr))
        c.update(cam_v=m["cam_v"], enc_v=m["enc_v"], speed_ratio=m["ratio"], cube_cm=m["cube_cm"])
        self.calib.append(m)
        self._save_calib()

    def speed_scale(self):
        """Factor from encoder cm to real cm for the twin's cube motion."""
        sc = C.get("speed_scale", 1.0)
        if sc == "auto":
            return float(np.median([m["ratio"] for m in self.calib])) if self.calib else 1.0
        return float(sc)

    def calib_summary(self):
        if not self.calib:
            return None
        med = lambda k: float(np.median([m[k] for m in self.calib]))
        ratio = med("ratio")
        return dict(n=len(self.calib), cam_v=med("cam_v"), enc_v=med("enc_v"), ratio=ratio,
                    radius_cm=cfg.ROLLER_RADIUS_CM * ratio, cube_cm=med("cube_cm") if all("cube_cm" in m for m in
                                                                                       self.calib) else None,
                    ppcm=self.plant.geometry()[0], scale=self.speed_scale())

    def _calib_path(self):
        f = C.get("calib_file")
        return os.path.join(ROOT, f) if f else None

    def _load_calib(self):
        p = self._calib_path()
        try:
            with open(p) as f:
                return json.load(f)[-10:]
        except (OSError, TypeError, ValueError):
            return []

    def _save_calib(self):
        p = self._calib_path()
        if not p:
            return
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(list(self.calib), f, indent=1)
        except OSError as e:
            state.log(f"Speed calibration not saved: {e}", "warn")

    def _src(self):
        return self.cube.get("src", "camera") if self.cube else self.plant.position_source()

    # ── states ──────────────────────────────────────────────────────────────────────────────
    def _s_idle(self, now, det, sp):
        pass

    def _s_safe_stop(self, now, det, sp):
        pass

    def _stalled(self, now):
        return now - max(self.since, self.t_moving) > C["stall_s"]

    def _s_run(self, now, det, sp):
        if self._stalled(now):
            return self._fault("belt not moving (stall)")
        src = self.plant.position_source()
        if src == "ultrasonic":
            togo = self.us_to_go(C["ultrasonic"]["classify_cm"])
            seen = togo is not None and togo > C["ultrasonic"]["arm_margin_cm"]
        else:
            seen = det is not None and det["status"] != "none" and det["s"] < -C["arm_margin_px"]
        if seen:
            self.cube = dict(id=self.next_id, t_detect=now, src=src)
            self.next_id += 1
            self._goto("OBJECT_DETECTED", f"cube #{self.cube['id']} approaching ({src})")

    def _s_object_detected(self, now, det, sp):
        if self._stalled(now):
            return self._fault("belt not moving (stall)")
        if self._src() == "ultrasonic":
            return self._us_object_detected(now, sp)
        if det and det["status"] != "none":
            self.cube["t_seen"] = now
            if det["s"] >= -self.lead_px(sp):
                if det["s"] > C["overshoot_px"]:
                    self.counts["missed"] += 1
                    state.log(f"Cycle: cube #{self.cube['id']} passed the stop line before stopping "
                              f"({det['s']:.0f} px) - it goes to the end", "warn")
                    self.cube["cls"] = "missed"
                    self._finish("missed")
                    return self._goto("RUN", "missed cube")
                self.cube.update(x_err_px=det["s"])
                self._stop_to_classify(now)
        elif now - self.cube.get("t_seen", self.cube["t_detect"]) > 1.5:
            self.cube = None
            self._goto("RUN", "cube lost before the stop line")

    def _us_object_detected(self, now, sp):
        U = C["ultrasonic"]
        togo = self.us_to_go(U["classify_cm"])
        if togo is None:
            if now - self.cube.get("t_seen", self.cube["t_detect"]) > 1.5:
                self.cube = None
                self._goto("RUN", "cube lost by the ultrasonic sensor")
            return
        self.cube["t_seen"] = now
        if togo <= abs(sp or 0.0) * U["latency_s"]:
            if togo < -U["overshoot_cm"]:
                self.counts["missed"] += 1
                self.cube["cls"] = "missed"
                self._finish("missed")
                return self._goto("RUN", "missed cube (ultrasonic)")
            self.cube.update(us_err_cm=round(togo, 1))
            self._stop_to_classify(now)

    def _stop_to_classify(self, now):
        self._calib_finish()
        self._belt("S")
        self.cube.update(t_stop_cmd=now)
        self.vote, self.face, self.probs = self.new_voter(), None, None
        self._seen_probs = []
        self._goto("CLASSIFYING", f"cube #{self.cube['id']} at the classification point")

    def _s_classifying(self, now, det, sp):
        c = self.cube
        if "t_stopped" not in c and sp is not None and abs(sp) < C["stopped_cm_s"]:
            c["t_stopped"] = now
        t_settle = c["t_stop_cmd"] + C["settle_s"]
        if "t_stopped" not in c:
            if now - c["t_stop_cmd"] > C["stop_timeout_s"]:
                return self._fault("belt did not stop")
            return                                  # frames only once the belt really stopped (no motion blur)
        t_ready = max(t_settle, c["t_stopped"] + 0.2)
        if now < t_ready or det is None or det["t"] < t_ready - 0.05:
            if now - t_ready > C["classify_timeout_s"]:
                return self._timeout_decision(now)
            return
        if "x_final_px" not in c and "us_final_cm" not in c:   # where the cube really stopped (tune the latencies)
            if c.get("src") == "ultrasonic":
                togo = self.us_to_go(C["ultrasonic"]["classify_cm"])
                if togo is not None:
                    c["us_final_cm"] = round(togo, 1)
            elif "s" in det:                                  # status "none" has no position
                c["x_final_px"] = round(det["s"], 1)
        if det["status"] == "ok":
            self.face = cv2.cvtColor(det["crop"], cv2.COLOR_BGR2RGB)
            self.probs = self.model.predict(self.face)
            self._seen_probs.append(np.asarray(self.probs, float))
            v = self.vote.update(True, self.probs, (det["cx"], det["cy"]))
        else:
            v = self.vote.update(False)
        c["votes"] = v["votes"]
        if v["decision"]:
            return self._decide(now, v["decision"], v["reason"])
        if now - t_ready > C["classify_timeout_s"]:
            self._timeout_decision(now)

    def _timeout_decision(self, now):
        """Not enough frames in time: decide with the faces seen so far (highest mean probability)."""
        if C.get("uncertain", False):
            return self._decide(now, "dudoso", "classification timeout")
        if not self._seen_probs:
            return self._decide(now, "other", "timeout: no face seen")
        m = np.mean(self._seen_probs, axis=0)
        best = int(m.argmax())
        self._decide(now, self.classes[best], f"timeout: highest mean p {m[best]:.2f} of {len(self._seen_probs)} frames")

    def _decide(self, now, cls, why):
        c = self.cube
        c.update(t_decision=now, cls=cls, why=why,
                 conf=float(self.probs.max()) if self.probs is not None else None,
                 probs=[round(float(x), 3) for x in self.probs] if self.probs is not None else None)
        self.counts[cls] += 1
        push = C["push"].get(cls)
        routine = push["routine"] if push else None
        reverse = not push and cls in C["reverse_classes"]
        c.update(routine=routine, target_cm=push["cm"] if push else None, us_target_cm=(push or {}).get("us_cm"))
        dest = routine or ("back to the origin" if reverse else "to the end basket")
        self.plant.publish(dict(event="decision", id=c["id"], cls=cls, why=why, conf=c["conf"], routine=routine,
                                dest=dest))
        state.log(f"Cycle: cube #{c['id']} = {cls} ({why}) -> {dest}")
        if routine and self.plant.live() and self.arm_online is False:
            return self._fault("robotic arm offline")
        c["odo0"] = self.odo
        self._belt("R" if reverse else "F", C["run_speed_pct"])
        self._goto("TO_ARM" if routine else "REVERSE" if reverse else "TO_END", dest)

    def _s_to_arm(self, now, det, sp):
        if self._stalled(now):
            return self._fault("belt not moving (stall)")
        c = self.cube
        dist = abs(self.odo - c["odo0"])
        togo = None
        if c.get("src") == "ultrasonic" and c.get("us_target_cm") is not None:
            togo = self.us_to_go(c["us_target_cm"])
            if togo is not None:
                c["arm_by"] = "ultrasonic"
                c.pop("us_lost", None)
            elif not c.get("us_lost"):
                c["us_lost"] = True
                state.log(f"Cycle: ultrasonic lost cube #{c['id']} on its way to the arm - using the encoder", "warn")
        if togo is not None:
            arrived = togo <= abs(sp or 0.0) * C["ultrasonic"]["latency_s"]
        else:
            c["arm_by"] = "encoder"
            arrived = dist >= c["target_cm"] - abs(sp or 0.0) * C["stop_latency_s"]
        if arrived:
            self._belt("S")
            self._start_push(now)

    def _start_push(self, now):
        c = self.cube
        c["t_arm_stop_cmd"] = now
        c["push_attempts"] = c.get("push_attempts", 0) + 1
        self._push_sent = False
        self._push_seen_start = False
        self._homing = self._home_seen = False
        n = c["push_attempts"]
        self._goto("PUSHING", f"{c['routine']} for cube #{c['id']}" + (f" (attempt {n})" if n > 1 else ""))

    def _s_pushing(self, now, det, sp):
        c = self.cube
        if not self._push_sent:
            stopped = sp is not None and abs(sp) < C["stopped_cm_s"]
            if not stopped and now - c["t_arm_stop_cmd"] < C["stop_timeout_s"]:
                return                              # wait for the belt to stop under the arm
            if not stopped:
                return self._fault("belt did not stop at the arm")
            c["arm_travel_cm"] = round(abs(self.odo - c["odo0"]), 2)   # where it really stopped
            if self.us is not None:
                c["arm_us_cm"] = round(self.us, 1)
            if not self.plant.send_arm(c["routine"]):
                return self._fault("MQTT broker unreachable: push not sent")
            c["t_push_cmd"] = now
            self._push_sent = True
            if not self.plant.live():               # Sandbox: simulated arm
                self._sim_done_at = now + C["sim_push_s"]
            return
        if not self.plant.live() and now >= self._sim_done_at:
            return self._pushed(now) if self._homing else self._push_done(now)
        if self._homing:
            if now - c["t_home_cmd"] > A["timeout_s"]:
                return self._fault(f"arm did not reach HOME within {A['timeout_s']:.0f} s")
        elif now - c["t_push_cmd"] > A["timeout_s"]:
            return self._fault(f"arm did not finish {c['routine']} within {A['timeout_s']:.0f} s")

    def _push_done(self, now):
        """Push routine finished: send the arm HOME (and wait for it) before the belt moves again."""
        self.cube["t_arm_done"] = now
        if not A.get("home_after_push", True):
            return self._pushed(now)
        if not self.plant.send_arm(A["home_cmd"]):
            return self._fault("MQTT broker unreachable: HOME not sent")
        self.cube["t_home_cmd"] = now
        self._homing, self._home_seen = True, False
        self.msg = f"{self.cube['routine']} done, arm going HOME"
        if not self.plant.live():
            self._sim_done_at = now + C["sim_home_s"]

    def _pushed(self, now):
        """Arm back home: check that the cube really left the belt (unless verification is off)."""
        if self._homing:
            self.cube["t_home_done"] = now
        V = C["verify"]
        if V["on"] and (self.plant.live() or V["in_sandbox"]):
            self._ver = []
            return self._goto("VERIFY", f"is cube #{self.cube['id']} still on the belt?")
        self._push_ok("push done")

    def _push_ok(self, msg):
        self._finish("pushed")
        self._belt("F", C["run_speed_pct"])
        self._goto("RUN", msg)

    def _cube_at_arm(self):
        """-> (seen, togo_cm, x_cm): is there a cube at the push point, how far it still is from it (cm of forward
        belt travel, negative = past it) and its belt position. Ultrasonic cubes use the sensor, others the camera."""
        c, V = self.cube, C["verify"]
        if c.get("src") == "ultrasonic" and c.get("us_target_cm") is not None:
            togo = self.us_to_go(c["us_target_cm"])
            if togo is None:
                return False, None, None
            return abs(togo) <= V["tol_cm"], togo, cfg.us_position_cm(self.us)
        if not self.last_blobs or self.last_x_stop is None:
            return False, None, None
        ppcm, sign = self.plant.geometry()
        x_t = self.last_x_stop + sign * c["target_cm"] * ppcm
        cx, _ = min(self.last_blobs, key=lambda b: abs(b[0] - x_t))
        togo = sign * (x_t - cx) / ppcm
        return abs(togo) <= V["tol_cm"], togo, self.plant.px_to_cm(cx)

    def _s_verify(self, now, det, sp):
        V, c = C["verify"], self.cube
        t = now - self.since
        if t < V["settle_s"]:
            return                                  # arm out of the way, cube settled / fallen
        if det is not None or c.get("src") == "ultrasonic":
            seen, togo, x = self._cube_at_arm()
            self._ver.append((seen, togo))
            if seen and x is not None:
                c["x_cm"] = x                       # the twin shows where it really is
        if t < V["settle_s"] + V["window_s"]:
            return
        n, hits = len(self._ver), [g for s_, g in self._ver if s_]
        if n == 0:
            state.log(f"Cycle: cube #{c['id']}: push could not be verified (no camera / sensor data)", "warn")
            c["verify"] = "unknown"
            return self._push_ok("push done (not verified)")
        if len(hits) / n < V["min_seen_frac"]:
            c["verify"] = "ok"
            return self._push_ok("push verified: cube left the belt")
        togo = float(np.median(hits))
        c["push_fails"] = c.get("push_fails", 0) + 1
        c["verify"] = "failed"
        state.log(f"Cycle: cube #{c['id']} still on the belt after {c['routine']} "
                  f"(attempt {c['push_attempts']}, {togo:+.1f} cm from the push point)", "warn")
        if c["push_attempts"] < V["attempts"]:
            c["correct_cm"] = round(togo, 1)
            if abs(togo) > V["correct_tol_cm"]:
                self._corr, self._corr_odo0 = togo, self.odo
                self._belt("F" if togo > 0 else "R", C["run_speed_pct"])
                return self._goto("CORRECTING", f"moving cube #{c['id']} {togo:+.1f} cm onto the push point")
            return self._start_push(now)
        c["odo0"], c["forced_reverse_s"] = self.odo, C["failed_reverse_s"]
        self._belt("R", C["run_speed_pct"])
        self._goto("REVERSE", f"push failed {c['push_attempts']} times: back to the origin "
                              f"({C['failed_reverse_s']:g} s)")

    def _s_correcting(self, now, det, sp):
        if self._stalled(now):
            return self._fault("belt not moving (stall)")
        if abs(self.odo - self._corr_odo0) >= abs(self._corr) - abs(sp or 0.0) * C["stop_latency_s"]:
            self._belt("S")
            self._start_push(now)

    def _s_reverse(self, now, det, sp):
        if self._stalled(now):
            return self._fault("belt not moving (stall)")
        dist = abs(self.odo - self.cube["odo0"])
        forced = self.cube.get("forced_reverse_s")
        if forced is not None:                      # failed push: always this long, no distance check
            if now - self.since >= forced:
                self.cube["reverse_cm"] = round(dist, 2)
                self._finish("push failed, returned to origin")
                self._belt("F", C["run_speed_pct"])
                self._goto("RUN", f"failed cube sent back ({dist:.0f} cm, {forced:g} s)")
            return
        if self._src() == "ultrasonic":             # the sensor sees the cube leave the belt at the origin
            by_cm = self.us is None or cfg.us_position_cm(self.us) is None
        else:
            by_cm = C["reverse_cm"] is not None and dist >= C["reverse_cm"]
        by_time = C["reverse_s"] is not None and now - self.since >= C["reverse_s"]
        if (by_cm and now - self.since > 0.5) or by_time:
            self.cube["reverse_cm"] = round(dist, 2)
            self._finish("returned to origin")
            self._belt("F", C["run_speed_pct"])
            self._goto("RUN", f"cube left the belt at the origin ({dist:.0f} cm, {now - self.since:.1f} s)")

    def _s_to_end(self, now, det, sp):
        if self._stalled(now):
            return self._fault("belt not moving (stall)")
        if abs(self.odo - self.cube["odo0"]) >= C["clear_cm"]:
            self._finish("belt end")
            self._goto("RUN", "cube cleared the camera")

    def _arm_events(self, now):
        while self._arm_q:
            t, text = self._arm_q.popleft()
            self.arm_last = text
            if self.state != "PUSHING" or not getattr(self, "_push_sent", False):
                continue
            if text.startswith(A["error_prefix"]):
                return self._fault(f"arm error: {text}")
            if self._homing:                        # "OK HOME -> ..." / "OK -> (...)": the arm took our HOME
                if text.startswith("OK"):
                    self._home_seen = True
                elif text == A["done_text"] and self._home_seen:
                    return self._pushed(now)
            elif self.cube["routine"] in text.upper():
                self._push_seen_start = True        # "PUSH1 inicio": the arm took OUR command
            elif text == A["done_text"] and self._push_seen_start:
                return self._push_done(now)

    # ── bookkeeping ─────────────────────────────────────────────────────────────────────────
    def _goto(self, new, msg=""):
        old, self.state, self.since, self.msg = self.state, new, self.clock(), msg
        state.log(f"Cycle: {old} -> {new}" + (f" ({msg})" if msg else ""), "alert" if new == "SAFE_STOP" else "info")
        self.plant.publish(dict(event="state", state=new, msg=msg, t=self.since))

    def _fault(self, why):
        self._belt("S", force=True)
        if self.state == "PUSHING" or self.plant.live():
            self.plant.send_arm("STOP")
        self.counts["faults"] += 1
        if self.cube:
            self._finish(f"fault: {why}")
        self._goto("SAFE_STOP", why)

    def _finish(self, outcome):
        c, self.cube = self.cube, None
        c["outcome"] = outcome
        lat = {}
        if "t_stop_cmd" in c and "t_stopped" in c:
            lat["stop_s"] = c["t_stopped"] - c["t_stop_cmd"]
        if "t_stop_cmd" in c and "t_decision" in c:
            lat["classify_s"] = c["t_decision"] - c["t_stop_cmd"]
        if "t_push_cmd" in c and "t_arm_done" in c:
            lat["arm_s"] = c["t_arm_done"] - c["t_push_cmd"]
        if "t_home_cmd" in c and "t_home_done" in c:
            lat["home_s"] = c["t_home_done"] - c["t_home_cmd"]
        lat["cycle_s"] = self.clock() - c["t_detect"]
        c["latency"] = {k: round(v, 3) for k, v in lat.items()}
        c["deadline_miss"] = [k for k, lim in C["deadlines"].items() if k in lat and lat[k] > lim]
        for k in c["deadline_miss"]:
            self.misses[k] += 1
            state.log(f"Cycle: cube #{c['id']} missed the {k} deadline ({lat[k]:.2f} s > {C['deadlines'][k]} s)", "warn")
        self.cubes.appendleft(c)
        self.plant.publish(dict(event="cube", **{k: v for k, v in c.items() if k not in ("votes", "track")}))
        self._log_csv(c)

    def _log_csv(self, c):
        path = os.path.join(ROOT, C["log_csv"]) if C.get("log_csv") else None
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cols = ["time", "id", "cls", "conf", "why", "routine", "outcome", "stop_s", "classify_s", "arm_s", "home_s", "cycle_s",
                "src", "x_cmd_px", "x_final_px", "us_err_cm", "us_final_cm", "target_cm", "arm_travel_cm", "us_target_cm",
                "arm_us_cm", "arm_by", "push_attempts", "verify", "correct_cm", "reverse_cm",
                "cam_v", "enc_v", "speed_ratio", "cube_cm", "deadline_miss", "probs"]
        new = not os.path.exists(path)
        if not new:                                 # columns changed since the file was started: keep it aside
            with open(path) as f:
                if f.readline().strip() != ",".join(cols):
                    os.replace(path, path.replace(".csv", time.strftime("_old_%Y%m%d_%H%M%S.csv")))
                    new = True
        try:
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(cols)
                lat = c.get("latency", {})
                w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), c["id"], c.get("cls"), c.get("conf"), c.get("why"),
                            c.get("routine"), c["outcome"], lat.get("stop_s"), lat.get("classify_s"), lat.get("arm_s"),
                            lat.get("home_s"), lat.get("cycle_s"), c.get("src"), c.get("x_err_px"), c.get("x_final_px"), c.get("us_err_cm"),
                            c.get("us_final_cm"), c.get("target_cm"), c.get("arm_travel_cm"), c.get("us_target_cm"),
                            c.get("arm_us_cm"), c.get("arm_by"), c.get("push_attempts"), c.get("verify"),
                            c.get("correct_cm"), c.get("reverse_cm"), c.get("cam_v"), c.get("enc_v"),
                            c.get("speed_ratio"), c.get("cube_cm"),
                            ";".join(c.get("deadline_miss", [])), c.get("probs")])
        except OSError as e:
            state.log(f"Cycle log not written: {e}", "warn")

    # ── for the page / video ────────────────────────────────────────────────────────────────
    def view(self):
        with self.lock:
            return dict(state=self.state, since=self.since, msg=self.msg, counts=dict(self.counts),
                        misses=dict(self.misses), cubes=list(self.cubes), cube=dict(self.cube) if self.cube else None,
                        face=self.face, probs=None if self.probs is None else list(self.probs),
                        vote=self.vote.view() if self.vote is not None and self.state == "CLASSIFYING" else None,
                        arm_online=self.arm_online, arm_last=self.arm_last, live=self.plant.live(),
                        classes=self.classes, src=self.plant.position_source(), us=self.us,
                        calib=self.calib_summary())

    def draw(self, img):
        """Video overlay: stop line, the face found and its corners, current state."""
        with self.lock:
            h, w = img.shape[:2]
            xs = int(w * C["stop_x_frac"])
            cv2.line(img, (xs, 0), (xs, h - 1), (0, 220, 255), 1, cv2.LINE_AA)
            d = self.last_det
            if d is not None and d.get("corners") is not None and abs(d["x_stop"] - w * C["stop_x_frac"]) < 1:
                col = {"CLASSIFYING": (0, 200, 255), "OBJECT_DETECTED": (60, 200, 60)}.get(self.state, (200, 200, 200))
                cv2.polylines(img, [np.asarray(d["corners"], np.int32)], True, col, 2, cv2.LINE_AA)
            txt = self.state + (f"  #{self.cube['id']}" if self.cube else "")
            if self.cube and self.cube.get("cls"):
                txt += f"  {self.cube['cls']}"
            cv2.putText(img, txt, (xs + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (xs + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
        return img


def start(bridge, mqtt):
    """Create the orchestrator, hook it to MQTT (arm status) and the video overlay, start its thread."""
    orch = Orchestrator(Plant(bridge, mqtt))
    mqtt.listen(A["status_topic"], orch.on_arm_status)
    mqtt.listen(A["online_topic"], orch.on_arm_online)
    state.cycle_overlay = orch.draw
    orch.start_thread()
    return orch
