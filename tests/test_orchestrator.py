"""The cycle state machine (orchestrator.py) against a simulated plant: belt kinematics from commands, a synthetic
camera band with a cube riding the belt, the real cube detector, a stub model and a scripted arm.
    python -m unittest tests.test_orchestrator -v      (conda env Control or conveyor-twin)"""
import copy
import unittest

import numpy as np

import config as cfg
import orchestrator as orc
from classifier import preprocess as pp
from classifier.tests.test_preprocess import belt, cube

PPCM = 10.0          # px per cm in the synthetic band
V = 8.0              # belt speed when running, cm/s
CMD_DELAY = 0.15     # serial + Arduino loop: a direction command takes effect this much later


class FakePlant:
    def __init__(self, t):
        self.t, self.live_ = t, True
        self.dir, self.v, self.travel_ = "S", 0.0, 0.0
        self.pending = []                                # (time it takes effect, direction)
        self.cubes = [30.0]                              # cube centre px at travel 0 (more cubes = further upstream)
        self.seq, self.arm, self.published = 0, [], []
        self.ignore_stop = False
        self.telemetry = True
        self.hidden = False                              # camera sees no cube (e.g. occluded)
        self.src = "camera"
        self.us_t, self.us_d = None, None
        self.bg = belt(1, seam=False, glare=False)

    # plant interface
    def live(self): return self.live_
    def belt(self, d, speed_pct=None, force=False):
        if d == "S" and self.ignore_stop and not force:
            return
        self.pending.append((self.t[0] + CMD_DELAY, d))
    def travel(self): return self.travel_
    def speed(self): return self.v
    def telemetry_age(self): return 0.0 if self.telemetry else 99.0
    def geometry(self): return PPCM, 1
    def px_to_cm(self, cx): return cx / PPCM
    def position_source(self): return self.src
    def us_sample(self): return None if self.us_t is None else (self.us_t, self.us_d)
    def send_arm(self, cmd): self.arm.append(cmd); return True
    def publish(self, payload): self.published.append(payload)

    def frame(self):
        self.seq += 1
        img = self.bg.copy()
        for x0 in self.cubes:
            x = x0 + self.travel_ * PPCM
            if -40 < x < 680 and not self.hidden:
                cube(img, int(x), 80, 56, 8)
        return img, self.t[0], self.seq

    def cube_x(self, i=0):
        return self.cubes[i] + self.travel_ * PPCM

    def advance(self, dt):
        while self.pending and self.pending[0][0] <= self.t[0]:
            self.dir = self.pending.pop(0)[1]
        target = {"F": V, "R": -V}.get(self.dir, 0.0)
        self.v += (target - self.v) * min(1.0, dt / 0.08)      # fast first-order motor
        self.travel_ += self.v * dt
        self.t[0] += dt
        if self.us_t is None or self.t[0] - self.us_t >= 0.1:     # HC-SR04 telemetry at 10 Hz, nearest cube
            u = cfg.ULTRASONIC
            ds = [-u["facing"] * (u["sensor_x_cm"] - x / PPCM) for x in map(self.cube_x, range(len(self.cubes)))
                  if 0 <= x / PPCM <= cfg.BELT_LENGTH_CM]
            self.us_t, self.us_d = self.t[0], min(ds) if ds else -1.0


class StubModel:
    def __init__(self, cls): self.set(cls)
    def set(self, cls, p=0.94):
        self.p = np.full(4, (1 - p) / 3)
        self.p[["solid", "stripes", "corners", "other"].index(cls)] = p
    def predict(self, face): return self.p


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self._cycle = copy.deepcopy(cfg.CYCLE)
        cfg.CYCLE["log_csv"] = None
        for cls, cm in (("solid", 15.0), ("stripes", 20.0), ("corners", 25.0)):     # a different stop per class
            cfg.CYCLE["push"][cls]["cm"] = cm
        self.t = [1000.0]
        self.plant = FakePlant(self.t)
        p = pp.load_params()
        p["edges"].update(chroma_gain=1.5, lo=40, hi=100, min_edge_px=12)
        p["mask"].update(close_r=2, open_r=2)
        p["blob"].update(min_area_px=120)
        self.model = StubModel("solid")
        self.o = orc.Orchestrator(self.plant, model=self.model, params=p, clock=lambda: self.t[0])

    def tearDown(self):
        cfg.CYCLE.clear()
        cfg.CYCLE.update(self._cycle)

    def run_until(self, cond, max_s=30.0, dt=0.05):
        t_end = self.t[0] + max_s
        while self.t[0] < t_end:
            self.plant.advance(dt)
            self.o.step()
            if cond():
                return True
        return False

    def until_state(self, s, max_s=30.0):
        self.assertTrue(self.run_until(lambda: self.o.state == s, max_s), f"never reached {s}, stuck in {self.o.state}")

    def push_routine(self, routine):
        self.until_state("PUSHING")
        self.assertTrue(self.run_until(lambda: self.plant.arm, 3.0), "push never sent")
        self.assertEqual(self.plant.arm[-1], routine)
        self.o.on_arm_status("IDLE")                       # stale IDLE before ours: must be ignored
        self.run_until(lambda: False, 0.2)
        self.assertEqual(self.o.state, "PUSHING")
        for msg in (f"{routine} inicio", "OK PUSH1 -> DEGS 130 170 120 0", "DONE", "DONE", "IDLE"):
            self.o.on_arm_status(msg)
        self.assertTrue(self.run_until(lambda: self.plant.arm[-1] == "HOME", 1.0), "HOME not sent after the push")
        self.run_until(lambda: False, 0.3)
        self.assertEqual(self.o.state, "PUSHING")               # belt waits for the arm to be home
        for msg in ("OK HOME -> DEGS 90 90 90 0", "DONE", "IDLE"):
            self.o.on_arm_status(msg)
        self.until_state("RUN", 2.0)

    def test_solid_cube_stops_at_centre_is_classified_and_pushed(self):
        self.assertEqual(self.o.start(), "started")
        self.until_state("OBJECT_DETECTED")
        self.until_state("CLASSIFYING")
        self.run_until(lambda: self.plant.v < 0.1, 2.0)
        x = self.plant.cube_x()
        self.assertLess(abs(x - 320), 25, f"cube stopped at x={x:.0f} px, stop line 320")
        self.until_state("TO_ARM")
        travel0 = self.plant.travel_
        self.push_routine("PUSH1")
        c = self.o.cubes[0]
        self.assertEqual((c["cls"], c["routine"], c["outcome"]), ("solid", "PUSH1", "pushed"))
        self.assertAlmostEqual(c["arm_travel_cm"], 15.0, delta=1.5)
        self.assertGreater(self.plant.travel_ - travel0, 15.0 - 1.5)
        for k in ("stop_s", "classify_s", "arm_s", "cycle_s"):
            self.assertIn(k, c["latency"])
        self.assertTrue(any(m.get("event") == "decision" and m["cls"] == "solid" for m in self.plant.published))

    def test_each_class_stops_at_its_own_push_position(self):
        for cls, routine, cm in (("stripes", "PUSH2", 20.0), ("corners", "PUSH3", 25.0)):
            with self.subTest(cls=cls):
                self.tearDown()                          # restore config before a fresh setUp
                self.setUp()
                self.model.set(cls)
                self.o.start()
                self.push_routine(routine)
                c = self.o.cubes[0]
                self.assertEqual((c["cls"], c["routine"], c["target_cm"]), (cls, routine, cm))
                self.assertAlmostEqual(c["arm_travel_cm"], cm, delta=1.5)

    def test_other_goes_back_to_the_origin_then_forward(self):
        self.model.set("other")
        self.o.start()
        self.until_state("REVERSE")
        self.run_until(lambda: self.plant.dir == "R", 1.0)
        x_turn = self.plant.cube_x()
        self.until_state("RUN", cfg.CYCLE["reverse_s"] + 2)
        self.run_until(lambda: self.plant.dir == "F", 1.0)
        self.assertEqual(self.plant.arm, [])
        c = self.o.cubes[0]
        self.assertEqual((c["cls"], c["outcome"]), ("other", "returned to origin"))
        self.assertLess(self.plant.cube_x(), x_turn - 20 * PPCM, "cube did not travel back")
        self.assertTrue(self.run_until(lambda: self.plant.v > V / 2, 2.0), "belt not running forward again")

    def test_reverse_ends_by_time_when_distance_is_off(self):
        cfg.CYCLE.update(reverse_cm=None, reverse_s=2.0)
        self.model.set("other")
        self.o.start()
        self.until_state("REVERSE")
        t0 = self.t[0]
        self.until_state("RUN", 5.0)
        self.assertAlmostEqual(self.t[0] - t0, 2.0, delta=0.15)

    def test_two_cubes_in_view(self):
        """Cube 1 rides away to the end while cube 2 approaches: both stopped and classified, nothing 'missed'."""
        self.model.set("other")
        cfg.CYCLE["reverse_classes"] = ()                # other/dudoso ride to the end here
        self.plant.cubes = [30.0, 30.0 - 260.0]          # second cube 26 cm upstream
        self.o.start()
        self.assertTrue(self.run_until(lambda: len(self.o.cubes) >= 2, 60.0), f"stuck in {self.o.state}")
        self.assertEqual([c["outcome"] for c in self.o.cubes], ["belt end", "belt end"])
        self.assertNotIn("missed", self.o.counts)
        self.assertLess(abs(self.o.cubes[0]["x_err_px"]), 25)

    def test_arm_timeout_is_a_fault_until_reset(self):
        self.o.start()
        self.until_state("PUSHING")
        self.until_state("SAFE_STOP", cfg.ARM["timeout_s"] + 5)
        self.run_until(lambda: False, 0.3)
        self.assertEqual(self.plant.dir, "S")
        self.assertEqual(self.plant.arm[-1], "STOP")
        self.assertIn("did not finish", self.o.msg)
        self.assertNotEqual(self.o.start(), "started")
        self.assertEqual(self.o.reset(), "reset")
        self.assertEqual(self.o.state, "IDLE")

    def test_arm_error(self):
        self.o.start()
        self.until_state("PUSHING")
        self.run_until(lambda: self.plant.arm, 3.0)
        self.o.on_arm_status("ERR cola llena")
        self.until_state("SAFE_STOP", 1.0)

    def test_arm_offline_blocks_push(self):
        self.o.on_arm_online("0")
        self.o.start()
        self.until_state("SAFE_STOP")
        self.assertIn("offline", self.o.msg)

    def test_belt_that_does_not_stop(self):
        self.plant.ignore_stop = True
        self.o.start()
        self.until_state("SAFE_STOP")
        self.assertIn("did not stop", self.o.msg)

    def test_telemetry_lost(self):
        self.o.start()
        self.run_until(lambda: False, 0.5)
        self.plant.telemetry = False
        self.until_state("SAFE_STOP", 1.0)
        self.assertIn("telemetry", self.o.msg)

    def test_estop_and_operator_stop(self):
        self.o.start()
        self.run_until(lambda: False, 0.5)
        self.assertEqual(self.o.stop(), "stopped")
        self.run_until(lambda: False, 0.3)
        self.assertEqual((self.o.state, self.plant.dir), ("IDLE", "S"))
        self.o.start()
        self.o.estop()
        self.assertEqual(self.o.state, "SAFE_STOP")

    def test_sandbox_simulates_the_arm(self):
        self.plant.live_ = False
        self.o.start()
        self.until_state("PUSHING")
        self.until_state("RUN", cfg.CYCLE["sim_push_s"] + 3)
        self.assertEqual(self.o.cubes[0]["outcome"], "pushed")


    def test_camera_losing_the_cube_while_classifying_is_not_a_fault(self):
        """Regression: a frame with no cube during CLASSIFYING crashed with KeyError 's' -> SAFE_STOP."""
        self.o.start()
        self.until_state("CLASSIFYING")
        self.plant.hidden = True
        self.assertTrue(self.run_until(lambda: self.o.state != "CLASSIFYING", 10.0))
        self.assertNotEqual(self.o.state, "SAFE_STOP", self.o.msg)
        self.assertEqual(self.o.cube["cls"], "other")                  # no face seen at all -> other

    def test_low_confidence_still_takes_the_highest_class(self):
        self.model.set("stripes", p=0.4)                # 0.4 vs 0.2 / 0.2 / 0.2: formerly "dudoso"
        self.o.start()
        self.push_routine("PUSH2")
        self.assertEqual(self.o.cubes[0]["cls"], "stripes")
        self.assertIn("home_s", self.o.cubes[0]["latency"])

    def test_belt_drawing_follows_the_cube_to_the_arm(self):
        self.o.start()
        self.until_state("TO_ARM")
        self.run_until(lambda: False, 1.0)
        (ob,) = self.o.objects()
        self.assertEqual(ob["label"], "solid")
        self.assertAlmostEqual(ob["x_cm"], self.plant.cube_x() / PPCM, delta=1.0)

    def us_reading(self):
        u = cfg.ULTRASONIC
        return -u["facing"] * (u["sensor_x_cm"] - self.plant.cube_x() / PPCM)

    def test_ultrasonic_stops(self):
        U = cfg.CYCLE["ultrasonic"]
        u = cfg.ULTRASONIC
        U["classify_cm"] = u["sensor_x_cm"] - 32.0         # cube centred at 32 cm = 320 px, under the camera
        for cls, us_cm in (("solid", U["classify_cm"] - 12.0), ("corners", U["classify_cm"] - 22.0)):
            cfg.CYCLE["push"][cls]["us_cm"] = us_cm
        self.plant.src = "ultrasonic"
        self.o.start()
        self.until_state("CLASSIFYING")
        self.run_until(lambda: self.plant.v < 0.1, 2.0)
        self.assertAlmostEqual(self.us_reading(), U["classify_cm"], delta=1.5)
        self.push_routine("PUSH1")
        c = self.o.cubes[0]
        self.assertEqual((c["src"], c["arm_by"]), ("ultrasonic", "ultrasonic"))
        self.assertAlmostEqual(c["arm_us_cm"], c["us_target_cm"], delta=1.5)
        self.assertAlmostEqual(c["arm_travel_cm"], 12.0, delta=1.5)

    def test_ultrasonic_reverse_until_the_cube_leaves_the_belt(self):
        cfg.CYCLE["ultrasonic"]["classify_cm"] = cfg.ULTRASONIC["sensor_x_cm"] - 32.0
        cfg.CYCLE.update(reverse_s=20.0)
        self.plant.src = "ultrasonic"
        self.model.set("other")
        self.o.start()
        self.until_state("REVERSE")
        self.until_state("RUN", 10.0)
        self.assertEqual(self.o.cubes[0]["outcome"], "returned to origin")
        self.assertLess(self.plant.cube_x() / PPCM, 1.0)          # it left the belt at x = 0
        self.assertGreater(self.plant.cube_x() / PPCM, -3.0)      # and the belt turned forward right after


if __name__ == "__main__":
    unittest.main()
