"""The cycle state machine (orchestrator.py) against a simulated plant: belt kinematics from commands, a synthetic
camera band with a cube riding the belt, the real cube detector, a stub model and a scripted arm.
    python -m unittest tests.test_orchestrator -v      (conda env Control or conveyor-twin)"""
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
    def send_arm(self, cmd): self.arm.append(cmd); return True
    def publish(self, payload): self.published.append(payload)

    def frame(self):
        self.seq += 1
        img = self.bg.copy()
        for x0 in self.cubes:
            x = x0 + self.travel_ * PPCM
            if -40 < x < 680:
                cube(img, int(x), 80, 56, 8)
        return img, self.t[0], self.seq

    def cube_x(self, i=0):
        return self.cubes[i] + self.travel_ * PPCM

    def advance(self, dt):
        while self.pending and self.pending[0][0] <= self.t[0]:
            self.dir = self.pending.pop(0)[1]
        target = V if self.dir == "F" else 0.0
        self.v += (target - self.v) * min(1.0, dt / 0.08)      # fast first-order motor
        self.travel_ += self.v * dt
        self.t[0] += dt


class StubModel:
    def __init__(self, cls): self.set(cls)
    def set(self, cls):
        self.p = np.full(4, 0.02)
        self.p[["solid", "stripes", "corners", "other"].index(cls)] = 0.94
    def predict(self, face): return self.p


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        cfg.CYCLE["log_csv"] = None
        self.t = [1000.0]
        self.plant = FakePlant(self.t)
        p = pp.load_params()
        p["edges"].update(chroma_gain=1.5, lo=40, hi=100, min_edge_px=12)
        p["mask"].update(close_r=2, open_r=2)
        p["blob"].update(min_area_px=120)
        self.model = StubModel("solid")
        self.o = orc.Orchestrator(self.plant, model=self.model, params=p, clock=lambda: self.t[0])

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
        self.assertAlmostEqual(c["arm_travel_cm"], cfg.CYCLE["arm_distance_cm"], delta=1.5)
        self.assertGreater(self.plant.travel_ - travel0, cfg.CYCLE["arm_distance_cm"] - 1.5)
        for k in ("stop_s", "classify_s", "arm_s", "cycle_s"):
            self.assertIn(k, c["latency"])
        self.assertTrue(any(m.get("event") == "decision" and m["cls"] == "solid" for m in self.plant.published))

    def test_stripes_goes_to_push2(self):
        self.model.set("stripes")
        self.o.start()
        self.push_routine("PUSH2")
        self.assertEqual(self.o.cubes[0]["cls"], "stripes")

    def test_other_is_dudoso_and_rides_to_the_end(self):
        self.model.set("other")
        self.o.start()
        self.until_state("TO_END")
        self.until_state("RUN")
        self.assertEqual(self.plant.arm, [])
        self.assertEqual((self.o.cubes[0]["cls"], self.o.cubes[0]["outcome"]), ("dudoso", "belt end"))

    def test_two_cubes_in_view(self):
        """Cube 1 rides away to the end while cube 2 approaches: both stopped and classified, nothing 'missed'."""
        self.model.set("other")
        self.plant.cubes = [30.0, 30.0 - 260.0]          # second cube 26 cm upstream
        self.o.start()
        self.assertTrue(self.run_until(lambda: len(self.o.cubes) >= 2, 60.0), f"stuck in {self.o.state}")
        self.assertEqual([c["outcome"] for c in self.o.cubes], ["belt end", "belt end"])
        self.assertNotIn("missed", self.o.counts)
        self.assertLess(abs(self.o.cubes[0]["x_err_px"]), 25)

    def test_corners_not_pushed(self):
        self.model.set("corners")
        self.o.start()
        self.until_state("TO_END")
        self.assertEqual(self.o.cubes[0]["cls"] if self.o.cubes else self.o.cube["cls"], "corners")

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


if __name__ == "__main__":
    unittest.main()
