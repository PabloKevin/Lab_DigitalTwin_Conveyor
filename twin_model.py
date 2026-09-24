"""
twin_model.py - the digital side of the twin.

Runs at 20 Hz and does four things:
  1. step_model()       simulated conveyor  <-- EDIT THIS to change / extend the physics
  2. compute_derived()  evaluates the "derived" variables of config.VARIABLES
  3. integrates belt travel (cm) from the encoder and from the model
  4. calls divergence.evaluate()

The model receives its inputs from state.params, which are written by controls that have
model_var="..." in config.CONTROLS (direction, speed_pct, fault_loss_pct, ...).
"""
import math
import threading
import time

import config as cfg
import divergence
from state import state


def step_model(dt):
    """Simple first-order model: motor RPM follows the commanded RPM with time constant tau."""
    p = state.params
    sign = {"F": 1, "R": -1}.get(p.get("direction", "S"), 0)
    loss = p.get("fault_loss_pct", 0.0) / 100.0
    target = sign * cfg.MAX_RPM * p.get("speed_pct", 0.0) / 100.0 * (1.0 - loss)

    alpha = 1.0 - math.exp(-dt / cfg.MODEL["tau_s"])
    rpm = state.model["rpm"]
    rpm += (target - rpm) * alpha
    state.model["rpm"] = rpm

    belt = cfg.ENCODER_SIGN * rpm / 60.0 * 2 * math.pi * cfg.ROLLER_RADIUS_CM
    state.set_value("model_rpm", rpm)
    state.set_value("model_belt_speed_cm_s", belt)
    state.travel_model += belt * dt

    # ---- add your own model variables here, e.g. a thermal model:
    # state.model["temp"] = ...; state.set_value("model_temp", state.model["temp"])


def compute_derived():
    snap = state.snapshot()
    for v in cfg.VARIABLES:
        if v.get("source") == "derived":
            try:
                val = v["fn"](snap)
            except (KeyError, TypeError, ZeroDivisionError):
                continue
            if val is not None:
                state.set_value(v["id"], val)


def _loop():
    last = time.time()
    while True:
        now = time.time()
        dt = max(1e-3, now - last)
        last = now
        step_model(dt)
        compute_derived()
        sp = state.get("belt_speed_cm_s")
        if sp is not None:
            state.travel_real += sp * dt
        divergence.evaluate()
        time.sleep(0.05)


def start():
    threading.Thread(target=_loop, daemon=True, name="twin-model").start()
