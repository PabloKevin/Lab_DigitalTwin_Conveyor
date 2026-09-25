"""
controls.py - conversion helpers + the logic executed when a control changes.
Adding a control only requires editing config.CONTROLS; this file rarely changes.
"""
import threading

import requests

import config as cfg
from state import state

BY_ID = {c["id"]: c for c in cfg.CONTROLS}


def typed_default(c):
    d = c.get("default")
    if c["kind"] == "switch":
        return bool(d)
    if c["kind"] in ("slider", "number"):
        return float(d)
    return d


def init():
    """Load defaults into state (call once at startup)."""
    for c in cfg.CONTROLS:
        val = typed_default(c)
        state.controls[c["id"]] = val
        if c.get("model_var"):
            state.params[c["model_var"]] = val


def parse(c, raw):
    """Dash component value -> typed value (None = ignore)."""
    if c["kind"] == "switch":
        return "on" in (raw or [])
    if c["kind"] in ("slider", "number"):
        return None if raw is None else float(raw)
    return raw


def to_component(c, typed):
    if c["kind"] == "switch":
        return ["on"] if typed else []
    return typed


def payload_text(c, typed):
    if isinstance(typed, bool):
        txt = "1" if typed else "0"
    elif isinstance(typed, float):
        txt = "%g" % typed
    else:
        txt = str(typed)
    return c.get("payload", "{value}").replace("{value}", txt), txt


def _camera_set(var, val, label):
    url = cfg.VISION["camera_control_url"].rstrip("/") + "/control"
    try:
        r = requests.get(url, params={"var": var, "val": val}, timeout=3)
        if r.status_code != 200:
            state.log(f"ESP32-CAM refused {var}={val} (HTTP {r.status_code})", "warn")
    except requests.RequestException as e:
        state.log(f"ESP32-CAM not reachable for {label}: {type(e).__name__}", "warn")


def apply(c, typed, bridge, force=False):
    """Store the value, update the twin parameter and (if configured + Live sync) send it to the conveyor (bridge.publish).
    force=True publishes even in Sandbox mode (used by the E-stop)."""
    state.controls[c["id"]] = typed
    if c.get("model_var"):
        state.params[c["model_var"]] = typed

    payload, txt = payload_text(c, typed)
    target = c.get("target", "real")
    where = []
    if target == "camera":                       # HTTP call to the ESP32-CAM, done in a thread so the UI never blocks
        val = int(typed)                         # bool -> 0/1, slider float -> int
        threading.Thread(target=_camera_set, args=(c["camera_var"], val, c["label"]), daemon=True).start()
        where.append("camera")
    if target in ("twin", "both"):
        where.append("twin")
    if target in ("real", "both") and c.get("topic"):
        if state.sync or force:
            if bridge.publish(c["topic"], payload, retain=c.get("retain", False)):
                where.append("real")
            else:
                where.append("real: NOT SENT (no link)")
        else:
            where.append("real: not sent (sandbox)")
    state.last_command = f"{c['label']} = {txt}  →  {', '.join(where)}"
    state.log(state.last_command)
    return state.last_command
