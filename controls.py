"""
controls.py - conversion helpers + the logic executed when a control changes.
Adding a control only requires editing config.CONTROLS; this file rarely changes.
"""
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


def apply(c, typed, bridge, force=False):
    """Store the value, update the twin parameter and (if configured + Live sync) publish to MQTT.
    force=True publishes even in Sandbox mode (used by the E-stop)."""
    state.controls[c["id"]] = typed
    if c.get("model_var"):
        state.params[c["model_var"]] = typed

    payload, txt = payload_text(c, typed)
    target = c.get("target", "real")
    where = []
    if target in ("twin", "both"):
        where.append("twin")
    if target in ("real", "both") and c.get("topic"):
        if state.sync or force:
            if bridge.publish(c["topic"], payload, retain=c.get("retain", False)):
                where.append("real")
            else:
                where.append("real: NOT SENT (no broker)")
        else:
            where.append("real: not sent (sandbox)")
    state.last_command = f"{c['label']} = {txt}  →  {', '.join(where)}"
    state.log(state.last_command)
    return state.last_command
