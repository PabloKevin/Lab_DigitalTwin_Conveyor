"""
divergence.py - evaluates config.DIVERGENCE_RULES.

To add a new kind of rule, add a branch in _check() that returns (known, delta, is_bad).
"""
import time

import config as cfg
from state import state


def _check(rule, now):
    kind = rule["kind"]

    if kind == "compare":
        a, b = state.get(rule["a"]), state.get(rule["b"])
        if a is None or b is None:
            return False, None, False
        delta = abs(a - b)
        tol = rule.get("abs_tol", 0) + rule.get("rel_tol", 0) * max(abs(a), abs(b))
        return True, delta, delta > tol

    if kind == "stale":
        age = state.age(rule["var"])
        if age is None:
            age = now - state.started            # never received
        return True, age, age > rule["max_age_s"]

    if kind == "object_position":
        objs = list(state.objects)
        errs = []
        for o in objs:
            e = o.get("err_cm")
            o["diverging"] = e is not None and abs(e) > rule["abs_tol"]
            if e is not None:
                errs.append(abs(e))
        if not errs:
            return False, None, False
        delta = max(errs)
        return True, delta, delta > rule["abs_tol"]

    return False, None, False


def evaluate():
    now = time.time()
    for rule in cfg.DIVERGENCE_RULES:
        st = state.divergences.setdefault(
            rule["id"], dict(active=False, known=False, delta=None, pending=None, since=None))

        if not state.sync and rule["kind"] != "stale":     # sandbox mode: twin is disconnected from reality
            st.update(known=False, delta=None, pending=None, active=False)
            continue

        known, delta, bad = _check(rule, now)
        st["known"], st["delta"] = known, delta

        if known and bad:
            st["pending"] = st["pending"] or now
            if not st["active"] and now - st["pending"] >= rule.get("hold_s", 0):
                st["active"], st["since"] = True, now
                state.log(f"DIVERGENCE {rule['label']}: Δ={delta:.1f} {rule.get('unit', '')} - {rule['cause']}", "alert")
        else:
            st["pending"] = None
            if st["active"]:
                st["active"] = False
                state.log(f"Divergence cleared: {rule['label']}")
