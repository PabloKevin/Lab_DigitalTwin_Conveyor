"""
sim_esp32.py - pretends to be the real conveyor so you can test the twin without hardware.

    python sim_esp32.py                 # normal conveyor
    python sim_esp32.py --loss 25       # real conveyor loses 25 % speed (friction) -> the twin raises a divergence
    (the loss can also be changed live:  mosquitto_pub -t conveyor/cmd/sim_loss -m 25)

Speaks exactly the MQTT protocol of firmware/esp32_conveyor:
  publishes  conveyor/telemetry   {"rpm":..,"setpoint":..,"output":..,"dir":"F"}   every 100 ms
             conveyor/status      online / offline (Last Will)
  listens    conveyor/cmd/direction (F|R|S), conveyor/cmd/speed (0-100), conveyor/cmd/kp|ki|kd
"""
import argparse
import json
import random
import time

import paho.mqtt.client as mqtt

import config as cfg

P = cfg.TOPIC_PREFIX
ap = argparse.ArgumentParser()
ap.add_argument("--host", default=cfg.MQTT["host"])
ap.add_argument("--port", type=int, default=cfg.MQTT["port"])
ap.add_argument("--loss", type=float, default=0.0, help="percent speed lost by the 'real' conveyor")
args = ap.parse_args()

st = dict(direction="S", speed=0.0, loss=args.loss, rpm=0.0, kp=2.0, ki=0.5, kd=0.1)


def on_message(client, userdata, msg):
    val = msg.payload.decode().strip()
    key = msg.topic.split("/")[-1]
    try:
        if key == "direction" and val in ("F", "R", "S"):
            st["direction"] = val
        elif key in ("speed", "loss", "kp", "ki", "kd") or key == "sim_loss":
            st["loss" if key == "sim_loss" else key] = float(val)
        print(f"cmd {key} = {val}")
    except ValueError:
        pass


try:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="sim-esp32")
except AttributeError:
    c = mqtt.Client(client_id="sim-esp32")
c.will_set(cfg.STATUS_TOPIC, "offline", retain=True)
c.on_message = on_message
c.connect(args.host, args.port)
c.subscribe(f"{P}/cmd/#")
c.publish(cfg.STATUS_TOPIC, "online", retain=True)
c.loop_start()
print(f"Simulated ESP32 -> {args.host}:{args.port}   (Ctrl+C to stop)")

dt = 0.1
try:
    while True:
        sign = {"F": 1, "R": -1}.get(st["direction"], 0)
        target = sign * cfg.MAX_RPM * st["speed"] / 100.0 * (1 - st["loss"] / 100.0)
        st["rpm"] += (target - st["rpm"]) * dt / 0.5                     # same dynamics as the twin model (tau = 0.5 s)
        rpm = st["rpm"] + random.gauss(0, 1.2)
        pwm = min(255, abs(target) / cfg.MAX_RPM * 255 * 1.05)
        c.publish(cfg.TOPIC_TELEMETRY, json.dumps(dict(
            rpm=round(rpm, 1), setpoint=round(sign * cfg.MAX_RPM * st["speed"] / 100.0, 1),
            output=round(pwm), dir=st["direction"])))
        time.sleep(dt)
except KeyboardInterrupt:
    c.publish(cfg.STATUS_TOPIC, "offline", retain=True)
    c.loop_stop()
