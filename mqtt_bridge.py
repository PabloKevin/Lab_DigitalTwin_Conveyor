"""
mqtt_bridge.py - subscribes to the topics declared in config.VARIABLES and publishes commands.
You normally do not need to edit this file.
"""
import json
import time

import paho.mqtt.client as mqtt

import config as cfg
from state import state, is_num


class MqttBridge:
    def __init__(self):
        # topic -> list of variable dicts that read from it
        self.by_topic = {}
        for v in cfg.VARIABLES:
            if v.get("source", "mqtt") == "mqtt":
                self.by_topic.setdefault(v["topic"], []).append(v)

        cid = cfg.MQTT["client_id"]
        try:                                     # paho-mqtt >= 2.0
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        except AttributeError:                   # paho-mqtt 1.x
            self.client = mqtt.Client(client_id=cid)
        if cfg.MQTT.get("username"):
            self.client.username_pw_set(cfg.MQTT["username"], cfg.MQTT.get("password"))
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.reconnect_delay_set(min_delay=1, max_delay=10)
        self._last_err = 0

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self):
        self.client.connect_async(cfg.MQTT["host"], cfg.MQTT["port"], cfg.MQTT["keepalive"])
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, props=None):
        failed = rc.is_failure if hasattr(rc, "is_failure") else rc != 0
        if failed:
            state.log(f"MQTT connection refused ({rc})", "alert")
            return
        state.mqtt_connected = True
        state.log(f"MQTT connected to {cfg.MQTT['host']}:{cfg.MQTT['port']}")
        prefix = cfg.TOPIC_PREFIX + "/"
        client.subscribe(prefix + "#")                    # everything under the prefix (also feeds the MQTT monitor)
        for topic in list(self.by_topic) + [cfg.STATUS_TOPIC]:
            if not topic.startswith(prefix):              # topics outside the prefix need their own subscription
                client.subscribe(topic)

    def _on_disconnect(self, client, userdata, *args):
        if state.mqtt_connected:
            state.log("MQTT disconnected - retrying", "alert")
        state.mqtt_connected = False

    # ── incoming ───────────────────────────────────────────────────────────
    def _on_message(self, client, userdata, msg):
        now = time.time()
        text = msg.payload.decode("utf-8", errors="replace").strip()
        state.raw[msg.topic] = (now, text)

        if msg.topic == cfg.STATUS_TOPIC:
            state.device_status = text.lower()
            state.log(f"ESP32 status: {text}")
            return

        vars_ = self.by_topic.get(msg.topic)
        if not vars_:
            return
        data = None
        if any(v.get("key") for v in vars_):
            try:
                data = json.loads(text)
            except ValueError:
                self._err(f"Bad JSON on {msg.topic}: {text[:60]}")
                return
        for v in vars_:
            key = v.get("key")
            val = self._extract(data, key) if key else text
            if val is None:
                continue
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    pass
            if is_num(val):
                val = float(val)
            state.set_value(v["id"], val, now)

    @staticmethod
    def _extract(data, key):
        cur = data
        for part in key.split("."):                     # "motor.rpm" works for nested JSON
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return None
        return cur

    def _err(self, msg):
        if time.time() - self._last_err > 5:            # avoid flooding the log
            self._last_err = time.time()
            state.log(msg, "warn")

    # ── outgoing ───────────────────────────────────────────────────────────
    def publish(self, topic, payload, retain=False, qos=1, quiet=False):
        if not state.mqtt_connected:                    # never queue commands: a stale "Forward" is dangerous
            if not quiet:
                state.log(f"Not sent (no broker): {topic} = {payload}", "warn")
            return False
        self.client.publish(topic, payload, qos=qos, retain=retain)
        return True
