"""
config.py  -  EVERYTHING you normally want to edit lives in this file.

  * VARIABLES         what data exists (from MQTT, derived, model or camera)
  * PANELS            which variables are shown in which window
  * PLOTS             which variables are drawn in which time-series chart
  * CONTROLS          widgets that change the digital twin and/or the real conveyor
  * DIVERGENCE_RULES  how physical vs digital mismatches are detected

To add a sensor:   1) add a dict to VARIABLES   2) (optional) add its id to a PANELS / PLOTS entry
To add a control:  add a dict to CONTROLS  (it publishes to MQTT and/or changes the twin model)
"""
import math

# ════════════════════════════════════════════════════════════════════════════
# MQTT
# ════════════════════════════════════════════════════════════════════════════
MQTT = dict(
    host="localhost",          # Mosquitto running on this PC
    port=1883,
    username=None,
    password=None,
    client_id="conveyor-digital-twin",
    keepalive=30,
)
TOPIC_PREFIX = "conveyor"
TOPIC_TELEMETRY = f"{TOPIC_PREFIX}/telemetry"   # ESP32 -> JSON {"rpm":..,"setpoint":..,"output":..,"dir":"F"}
STATUS_TOPIC = f"{TOPIC_PREFIX}/status"         # ESP32 -> "online" / "offline" (Last Will)

# ════════════════════════════════════════════════════════════════════════════
# Conveyor geometry
# ════════════════════════════════════════════════════════════════════════════
BELT_LENGTH_CM = 60.0
ROLLER_RADIUS_CM = 5.0     # drive roller radius (lab guide: r = 5 cm)
MAX_RPM = 300.0            # RPM at 100 % speed (same mapping as the Arduino code)
ENCODER_SIGN = 1           # set to -1 if "Forward" gives negative RPM

# ════════════════════════════════════════════════════════════════════════════
# Web app
# ════════════════════════════════════════════════════════════════════════════
WEB = dict(host="127.0.0.1", port=8050, open_browser=True)
HISTORY_SECONDS = 60       # length of the time-series plots
HISTORY_DT = 0.1           # minimum spacing between stored points (s)
STALE_AFTER_S = 2.0        # a value older than this is shown as "—" / considered missing

# ════════════════════════════════════════════════════════════════════════════
# Camera + vision
# ════════════════════════════════════════════════════════════════════════════
VISION = dict(
    mode="yolo",           # "yolo" = real camera + YOLO | "sim" = fake objects (no camera) | "off"
    camera_url="http://192.168.1.50:81/stream",   # ESP32-CAM MJPEG stream (also accepts a video file or 0 for a webcam)
    model="yolo11n.pt",    # any Ultralytics weights; use your own trained .pt for your objects
    conf=0.35,
    imgsz=640,
    device=None,           # None = auto, "cpu", "cuda:0", "mps"
    classes=None,          # e.g. [39, 41] to keep only some COCO classes; None = all
    tracker="bytetrack.yaml",
    infer_fps=10,          # max inferences per second
    # Region of the image that contains the belt (x1, y1, x2, y2) in pixels.
    # x1 -> 0 cm and x2 -> BELT_LENGTH_CM. Tune it live in the "Camera calibration" controls.
    belt_roi_px=(20, 60, 620, 420),
    flip_x=False,          # True if the camera sees the belt mirrored (forward = towards the left)
    speed_window_s=1.0,    # window used to fit each object's speed
    lost_after_s=1.5,      # forget a track not seen for this long
    publish_topic=f"{TOPIC_PREFIX}/vision/objects",   # camera results are published back to MQTT
    publish_hz=5,
    # --- only for mode="sim" ---
    sim_spawn_every_s=6.0,
    sim_slip=1.0,          # 1.0 = objects move with the belt; 0.8 = they slip (creates a divergence)
)

# ════════════════════════════════════════════════════════════════════════════
# VARIABLES
#   source="mqtt"     read from an MQTT topic (topic + key). key=None -> scalar payload
#   source="derived"  computed here: fn(values_dict) -> value
#   source="model"    written by twin_model.py
#   source="vision"   written by vision.py
#   Optional: unit, fmt, color (plot line), warn=(lo,hi), alarm=(lo,hi)  (None = no limit), hidden=True
# ════════════════════════════════════════════════════════════════════════════
VARIABLES = [
    # --- from the ESP32 --------------------------------------------------
    dict(id="rpm", label="Motor speed", unit="RPM", source="mqtt", topic=TOPIC_TELEMETRY, key="rpm", fmt="{:.0f}", color="#1f5fbf"),
    dict(id="setpoint", label="PID setpoint", unit="RPM", source="mqtt", topic=TOPIC_TELEMETRY, key="setpoint", fmt="{:.0f}", color="#8a97a8"),
    dict(id="pwm", label="PWM output", unit="/255", source="mqtt", topic=TOPIC_TELEMETRY, key="output", fmt="{:.0f}",
         warn=(None, 230), alarm=(None, 250)),
    dict(id="direction", label="Direction", source="mqtt", topic=TOPIC_TELEMETRY, key="dir"),
    # EXAMPLE - uncomment to add a motor current sensor (ACS712) in 2 minutes:
    # dict(id="current", label="Motor current", unit="A", source="mqtt", topic=TOPIC_TELEMETRY, key="current",
    #      fmt="{:.2f}", warn=(None, 1.5), alarm=(None, 2.5)),
    # A sensor on its own topic (scalar payload) works too:
    # dict(id="temp", label="Motor temp", unit="°C", source="mqtt", topic=f"{TOPIC_PREFIX}/telemetry/temp", key=None),

    # --- derived from other variables -------------------------------------
    # v is a dict {id: value} with the fresh values. If a key is missing the variable is skipped.
    dict(id="belt_speed_cm_s", label="Belt speed (encoder)", unit="cm/s", source="derived", fmt="{:.1f}",
         color="#1f5fbf",
         fn=lambda v: ENCODER_SIGN * v["rpm"] / 60.0 * 2 * math.pi * ROLLER_RADIUS_CM),   # Eq. (1) of the lab guide

    # --- digital twin model (twin_model.py) --------------------------------
    dict(id="model_rpm", label="Model speed", unit="RPM", source="model", fmt="{:.0f}", color="#d98a00"),
    dict(id="model_belt_speed_cm_s", label="Model belt speed", unit="cm/s", source="model", fmt="{:.1f}", color="#d98a00"),

    # --- from the camera (vision.py) ----------------------------------------
    dict(id="cam_belt_speed_cm_s", label="Belt speed (camera)", unit="cm/s", source="vision", fmt="{:.1f}", color="#7a3fb0"),
    dict(id="cam_objects", label="Objects on belt", source="vision", fmt="{:.0f}"),
    dict(id="cam_fps", label="Vision rate", unit="fps", source="vision", fmt="{:.1f}"),
]

# ════════════════════════════════════════════════════════════════════════════
# PANELS  (variable "windows").  Variables not listed anywhere go to an "Other" panel.
# ════════════════════════════════════════════════════════════════════════════
PANELS = [
    dict(title="Drive", vars=["rpm", "setpoint", "pwm", "direction", "belt_speed_cm_s"]),
    dict(title="Camera", vars=["cam_belt_speed_cm_s", "cam_objects", "cam_fps"]),
    dict(title="Twin model", vars=["model_rpm", "model_belt_speed_cm_s"]),
]

# ════════════════════════════════════════════════════════════════════════════
# PLOTS  (time series).  Each series is a variable id.
# ════════════════════════════════════════════════════════════════════════════
PLOTS = [
    dict(id="p_rpm", title="Motor speed (RPM)", series=["rpm", "setpoint", "model_rpm"]),
    dict(id="p_belt", title="Belt speed (cm/s)", series=["belt_speed_cm_s", "cam_belt_speed_cm_s", "model_belt_speed_cm_s"]),
]

# ════════════════════════════════════════════════════════════════════════════
# CONTROLS
#   kind:    "slider" | "number" | "buttons" | "switch"
#   target:  "real"  -> publish to MQTT only
#            "twin"  -> change the digital model / vision parameter only (what-if, calibration, fault injection)
#            "both"  -> publish to MQTT and change the twin
#   topic:   MQTT topic to publish to.  payload: optional template, "{value}" is replaced
#   model_var: name of the twin parameter this control writes (state.params[model_var])
# ════════════════════════════════════════════════════════════════════════════
_R = VISION["belt_roi_px"]
CONTROLS = [
    dict(id="direction", group="Motion", label="Direction", kind="buttons",
         options=[("Forward", "F"), ("Stop", "S"), ("Reverse", "R")], default="S",
         target="both", topic=f"{TOPIC_PREFIX}/cmd/direction", model_var="direction"),
    dict(id="speed", group="Motion", label="Speed setpoint", unit="%", kind="slider",
         min=0, max=100, step=1, default=0,
         target="both", topic=f"{TOPIC_PREFIX}/cmd/speed", model_var="speed_pct"),

    dict(id="kp", group="PID gains", label="Kp", kind="number", step=0.1, default=2.0,
         target="real", topic=f"{TOPIC_PREFIX}/cmd/kp"),
    dict(id="ki", group="PID gains", label="Ki", kind="number", step=0.1, default=0.5,
         target="real", topic=f"{TOPIC_PREFIX}/cmd/ki"),
    dict(id="kd", group="PID gains", label="Kd", kind="number", step=0.05, default=0.1,
         target="real", topic=f"{TOPIC_PREFIX}/cmd/kd"),

    # Twin-only "what-if / fault injection" (lab guide section 5.3):
    dict(id="fault_loss", group="Twin what-if", label="Inject speed loss (twin only)", unit="%", kind="slider",
         min=0, max=60, step=1, default=0, target="twin", model_var="fault_loss_pct"),

    # Camera calibration - changes take effect immediately, watch the overlay in the camera window:
    dict(id="roi_x1", group="Camera calibration", label="Belt start x (px)  = 0 cm", kind="number", step=5, default=_R[0],
         target="twin", model_var="roi_x1"),
    dict(id="roi_x2", group="Camera calibration", label="Belt end x (px)  = 60 cm", kind="number", step=5, default=_R[2],
         target="twin", model_var="roi_x2"),
    dict(id="roi_y1", group="Camera calibration", label="Belt top y (px)", kind="number", step=5, default=_R[1],
         target="twin", model_var="roi_y1"),
    dict(id="roi_y2", group="Camera calibration", label="Belt bottom y (px)", kind="number", step=5, default=_R[3],
         target="twin", model_var="roi_y2"),
    dict(id="flip_x", group="Camera calibration", label="Mirror image", kind="switch", default=VISION["flip_x"],
         target="twin", model_var="flip_x"),

    # EXAMPLE of a switch that talks to the real conveyor with a JSON payload:
    # dict(id="pid_on", group="PID gains", label="PID enabled", kind="switch", default=True,
    #      target="real", topic=f"{TOPIC_PREFIX}/cmd/pid", payload='{"enabled": {value}}'),
]

# E-stop button (top right): sets this control to this value, which publishes it like any other control.
ESTOP = dict(control="direction", value="S")

# ════════════════════════════════════════════════════════════════════════════
# TWIN MODEL parameters (see twin_model.py -> step_model)
# ════════════════════════════════════════════════════════════════════════════
MODEL = dict(
    tau_s=0.5,     # first-order time constant of the motor+belt (s)
)

# ════════════════════════════════════════════════════════════════════════════
# DIVERGENCE RULES  (only evaluated while "Live sync" is on, except kind="stale")
#   kind="compare"          |a - b| > abs_tol  (+ rel_tol * max(|a|,|b|))   for two variable ids
#   kind="object_position"  camera position of an object vs position predicted from encoder travel
#   kind="stale"            variable has not been updated for max_age_s
#   hold_s: condition must persist this long before raising the alert
# ════════════════════════════════════════════════════════════════════════════
DIVERGENCE_RULES = [
    dict(id="speed", kind="compare", label="Motor speed vs model", a="rpm", b="model_rpm",
         abs_tol=30, rel_tol=0.0, hold_s=1.0, unit="RPM",
         cause="Bearing wear or friction", action="Preventive maintenance"),
    dict(id="slip", kind="compare", label="Belt speed: encoder vs camera", a="belt_speed_cm_s", b="cam_belt_speed_cm_s",
         abs_tol=3.0, rel_tol=0.1, hold_s=2.0, unit="cm/s",
         cause="Belt slipping on the roller", action="Re-tension the belt, recalibrate the camera"),
    dict(id="objpos", kind="object_position", label="Object position vs prediction",
         abs_tol=4.0, hold_s=0.5, unit="cm",
         cause="Object sliding on the belt", action="Check the belt surface, recalibrate the camera"),
    dict(id="link", kind="stale", label="Telemetry link", var="rpm", max_age_s=2.0, unit="s",
         cause="WiFi / MQTT latency or loss", action="Check WiFi signal, reduce message size"),
]
