"""
config.py  -  EVERYTHING you normally want to edit lives in this file.

  * VARIABLES         what data exists (from the serial link, derived, model or camera)
  * PANELS            which variables are shown in which window
  * PLOTS             which variables are drawn in which time-series chart
  * CONTROLS          widgets that change the digital twin and/or the real conveyor
  * DIVERGENCE_RULES  how physical vs digital mismatches are detected

To add a sensor:   1) add a dict to VARIABLES   2) (optional) add its id to a PANELS / PLOTS entry
To add a control:  add a dict to CONTROLS  (it sends a command to the conveyor and/or changes the twin model)
"""
import math

# ════════════════════════════════════════════════════════════════════════════
# MQTT
#   Only used for the camera/vision pipeline now (vision.py publishes detected objects to
#   VISION["publish_topic"]). The conveyor (motor + encoder) talks over USB serial instead -
#   see SERIAL below - because the ESP32's H-bridge wiring wasn't driving the motor reliably,
#   so the conveyor controller moved back to an Arduino UNO (firmware/arduino_uno_conveyor/).
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
# TOPIC_TELEMETRY / STATUS_TOPIC are no longer real MQTT topics - serial_bridge.py reuses these
# strings as internal keys so VARIABLES (below) and the "cmd/..." control topics don't need to change.
TOPIC_TELEMETRY = f"{TOPIC_PREFIX}/telemetry"   # Arduino -> JSON {"rpm":..,"setpoint":..,"output":..,"dir":"F"}
STATUS_TOPIC = f"{TOPIC_PREFIX}/status"         # unused now (serial has no equivalent to the MQTT Last Will)

# ════════════════════════════════════════════════════════════════════════════
# Serial (Arduino UNO conveyor controller)
#   Protocol (must match firmware/arduino_uno_conveyor/arduino_uno_conveyor.ino):
#     Arduino -> PC   one JSON line every 100 ms, same fields as the old MQTT telemetry payload
#     PC -> Arduino   one line per command: F | R | S   (direction)
#                     V<0..100>                          (speed, % of MAX_RPM)
#                     P<float> | I<float> | D<float>     (Kp / Ki / Kd)
#                     H                                  (keep-alive, arms the Arduino's failsafe)
# ════════════════════════════════════════════════════════════════════════════
SERIAL = dict(
    port="/dev/ttyUSB1",   # set explicitly: ttyUSB0 turned out to be the ESP32-CAM's own USB-serial
                           # connection (also CH340), not the Arduino - confirmed by reading raw serial:
                           # it printed the ESP32-CAM firmware's "WiFi RSSI ... free heap ..." debug line.
    baud=115200,
    reconnect_s=2.0,
    heartbeat_s=1.0,     # send the "H" keep-alive this often; the Arduino stops the motor after
                         # FAILSAFE_MS (3 s) without it, i.e. if app.py dies or the USB link drops
)

# ════════════════════════════════════════════════════════════════════════════
# Conveyor geometry
# ════════════════════════════════════════════════════════════════════════════
BELT_LENGTH_CM = 68.0
ROLLER_RADIUS_CM = 1.25     # drive roller radius (lab guide: r = 5 cm)
MAX_RPM = 300.0            # RPM at 100 % speed - must match MAX_RPM in arduino_uno_conveyor.ino
ENCODER_SIGN = 1           # set to -1 if "Forward" gives negative RPM

# Physical ground-truth reference glued/taped onto the belt, used to check and fine-tune the
# camera calibration (belt_roi_px below). Measured by hand with a ruler:
#   - x_start_cm: distance from the centre of the LEFT roller (= x = 0 cm) to the start of the mark
#   - width_cm:   horizontal width of the mark
# vision.py draws this as a yellow reference on the camera overlay; nudge roi_x1/roi_x2 (Camera
# calibration controls) until the yellow mark lines up with the real mark in the image.
CALIBRATION_MARK = dict(x_start_cm=28.7, width_cm=3.90)

# HC-SR04 ultrasonic sensor (wired to the Arduino), looking along the belt at the objects on it.
# Belt position of the object = sensor_x_cm + facing * distance (+ facing * object_half_len_cm).
ULTRASONIC = dict(
    sensor_x_cm=BELT_LENGTH_CM + 5.0,   # 5 cm past the far (forward) end of the belt
    facing=-1,                          # -1: looks back towards x = 0 (objects moving Forward approach it)
                                        # +1: if it is mounted before x = 0 looking forward instead
    object_half_len_cm=0.0,             # the sensor sees the object's NEAR face, the camera its centre:
                                        # set to half the object's length (along the belt) to compare them
)


def us_position_cm(distance_cm):
    """Ultrasonic distance -> belt position of the object's centre (cm), or None if nothing is on the belt."""
    if distance_cm is None or distance_cm < 0:          # firmware sends -1 when there is no echo
        return None
    u = ULTRASONIC
    x = u["sensor_x_cm"] + u["facing"] * (distance_cm + u["object_half_len_cm"])
    return x if 0.0 <= x <= BELT_LENGTH_CM else None

# ════════════════════════════════════════════════════════════════════════════
# Web app
# ════════════════════════════════════════════════════════════════════════════
WEB = dict(host="0.0.0.0", port=8050, open_browser=False)
# host="0.0.0.0": listen on every network interface, not just localhost, so other machines on the
# same LAN can open http://<orangepi-ip>:8050 (find the IP with `hostname -I` on the OrangePi).
# open_browser=False: no desktop/browser to launch on a headless board - set True if you do have one.
HISTORY_SECONDS = 60       # length of the time-series plots
HISTORY_DT = 0.1           # minimum spacing between stored points (s)
STALE_AFTER_S = 2.0        # a value older than this is shown as "—" / considered missing

# ════════════════════════════════════════════════════════════════════════════
# Camera + vision
# ════════════════════════════════════════════════════════════════════════════
VISION = dict(
    mode="bgsub",           # "bgsub" = real camera + background subtraction | "sim" = fake objects (no camera) | "off"
    # Camera link. USB serial (firmware/esp32_cam_serial/, PC side camserial.py) - the lab WiFi was too slow:
    #   "serial:/dev/ttyUSB0"          (default 2 Mbaud)   or "serial:/dev/ttyUSB0@921600" if the link is noisy.
    # Both the Arduino and the ESP32-CAM-MB are CH340 -> /dev/ttyUSB0/1 can swap between reboots; the stable names are
    # in /dev/serial/by-id/ (ls -l /dev/serial/by-id) - but two CH340s look alike there too, so the safest check is
    # the "[info] ESP32-CAM: ready ..." line at startup. Without a camera: python sim_camserial.py -> serial:/dev/pts/N
    camera_url="serial:/dev/ttyUSB0",
    # --- WiFi alternative (firmware/esp32_cam/esp32cam_stream.ino): set camera_url to the HTTP stream instead ---
    # ESP32-CAM firmware: stream on port 81, settings endpoint /control on port 80 (also accepts a video file or 0 = webcam).
    # Using the ESP32-CAM's real IP instead of esp32cam.local (mDNS) - avoids depending on avahi-daemon/
    # libnss-mdns being installed, which isn't a given on minimal OrangePi images. Found via nmap -p 80,81
    # <subnet>: the host with both 80 and 81 open is the camera. Re-check this if the camera gets a new
    # DHCP lease (e.g. after a router reboot) - a static DHCP reservation on the router avoids that.
    # camera_url="http://10.82.234.15:81/stream",
    camera_control_url="http://10.82.234.15",     # only used with an http:// camera_url
    # Background subtraction (OpenCV MOG2) - no GPU/ML runtime, cheap enough for a Pi-class board.
    # The belt must be empty and mostly static for a few seconds after startup so it can learn the background.
    process_fps=8,           # max detections per second - the video itself is still shown at full camera
                             # rate (see BgSubVision.loop); lower this first if the OrangePi's CPU can't keep up
    bg_history=500,          # frames used to build the background model
    bg_var_threshold=25,     # MOG2 sensitivity: lower = more sensitive (more false positives from lighting flicker)
    bg_learning_rate=-1,     # -1 = auto (~1/history); 0 = never adapt (background frozen after startup)
    min_area_px=200,         # contour area filters, in pixels of the CAMERA FRAME - tune to your object size while
    max_area_px=20000,       # watching the camera overlay (too low -> noise gets detected, too high -> misses the object)
    track_max_dist_px=80,    # max centroid movement between frames to still count as the same object
    track_max_missed=10,     # frames a track can go undetected before it's dropped
    # Region of the image that contains the belt (x1, y1, x2, y2) in pixels OF THE CAMERA FRAME.
    # x1 -> 0 cm and x2 -> BELT_LENGTH_CM. Tune it live in the "Camera calibration" controls.
    # Defaults are for the firmware's cropped frame: 640x160 (VGA 640x480, middle third kept, see esp32cam_stream.ino).
    # If you change the frame size / crop in the firmware, re-do the calibration.
    # Use CALIBRATION_MARK below (the red square on the belt) as a ground-truth reference while tuning roi_x1/roi_x2:
    # its expected pixel position is drawn on the camera overlay as a yellow marker - move the ROI sliders until
    # the yellow mark lines up with the real red square in the image.
    # Calibrated on the real belt with the crop OFF (full 640x480 frame): 680 px = 68 cm -> 10 px/cm.
    # x2 = 680 is past the frame edge on purpose - it sets the scale, the belt's last cm are just out of view.
    belt_roi_px=(5, 125, 320, 175),
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
# DETECTION FILTERS  (vision.py -> classify / BgSubVision._detect / VisionBase.process)
#   Every blob that background subtraction finds is measured:
#     w_cm    width along the belt            h_frac  height / belt band height (1.0 = whole belt)
#     fill    blob area / bounding box area (circle ~0.8, rectangle ~1.0, reflections usually low)
#     hue     mean OpenCV hue 0..180 (orange ~5-25)   sat / val  mean saturation / brightness 0..255
#   Turn "Show blob measurements" on (page, "Detection filters") to see these numbers on the video and
#   tune the ranges below. Rejected blobs are drawn in magenta with the reason. Each filter has its own
#   on/off switch on the page (defaults = "on" here); to remove one for good, delete its entry here,
#   its switch in CONTROLS and its few lines in vision.py (search for its name).
# ════════════════════════════════════════════════════════════════════════════
DETECTION_FILTERS = dict(
    # the belt's own white seam: ~2 cm wide, spans the whole belt height
    belt_line=dict(on=True, w_cm=(1.0, 3.5), min_h_frac=0.75),
    # drop blobs that match no OBJECT_CLASSES entry (reflections, hands, noise)
    known_only=dict(on=True),
    # a blob must be detected in this many detection frames in a row before it is shown (reflections flicker)
    persistence=dict(on=True, min_hits=3),
    # drop tracks that stay in place while the belt moves (a fixed glare spot). OFF by default: it would also
    # hide a real object that is stuck/blocked - exactly what the "Object position" divergence should catch.
    static=dict(on=False, belt_travel_cm=4.0, max_move_cm=1.0),
    # print the measurements of every blob on the video (for tuning)
    debug=dict(on=True),
)

# The objects that are really put on the belt. A blob gets the label of the FIRST class whose ranges all
# contain its measurements (keys you leave out are not checked). Add a dict here to recognise a new object.
OBJECT_CLASSES = [
    dict(label="orange cilinder", w_cm=(2.0, 5.0), h_frac=(0.10, 0.5), fill=(0.1, 1.0), hue=(0, 50), sat=(0, 255)),
    dict(label="metal box", w_cm=(5.0, 16.0), h_frac=(0.5, 1.0), fill=(0.5, 1.0), hue=(0, 180), sat=(0, 200)),
]

# ════════════════════════════════════════════════════════════════════════════
# VARIABLES
#   source="serial"   read from the Arduino UNO telemetry line (topic + key, see serial_bridge.py).
#                     key=None -> scalar payload. "topic" here is just a dispatch key, not a real MQTT topic.
#   source="derived"  computed here: fn(values_dict) -> value
#   source="model"    written by twin_model.py
#   source="vision"   written by vision.py
#   Optional: unit, fmt, color (plot line), warn=(lo,hi), alarm=(lo,hi)  (None = no limit), hidden=True
# ════════════════════════════════════════════════════════════════════════════
VARIABLES = [
    # --- from the Arduino UNO (firmware/arduino_uno_conveyor/), over serial -----------------
    dict(id="rpm", label="Motor speed", unit="RPM", source="serial", topic=TOPIC_TELEMETRY, key="rpm", fmt="{:.0f}", color="#1f5fbf"),
    dict(id="setpoint", label="PID setpoint", unit="RPM", source="serial", topic=TOPIC_TELEMETRY, key="setpoint", fmt="{:.0f}", color="#8a97a8"),
    dict(id="pwm", label="PWM output", unit="/255", source="serial", topic=TOPIC_TELEMETRY, key="output", fmt="{:.0f}",
         warn=(None, 230), alarm=(None, 250)),
    dict(id="direction", label="Direction", source="serial", topic=TOPIC_TELEMETRY, key="dir"),
    # speed % the Arduino actually holds - if this doesn't follow the slider, commands aren't arriving
    dict(id="speed_ack", label="Speed received", unit="%", source="serial", topic=TOPIC_TELEMETRY,
         key="speed_pct", fmt="{:.0f}"),
    # HC-SR04 wired to the Arduino (optional, see the firmware's ULTRASONIC flag) - independent of the camera.
    # distance_cm = -1 -> no object in range; obj_speed_cm_s > 0 -> object approaching the sensor.
    dict(id="us_distance_cm", label="Distance (ultrasonic)", unit="cm", source="serial", topic=TOPIC_TELEMETRY,
         key="distance_cm", fmt="{:.1f}"),
    dict(id="us_obj_speed_cm_s", label="Approach speed (ultrasonic)", unit="cm/s", source="serial", topic=TOPIC_TELEMETRY,
         key="obj_speed_cm_s", fmt="{:.1f}"),
    # EXAMPLE - uncomment to add a motor current sensor (ACS712) in 2 minutes:
    # dict(id="current", label="Motor current", unit="A", source="serial", topic=TOPIC_TELEMETRY, key="current",
    #      fmt="{:.2f}", warn=(None, 1.5), alarm=(None, 2.5)),

    # --- derived from other variables -------------------------------------
    # v is a dict {id: value} with the fresh values. If a key is missing the variable is skipped.
    dict(id="belt_speed_cm_s", label="Belt speed (encoder)", unit="cm/s", source="derived", fmt="{:.1f}",
         color="#1f5fbf",
         fn=lambda v: ENCODER_SIGN * v["rpm"] / 60.0 * 2 * math.pi * ROLLER_RADIUS_CM),   # Eq. (1) of the lab guide
    # Calibration helper: the ROLLER_RADIUS_CM that would make the encoder belt speed equal the camera's.
    # Read it with objects riding on the belt at a steady speed, then put that value in ROLLER_RADIUS_CM.
    dict(id="radius_fit_cm", label="Roller radius matching camera", unit="cm", source="derived", fmt="{:.2f}",
         fn=lambda v: ROLLER_RADIUS_CM * v["cam_belt_speed_cm_s"] / v["belt_speed_cm_s"]
         if abs(v["belt_speed_cm_s"]) > 2.0 and abs(v["cam_belt_speed_cm_s"]) > 1.0 else None),
    dict(id="us_x_cm", label="Object position (ultrasonic)", unit="cm", source="derived", fmt="{:.1f}",
         color="#2f8f5b", fn=lambda v: us_position_cm(v["us_distance_cm"])),

    # --- digital twin model (twin_model.py) --------------------------------
    dict(id="model_rpm", label="Model speed", unit="RPM", source="model", fmt="{:.0f}", color="#d98a00"),
    dict(id="model_belt_speed_cm_s", label="Model belt speed", unit="cm/s", source="model", fmt="{:.1f}", color="#d98a00"),

    # --- from the camera (vision.py) ----------------------------------------
    dict(id="cam_belt_speed_cm_s", label="Belt speed (camera)", unit="cm/s", source="vision", fmt="{:.1f}", color="#7a3fb0"),
    dict(id="cam_objects", label="Objects on belt", source="vision", fmt="{:.0f}"),
    dict(id="cam_rejected", label="Blobs rejected by filters", source="vision", fmt="{:.0f}"),
    dict(id="cam_fps", label="Vision rate", unit="fps", source="vision", fmt="{:.1f}"),
    dict(id="cam_lag_ms", label="Camera delay (above minimum)", unit="ms", source="vision", fmt="{:.0f}",
         warn=(None, 300), alarm=(None, 800)),
]

# ════════════════════════════════════════════════════════════════════════════
# PANELS  (variable "windows").  Variables not listed anywhere go to an "Other" panel.
# ════════════════════════════════════════════════════════════════════════════
PANELS = [
    dict(title="Drive", vars=["rpm", "setpoint", "pwm", "direction", "speed_ack", "belt_speed_cm_s"]),
    dict(title="Camera", vars=["cam_belt_speed_cm_s", "radius_fit_cm", "cam_objects", "cam_rejected", "cam_fps", "cam_lag_ms"]),
    dict(title="Ultrasonic", vars=["us_distance_cm", "us_x_cm", "us_obj_speed_cm_s"]),
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
#   target:  "camera" -> ESP32-CAM setting (needs camera_var): "set <var> <val>" over serial, or HTTP /control
#            "real"  -> sent to the conveyor (over serial, via serial_bridge.py) only
#            "twin"  -> change the digital model / vision parameter only (what-if, calibration, fault injection)
#            "both"  -> sent to the conveyor and changes the twin
#   topic:   "conveyor/cmd/<key>" - serial_bridge.py maps <key> to a one-letter serial command (see README §3).
#            payload: optional template, "{value}" is replaced
#   model_var: name of the twin parameter this control writes (state.params[model_var])
# ════════════════════════════════════════════════════════════════════════════
_R = VISION["belt_roi_px"]
CONTROLS = [
    dict(id="direction", group="Motion", label="Direction", kind="buttons",
         options=[("Forward", "F"), ("Stop", "S"), ("Reverse", "R")], default="S",
         target="both", topic=f"{TOPIC_PREFIX}/cmd/direction", model_var="direction"),
    dict(id="speed", group="Motion", label="Speed setpoint", unit="%", kind="slider",
         min=20, max=100, step=1, default=20,   # below ~20 % (60 RPM) the belt stalls; use Stop to stop
         target="both", topic=f"{TOPIC_PREFIX}/cmd/speed", model_var="speed_pct"),

    # Defaults = the gains validated on the real conveyor; serial_bridge.py pushes them (and the speed)
    # to the Arduino every time it connects, so keep these in sync with the firmware's kp/ki/kd.
    dict(id="kp", group="PID gains", label="Kp", kind="number", step=0.1, default=0.4,
         target="real", topic=f"{TOPIC_PREFIX}/cmd/kp"),
    dict(id="ki", group="PID gains", label="Ki", kind="number", step=0.1, default=0.5,
         target="real", topic=f"{TOPIC_PREFIX}/cmd/ki"),
    dict(id="kd", group="PID gains", label="Kd", kind="number", step=0.05, default=0.0,
         target="real", topic=f"{TOPIC_PREFIX}/cmd/kd"),

    # Which sensor places the object(s) drawn on the belt and listed in the objects table:
    dict(id="pos_source", group="Belt view", label="Object position from", kind="buttons",
         options=[("Camera", "camera"), ("Ultrasonic", "ultrasonic")], default="camera",
         target="twin", model_var="pos_source"),

    # Sorting cycle: where the cube position comes from (shown inside the "Sorting cycle" card):
    dict(id="cycle_source", group="Sorting cycle", label="Stop positions from", kind="buttons",
         options=[("Camera", "camera"), ("Ultrasonic", "ultrasonic")], default="camera",
         target="twin", model_var="cycle_source"),

    # Twin-only "what-if / fault injection" (lab guide section 5.3):
    dict(id="fault_loss", group="Twin what-if", label="Inject speed loss (twin only)", unit="%", kind="slider",
         min=0, max=60, step=1, default=0, target="twin", model_var="fault_loss_pct"),

    # Camera calibration - changes take effect immediately, watch the overlay in the camera window:
    dict(id="roi_x1", group="Camera calibration", label="Belt start x (px)  = 0 cm", kind="number", step=5, default=_R[0],
         target="twin", model_var="roi_x1"),
    dict(id="roi_x2", group="Camera calibration", label=f"Belt end x (px)  = {BELT_LENGTH_CM:g} cm", kind="number", step=5, default=_R[2],
         target="twin", model_var="roi_x2"),
    dict(id="roi_y1", group="Camera calibration", label="Belt top y (px)", kind="number", step=5, default=_R[1],
         target="twin", model_var="roi_y1"),
    dict(id="roi_y2", group="Camera calibration", label="Belt bottom y (px)", kind="number", step=5, default=_R[3],
         target="twin", model_var="roi_y2"),
    dict(id="flip_x", group="Camera calibration", label="Mirror image", kind="switch", default=VISION["flip_x"],
         target="twin", model_var="flip_x"),

    # Detection filters (see DETECTION_FILTERS) - flip them live to see what each one does:
    dict(id="filt_belt_line", group="Detection filters", label="Ignore the belt's white line", kind="switch",
         default=DETECTION_FILTERS["belt_line"]["on"], target="twin", model_var="filt_belt_line"),
    dict(id="filt_known_only", group="Detection filters", label="Only ball / box (drop reflections)", kind="switch",
         default=DETECTION_FILTERS["known_only"]["on"], target="twin", model_var="filt_known_only"),
    dict(id="filt_persistence", group="Detection filters",
         label=f"Must be seen {DETECTION_FILTERS['persistence']['min_hits']} frames in a row", kind="switch",
         default=DETECTION_FILTERS["persistence"]["on"], target="twin", model_var="filt_persistence"),
    dict(id="filt_static", group="Detection filters", label="Drop blobs not moving with the belt", kind="switch",
         default=DETECTION_FILTERS["static"]["on"], target="twin", model_var="filt_static"),
    dict(id="filt_debug", group="Detection filters", label="Show blob measurements on video", kind="switch",
         default=DETECTION_FILTERS["debug"]["on"], target="twin", model_var="filt_debug"),

    # ESP32-CAM settings: "set <camera_var> <int>" over serial (or http://<camera>/control?var=..&val=.. on WiFi).
    # They apply on the camera immediately; over serial the current values are also re-sent every time the camera
    # (re)connects or reboots.
    # Moving belt tip: switch "Auto exposure" off and use a short manual exposure to reduce motion blur (add light!).
    dict(id="cam_crop", group="Camera settings", label="Crop to belt band (low latency)", kind="switch", default=True,
         target="camera", camera_var="crop"),
    # ^ turn off to see the WHOLE frame the camera captures (e.g. to re-aim it or sanity-check calibration);
    # the frame size changes when you flip this (640x160 cropped vs 640x480 full), so re-do the ROI below
    # after toggling it - the background model also needs a moment to relearn at the new frame size.
    dict(id="cam_quality", group="Camera settings", label="JPEG quality (10 best … 63 smallest)", kind="slider",
         min=10, max=63, step=1, default=16, target="camera", camera_var="quality"),
    dict(id="cam_brightness", group="Camera settings", label="Brightness", kind="slider",
         min=-2, max=2, step=1, default=0, target="camera", camera_var="brightness"),
    dict(id="cam_contrast", group="Camera settings", label="Contrast", kind="slider",
         min=-2, max=2, step=1, default=0, target="camera", camera_var="contrast"),
    dict(id="cam_saturation", group="Camera settings", label="Saturation", kind="slider",
         min=-2, max=2, step=1, default=1, target="camera", camera_var="saturation"),   # -2 washed yellow into white
    dict(id="cam_autoexp", group="Camera settings", label="Auto exposure", kind="switch", default=True,
         target="camera", camera_var="exposure_ctrl"),
    dict(id="cam_exposure", group="Camera settings", label="Manual exposure (0 … 1200)", kind="slider",
         min=0, max=1200, step=10, default=300, target="camera", camera_var="aec_value"),
    dict(id="cam_autogain", group="Camera settings", label="Auto gain", kind="switch", default=True,
         target="camera", camera_var="gain_ctrl"),
    dict(id="cam_gain", group="Camera settings", label="Manual gain (0 … 30)", kind="slider",
         min=0, max=30, step=1, default=0, target="camera", camera_var="agc_gain"),

    # EXAMPLE of a switch that talks to the real conveyor with a JSON payload:
    # dict(id="pid_on", group="PID gains", label="PID enabled", kind="switch", default=True,
    #      target="real", topic=f"{TOPIC_PREFIX}/cmd/pid", payload='{"enabled": {value}}'),
]

# E-stop button (top right): sets this control to this value, which publishes it like any other control.
ESTOP = dict(control="direction", value="S")

# ════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION CYCLE (orchestrator.py) - the automatic sorting state machine
#   IDLE -> RUN -> OBJECT_DETECTED -> CLASSIFYING -> TO_ARM -> PUSHING -> RUN       (classes in push=)
#                                                 \-> REVERSE (back to the origin) -> RUN   (reverse_classes=)
#   any fault / E-stop -> SAFE_STOP (belt stopped, arm STOP) until "Reset" on the page
# One cube at a time between the camera and the arm. Distances in cm of belt travel (encoder).
# ════════════════════════════════════════════════════════════════════════════
CYCLE = dict(
    model="classifier/models/cube_classifier.npz",   # NumPy CNN (python -m classifier.export, then copy here)
    # Where the cube is: "camera" (stop line in the image + encoder to the arm) or "ultrasonic" (HC-SR04 readings,
    # stops at the sensor distances below). Default only - switch it live in the page's "Sorting cycle" card.
    position_source="camera",
    ultrasonic=dict(
        classify_cm=30.0,      # <<< MEASURE: sensor reading with the cube centred under the camera (classification)
        arm_margin_cm=4.0,     # a cube must be at least this far before the classification point to be armed
        overshoot_cm=8.0,      # found this far past the point before stopping -> missed
        latency_s=0.35,        # telemetry period (0.1 s) + median + command + motor stop; the stop is sent this early.
                               # TUNE with logs/cycles.csv: us_final_cm and arm_us_cm - us_target_cm should be ~0
        median_n=3,            # median of this many readings (HC-SR04 jitter)
    ),
    run_speed_pct=25,          # belt speed while the cycle runs (% of MAX_RPM)
    stop_x_frac=0.5,           # stop when the cube centre reaches this fraction of the image width (0.5 = middle)
    arm_margin_px=40,          # a cube must be seen at least this far BEFORE the stop line to be armed (no re-trigger)
    stop_latency_s=0.25,       # frame age + command + motor stop; the stop command is sent this much earlier.
                               # TUNE with logs/cycles.csv: x_final_px (camera) and arm_travel_cm - target_cm
                               # (arm) should be ~0; positive = stopped too late -> increase, negative -> decrease
    overshoot_px=80,           # cube found this far PAST the line before stopping -> missed (goes to the end)
    settle_s=0.6,              # wait after stopping before taking the classification frames (no motion blur)
    classify_timeout_s=4.0,    # not enough frames within this time (after settle) -> decide with the faces seen
                               # so far (none seen -> "other")
    uncertain=False,           # False: the class is simply the highest mean probability over the frames.
                               # True: old rule with "dudoso" (low confidence / no majority, params.json "decision")
    # Push positions: class -> arm routine and where the cube must stop so that routine hits it:
    #   cm     camera mode: belt travel (encoder) from the camera stop line (image centre) to the push point
    #   us_cm  ultrasonic mode: sensor reading with the cube at the push point  <<< MEASURE (page: "Distance (ultrasonic)")
    push={
        "solid":   dict(routine="PUSH1", cm=2.0, us_cm=19.0),
        "stripes": dict(routine="PUSH2", cm=23.0, us_cm=15.0),
        "corners": dict(routine="PUSH3", cm=28.0, us_cm=10.0),
    },
    # Classes sent back to the origin: belt reverses until the cube has left the belt, then forward again.
    reverse_classes=("other", "dudoso"),
    reverse_cm=30.0,           # <<< MEASURE: stop line -> origin end of the belt + a few cm (None = time only).
                               # Ultrasonic mode: done when the sensor sees the cube past the belt start instead
    reverse_s=5.0,             # ...or this long in reverse, whichever comes first (None = distance only)
    # After each push (and HOME): is the cube still at the push point? (camera, or the sensor for ultrasonic cubes)
    verify=dict(
        on=True,
        settle_s=0.5,          # wait after HOME before looking
        window_s=1.0,          # look for this long ...
        min_seen_frac=0.5,     # ... cube seen in at least this fraction of the frames / readings -> push failed
        tol_cm=5.0,            # "at the push point" = within this distance of it
        attempts=2,            # pushes per cube; between them the belt moves the cube back onto the push point
        correct_tol_cm=0.5,    # closer than this: push again without moving the belt
        in_sandbox=False,      # Sandbox: the arm is simulated, the cube never really leaves -> skip the check
    ),
    failed_reverse_s=7.0,      # every push failed: belt in reverse for this long (no distance check), then forward
    coast_s=0.3,               # the belt still moves this long after a stop command (serial + motor stopping);
                               # after that a stopped belt does not move the twin's cube / the odometer
    speed_lag_s=0.4,           # lag of the Arduino's filtered speed (EMA 0.2/0.8 at 10 Hz -> 0.1*0.8/0.2 s): the
                               # travel integrated from it is that far behind, the odometer adds lag * speed.
                               # TUNE both with logs/cycles.csv: arm_travel_cm should match the real cube position
    clear_cm=12.0,             # any class in neither list: belt travel before looking for the next cube (belt end)
    stopped_cm_s=0.8,          # |belt speed| below this = stopped
    stop_timeout_s=3.0,        # stop commanded, measured speed still above stopped_cm_s after this -> fault.
                               # The Arduino reports a FILTERED speed (EMA 0.2/0.8 at 10 Hz, 0 only after 400 ms
                               # without encoder pulses), so "stopped" is seen ~1-1.5 s after the belt really stops
    stall_s=3.0,               # belt commanded to run but not moving for this long -> fault
    camera_timeout_s=2.0,      # no camera frame for this long while running -> fault
    telemetry_timeout_s=2.0,   # no conveyor telemetry for this long (Live sync) -> fault
    sim_push_s=2.5,            # Sandbox: the arm is simulated, "push" takes this long
    sim_home_s=1.0,            # ...and HOME this long
    log_csv="logs/cycles.csv", # one line per cube (times, class, latencies, deadline checks)
    publish_topic=f"{TOPIC_PREFIX}/cycle",   # state + every decision are published here (JSON)
    # Deadlines (lab section 6.1) checked for every cube; a miss is logged as a warning and counted
    deadlines=dict(stop_s=2.0,       # stop command -> belt seen stopped (filtered speed, see stop_timeout_s)
                   classify_s=3.0,   # stop command -> class decided
                   arm_s=10.0),      # push command -> arm reports IDLE
)

# Robotic arm (EEZYbotARM Mk2, firmware coppeliaSim_panaderia_ws/firmware/esp32_eezybot_linear): plain-text MQTT.
# Its MQTT_HOST must be this PC's IP (mosquitto.conf listens on the LAN).
ARM = dict(
    cmd_topic="eezybot/cmd",         # we send the routine of CYCLE["push"] ("PUSH1" / "PUSH2" / "PUSH3") or "STOP"
    status_topic="eezybot/status",   # it answers "PUSH1 inicio", "DONE", "IDLE" (routine finished), "ERR ..."
    online_topic="eezybot/online",   # retained "1" / "0"
    home_after_push=True,            # after each push send home_cmd and wait for it before the belt moves again
    home_cmd="HOME",
    done_text="IDLE",
    error_prefix="ERR",
    timeout_s=15.0,                  # push sent, no IDLE within this time -> fault
)

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
    dict(id="link", kind="stale", label="Telemetry link", var="rpm", max_age_s=2.0, unit="s",
         cause="USB serial link lost or Arduino reset", action="Check the USB cable/port, reflash if needed"),
]
