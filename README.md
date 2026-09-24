# Conveyor digital twin (ESP32 · MQTT · camera + YOLO · Dash)

A local web page that mirrors a 60 cm conveyor in real time, lets you change the real machine and the
digital model from the same screen, and flags when physical and digital behaviour diverge.

```
 ESP32 (PID, encoder, L298N) ──MQTT──►  Mosquitto  ◄──MQTT──  app.py  ◄── browser (localhost:8050)
 ESP32-CAM ────── HTTP MJPEG stream ──────────────────────►  vision.py (YOLO + tracking)
```

## 1. Try it without hardware (5 minutes)

```bash
pip install -r requirements.txt        # ultralytics is only needed for real YOLO vision
mosquitto -c mosquitto.conf -v         # terminal 1  (install Mosquitto first, see below)
python sim_esp32.py                    # terminal 2  fake conveyor
# in config.py set VISION["mode"] = "sim"   (fake objects riding the belt)
python app.py                          # terminal 3  opens http://127.0.0.1:8050
```

In the page: set **Direction = Forward** and **Speed = 10 %**. Then try
* *Twin what-if → Inject speed loss* (twin only): the model slows down, the **Motor speed vs model** rule fires.
* `python sim_esp32.py --loss 25`: the *real* conveyor loses speed instead. Same alert, opposite cause.
* `VISION["sim_slip"] = 0.7` in `config.py`: objects slip on the belt, **belt slip** and **object position** rules fire.
* **Sandbox** (top right): commands no longer go to the machine, you only play with the twin. **E-stop** always sends *Stop*.

Installing Mosquitto: Windows → installer from mosquitto.org; Ubuntu → `sudo apt install mosquitto mosquitto-clients`; macOS → `brew install mosquitto`.
`mosquitto.conf` opens port 1883 to the LAN without passwords: fine for a lab, not for the internet.

## 2. With the real conveyor

1. **ESP32**: open `firmware/esp32_conveyor/esp32_conveyor.ino`, set WiFi + the PC's IP, install *PubSubClient*, flash.
   It runs the PID locally (like the lab guide) and speaks the protocol below. The motor stops if MQTT is lost for 3 s.
2. **Camera**: flash the *CameraWebServer* example on the ESP32-CAM (VGA or QVGA is enough), open `http://<ip>` once to check it,
   then set `VISION["camera_url"] = "http://<ip>:81/stream"` and `VISION["mode"] = "yolo"`.
3. **Geometry**: set `ROLLER_RADIUS_CM` and `MAX_RPM` for *your* drive. With the guide's numbers (r = 5 cm, 300 RPM) the belt runs
   94 cm/s at 100 %, so a 60 cm belt is crossed in under a second. Use the real gear ratio.
4. **Calibrate the camera**: in *Camera calibration* type the pixel columns where the belt starts (= 0 cm) and ends (= 60 cm) and
   the top/bottom rows. The video shows the ROI and a ruler every 10 cm; adjust until it matches the belt. Use *Mirror image* if
   forward moves to the left. Put a ruler or tape marks on the belt to check.
5. **Objects**: `yolo11n.pt` (COCO) only knows everyday classes. For your own objects, train a model (Ultralytics) and set
   `VISION["model"] = "my_objects.pt"`. Use `VISION["classes"]` to filter.

## 3. MQTT protocol

| Topic | Direction | Payload |
|---|---|---|
| `conveyor/telemetry` | ESP32 → PC | `{"rpm":-12.0,"setpoint":100.0,"output":120,"dir":"F"}` (10 Hz, rpm is signed) |
| `conveyor/status` | ESP32 → PC | `online` / `offline` (retained + Last Will) |
| `conveyor/cmd/direction` | PC → ESP32 | `F`, `R` or `S` |
| `conveyor/cmd/speed` | PC → ESP32 | `0`–`100` (% of `MAX_RPM`) |
| `conveyor/cmd/kp` `ki` `kd` | PC → ESP32 | number |
| `conveyor/vision/objects` | PC → anyone | `{"ts":…, "belt_speed_cm_s":…, "objects":[{"id":1,"label":"box","x_cm":23.4,"speed_cm_s":15.8}]}` |

Everything under `conveyor/#` appears in the **MQTT monitor** at the bottom of the page, handy for debugging.

## 4. Customising (all in `config.py`)

**Add a sensor** (e.g. motor current sent by the ESP32 as `"current"` in the telemetry JSON):
```python
dict(id="current", label="Motor current", unit="A", source="mqtt", topic=TOPIC_TELEMETRY, key="current",
     fmt="{:.2f}", warn=(None, 1.5), alarm=(None, 2.5)),
```
Then put `"current"` in a `PANELS` entry (a window) and/or a `PLOTS` entry (a chart). Variables not placed anywhere land in an "Other" window.
A sensor on its own topic with a plain number payload: `topic="conveyor/temp", key=None`.

**Add a computed value**: `source="derived"` with `fn=lambda v: v["current"] * 12` (watts). It is skipped while an input is missing.

**Add a control** that changes the real conveyor:
```python
dict(id="pid_on", group="PID gains", label="PID enabled", kind="switch", default=True,
     target="real", topic="conveyor/cmd/pid", payload='{"enabled": {value}}'),
```
`kind`: `slider`, `number`, `buttons`, `switch`. `target`: `"real"` (publish), `"twin"` (only changes the model), `"both"`.
A control with `model_var="x"` writes `state.params["x"]`, which `twin_model.step_model()` and `vision.py` can read.

**Change the twin physics**: edit `step_model()` in `twin_model.py` (currently a first-order lag, `MODEL["tau_s"]`). Add new
model outputs with `state.set_value("model_xyz", …)` and declare them as `source="model"` variables.

**Add a divergence rule**: append to `DIVERGENCE_RULES`; `kind="compare"` needs two variable ids and a tolerance, e.g.
```python
dict(id="power", kind="compare", label="Power vs model", a="power", b="model_power", abs_tol=2, hold_s=2, unit="W",
     cause="Overload", action="Inspect the belt"),
```
New kinds go in `divergence._check()`.

## 5. How the checks work

* **Motor speed vs model**: the twin model receives the same commands as the real conveyor; measured RPM should follow it.
* **Belt speed: encoder vs camera**: encoder RPM × 2πr against the median speed of tracked objects → detects belt slip.
* **Object position vs prediction**: every object's position from the camera against where the encoder travel says it should be.
* **Telemetry link**: telemetry older than `max_age_s`.
* Rules only run in **Live sync** (except the link check) and need `hold_s` seconds of persistence before raising an alert.

## 6. Files

| File | Role |
|---|---|
| `config.py` | everything you edit: variables, panels, plots, controls, rules, camera, MQTT |
| `app.py` | the web page (layout is generated from `config.py`), `assets/style.css` = look |
| `mqtt_bridge.py` | MQTT in/out |
| `twin_model.py` | simulated conveyor, derived variables, belt travel, runs the rules |
| `divergence.py` | rule engine |
| `vision.py` | MJPEG reader, YOLO tracking, pixel→cm, speed, annotated video, simulated vision |
| `controls.py` | what happens when a widget changes |
| `state.py` | shared thread-safe state |
| `sim_esp32.py` | fake ESP32 for testing |
| `firmware/esp32_conveyor/` | ESP32 sketch |

## 7. Known limitations

* Frame timestamps are the PC's arrival time, so WiFi/stream latency (typically 100–300 ms) is not compensated. It shows up as a small constant position offset.
* Object speed comes from tracking the object's centre over ~1 s; very fast belts or low `infer_fps` make it noisy.
* Run `python app.py` (not with Flask's reloader): the reloader would start every thread and MQTT client twice.
