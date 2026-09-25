# Conveyor digital twin (Arduino UNO · pyserial · ESP32-CAM · background subtraction · Dash)

A local web page that mirrors a 60 cm conveyor in real time, lets you change the real machine and the
digital model from the same screen, and flags when physical and digital behaviour diverge. Runs fine
on a small headless board (no GPU needed) - no PyTorch/YOLO anywhere in the stack.

```
 Arduino UNO (PID, encoder, XY-15AS) ──USB serial──►  app.py  ◄── browser (localhost:8050)
 ESP32-CAM ────────── HTTP MJPEG stream ─────────────►  vision.py (bg subtraction + tracking) ──MQTT──► Mosquitto (optional, outbound only)
```

The conveyor (motor + encoder + PID) is driven by an Arduino UNO over a direct USB-serial link
(`serial_bridge.py`) - an ESP32 was tried first but its H-bridge wiring didn't drive the motor
reliably, see `firmware/esp32_conveyor/` if you want to revisit that path. The camera is unrelated
to this and still an ESP32-CAM talking plain HTTP; MQTT/Mosquitto is now only used one-way, for
`vision.py` to broadcast detected objects (nothing in this app reads that topic back). Object
detection is background subtraction (OpenCV MOG2) + contours, not a neural network - deliberately
light enough to run on a Pi-class board (e.g. an Orange Pi with 4 GB RAM, no GPU); the trade-off is
it can tell *something* moved and roughly how big it is, but not what it is (no class label).

## 1. Try it without hardware (5 minutes)

```bash
pip install -r requirements.txt
python sim_conveyor.py                 # terminal 1  fake conveyor over a virtual serial port
# it prints a path like /dev/pts/4 - copy it into config.py's SERIAL["port"] (replacing "auto")
# in config.py also set VISION["mode"] = "sim"   (fake objects riding the belt)
python app.py                          # terminal 2
```

`app.py` listens on `WEB["host"] = "0.0.0.0"` by default, so open `http://<board-ip>:8050` from any machine
on the same LAN (find the board's IP with `hostname -I`), not just `http://127.0.0.1:8050` on the board itself.
If it doesn't load from another machine, check the board's firewall allows port 8050 (e.g. `sudo ufw allow 8050/tcp`).

In the page: set **Direction = Forward** and **Speed = 10 %**. Then try
* *Twin what-if → Inject speed loss* (twin only): the model slows down, the **Motor speed vs model** rule fires.
* `python sim_conveyor.py --loss 25`: the *real* conveyor loses speed instead. Same alert, opposite cause.
* `VISION["sim_slip"] = 0.7` in `config.py`: objects slip on the belt, **belt slip** and **object position** rules fire.
* **Sandbox** (top right): commands no longer go to the machine, you only play with the twin. **E-stop** always sends *Stop*.

`sim_conveyor.py` uses a Unix pty pair, so it only runs on Linux/macOS; on Windows use WSL, or wire up a
real Arduino. `sim_conveyor.py` picks a new pty path every run - re-copy it into `config.py` each time.

Mosquitto is optional (only needed if you want `vision.py`'s object broadcasts to go somewhere): Windows →
installer from mosquitto.org; Ubuntu → `sudo apt install mosquitto mosquitto-clients`; macOS → `brew install mosquitto`.
`mosquitto.conf` opens port 1883 to the LAN without passwords: fine for a lab, not for the internet.

## 2. With the real conveyor

1. **Arduino UNO**: open `firmware/arduino_uno_conveyor/arduino_uno_conveyor.ino`, install the *PID* library
   (PID_v1, by Brett Beauregard), flash. It runs the PID locally and speaks the serial protocol below.
   Plug it into the PC via USB, then in `config.py` set `SERIAL["port"]` - leave it `"auto"` to let
   `serial_bridge.py` pick the first port that looks like an Arduino, or set it explicitly (check with
   `ls /dev/tty*` before/after plugging it in - usually `/dev/ttyACM0` or `/dev/ttyUSB0` on Linux). The
   motor stops if the app's `H` keep-alive (sent every second) is missing for 3 s - i.e. `app.py` died or
   the USB link dropped. That failsafe only arms once the app has connected, so you can still drive it by
   hand from the Arduino IDE Serial Monitor (115200 baud, "Newline"): `V50`, `F`, `R`, `S`, like the old sketch.
   Opening the port resets the UNO; `app.py` then pushes the page's speed and PID gains to it, but leaves
   the belt stopped until you press a direction again.
2. **Camera** (ESP32-CAM, see `firmware/esp32_cam/README.md`): the twin works with the stream on `:81/stream` and the
   `/control` endpoint of your firmware. In `config.py` set `VISION["camera_url"]` and `VISION["camera_control_url"]`
   (IP or `esp32cam.local`) and `VISION["mode"] = "bgsub"`. If `.local` does not resolve on Ubuntu:
   `sudo apt install avahi-daemon libnss-mdns`, or just use the IP printed on the camera's serial monitor.
   By default the firmware (`CROP_MIDDLE_THIRD`) captures VGA (640x480) and crops to the middle third vertically before
   sending, since only the belt band is useful and this cuts WiFi latency a lot — so the frame the PC actually receives
   is **640x160**. Set `CROP_MIDDLE_THIRD 0` in the firmware to send full, hardware-JPEG-encoded frames instead (less
   ESP32 CPU, more bytes over WiFi) — then re-do the ROI, since the frame height is no longer cropped.
3. **Geometry**: set `ROLLER_RADIUS_CM` and `MAX_RPM` for *your* drive. With the guide's numbers (r = 5 cm, 300 RPM) the belt runs
   94 cm/s at 100 %, so a 60 cm belt is crossed in under a second. Use the real gear ratio.
4. **Calibrate the camera**: in *Camera calibration* type the pixel columns where the belt starts (= 0 cm) and ends (= 60 cm) and
   the top/bottom rows. The video shows the ROI, a ruler every 10 cm, and a yellow marker at `CALIBRATION_MARK` (a small
   physical mark measured by hand on the real belt, in `config.py`) — adjust the ROI sliders until that marker lines up
   with the real mark in the image. Use *Mirror image* if forward moves to the left.
5. **Objects**: with the belt running empty, wait a few seconds after startup for `bg_history` frames to build the
   background model, then place an object on the belt - it should get boxed on the camera overlay. If it's missed or
   noise gets boxed instead, tune `min_area_px`/`max_area_px` (contour size filter, in pixels) and `bg_var_threshold`
   (sensitivity) in `config.py`'s `VISION` dict while watching the overlay. There's no object classification (no
   `"label"` beyond a generic `"object"`) since there's no neural network in this pipeline - only position/speed.

## 3. Serial protocol (conveyor) and MQTT (camera only)

| Line | Direction | Payload |
|---|---|---|
| telemetry | Arduino → PC | `{"rpm":-12.0,"setpoint":100.0,"output":120,"dir":"F","distance_cm":58.5,"obj_speed_cm_s":0.0}` (10 Hz, rpm is signed; the `distance_cm`/`obj_speed_cm_s` fields need the optional HC-SR04, see the firmware's `ULTRASONIC` flag) |
| direction | PC → Arduino | `F`, `R` or `S` |
| speed | PC → Arduino | `V0`–`V100` (% of `MAX_RPM`) |
| PID gains | PC → Arduino | `P<float>`, `I<float>`, `D<float>` |

`config.py`'s `CONTROLS` still declare MQTT-style topics (`conveyor/cmd/direction`, etc.) - `serial_bridge.py`
translates those into the lines above, so you don't touch `config.py`/`controls.py` to work with the new transport.

| Topic | Direction | Payload |
|---|---|---|
| `conveyor/vision/objects` | PC → anyone (MQTT) | `{"ts":…, "belt_speed_cm_s":…, "objects":[{"id":1,"label":"box","x_cm":23.4,"speed_cm_s":15.8}]}` |

The last raw telemetry line and any MQTT traffic under `conveyor/#` both appear in the **Comm monitor** at the
bottom of the page, handy for debugging either link.

## 4. Customising (all in `config.py`)

**Add a sensor** (e.g. motor current, added to the Arduino's telemetry JSON as `"current"`):
```python
dict(id="current", label="Motor current", unit="A", source="serial", topic=TOPIC_TELEMETRY, key="current",
     fmt="{:.2f}", warn=(None, 1.5), alarm=(None, 2.5)),
```
Then put `"current"` in a `PANELS` entry (a window) and/or a `PLOTS` entry (a chart). Variables not placed anywhere land in an "Other" window.
Since the Arduino only sends one JSON line, add the field there too (see how `distance_cm`/`obj_speed_cm_s` are added in `arduino_uno_conveyor.ino`) - `key` just picks it out of that same line.

**Add a computed value**: `source="derived"` with `fn=lambda v: v["current"] * 12` (watts). It is skipped while an input is missing.

**Add a control** that changes the real conveyor:
```python
dict(id="accel_limit", group="PID gains", label="Accel limit", kind="number", default=50,
     target="real", topic="conveyor/cmd/accel"),
```
`kind`: `slider`, `number`, `buttons`, `switch`. `target`: `"real"` (sent to the Arduino), `"twin"` (only changes the model), `"both"`.
A control with `model_var="x"` writes `state.params["x"]`, which `twin_model.step_model()` and `vision.py` can read.
For `target="real"`/`"both"`, `topic` must be `conveyor/cmd/<key>` - `serial_bridge.py`'s `publish()` maps `<key>` to a
one-letter serial command (`direction`→as-is, `speed`→`V`, `kp`/`ki`/`kd`→`P`/`I`/`D`); add a new `<key>` case there
**and** a matching `case` in the Arduino's command switch if you add a control the firmware doesn't already understand
(like `accel_limit` above) - otherwise the command is silently dropped.

**Camera settings from the page**: the *Camera settings* controls (`target="camera"`) call your firmware's `/control?var=..&val=..`
(quality, brightness, contrast, saturation, exposure, gain). For a moving belt, turn *Auto exposure* off and use a short manual
exposure with more light to reduce motion blur. Add another with `camera_var="..."` for any variable your firmware handles.

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
* **Telemetry link**: telemetry (now over serial) older than `max_age_s`.
* Rules only run in **Live sync** (except the link check) and need `hold_s` seconds of persistence before raising an alert.

## 6. Files

| File | Role |
|---|---|
| `config.py` | everything you edit: variables, panels, plots, controls, rules, camera, serial, MQTT |
| `app.py` | the web page (layout is generated from `config.py`), `assets/style.css` = look |
| `serial_bridge.py` | conveyor in/out over USB serial (Arduino UNO) |
| `mqtt_bridge.py` | MQTT in/out (camera/vision object broadcasts only) |
| `twin_model.py` | simulated conveyor, derived variables, belt travel, runs the rules |
| `divergence.py` | rule engine |
| `vision.py` | MJPEG reader, background-subtraction + centroid tracking, pixel→cm, speed, annotated video, simulated vision |
| `controls.py` | what happens when a widget changes |
| `state.py` | shared thread-safe state |
| `sim_conveyor.py` | fake Arduino for testing (virtual serial port) |
| `firmware/arduino_uno_conveyor/` | Arduino UNO controller sketch (PID + serial) - the one currently used |
| `firmware/esp32_conveyor/` | ESP32 controller sketch (PID + WiFi/MQTT) - kept for reference, not currently used (H-bridge driving issue) |
| `firmware/esp32_cam/` | notes for the ESP32-CAM sketch and `secrets.h` template |

## 7. Known limitations

* Camera timing: the ESP32-CAM stamps every frame (`X-Timestamp`, time since boot). `vision.py` maps it to PC time using the
  least-delayed frames (`ClockSync`) and compares each frame with the encoder travel *at that moment*, so WiFi jitter does not
  create false divergences. The remaining constant delay (transport minimum) is not compensated; `Camera delay` shows the
  jitter on top of it. Timestamps are only used with the HTTP stream; videos/webcams use the PC clock.
* Object speed comes from tracking the object's centre over ~1 s; very fast belts or a slow camera frame rate make it noisy.
* Background subtraction needs the belt empty and still for a few seconds after startup (or after `flip_x`/ROI/crop
  changes) to relearn the background - an object already on the belt at startup may be missed until it moves away and back.
* Run `python app.py` (not with Flask's reloader): the reloader would start every thread, the serial reader and the MQTT client twice.
* Only one process can hold the Arduino's serial port at a time - close the Arduino IDE's Serial Monitor before running `app.py`.
