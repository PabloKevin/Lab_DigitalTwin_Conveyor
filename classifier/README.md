# Lab 03 classifier (ESP32-CAM, int8 CNN)

Use the conda env **`Control`** (TensorFlow 2.20, OpenCV with GUI): `conda activate Control`, run from the repo root.

| File | Role |
|---|---|
| `params.json` | every segmentation / crop / decision constant - shared with the firmware |
| `preprocess.py` | band → Canny (Y + chroma) → close/fill → largest valid blob → fixed-scale 64×64 crop |
| `capture.py` | live dataset capture from the ESP32-CAM (`/capture`), tuning trackbars |
| `tests/` | `python -m unittest classifier.tests.test_preprocess -v` |

## Taking the bench photos (no conveyor yet)
1. Flash the current `firmware/esp32_cam/esp32cam_stream.ino` (it already has `/capture` and the crop).
2. Mount the camera at the **same height and angle as over the belt** - `crop.px_per_cm` (10 px/cm) must stay
   true, otherwise sizes in the crops are wrong. Check: a ruler in the image should read ~10 px per cm.
3. `python -m classifier.capture --url http://<cam-ip> --session white_desk_lamp --tune`
   - first tune with the trackbars until only the object is boxed green, press **p** to save params.
   - **b** a few times with nothing in view (background shots, used to measure false triggers).
   - pick the class with **1..4**, toggle **a** (auto-save every 0.7 s) and slowly rotate / move the object.
4. One **session per setup**: change background (≥ 3 plain ones, one close to the belt colour), light and
   session name. The train/test split is done by session, so the test set is never the same sitting.
5. `other` = 10-20 random objects that are none of the 3 classes.
6. Target 150-300 saved shots per class.
