# Lab 03 classifier - Rubik cube faces (ESP32-CAM, int8 CNN)

Classes (`params.json` → `model.classes`), for 3x3 and 4x4 cubes alike:

| class | face | destination |
|---|---|---|
| `solid` | whole face one colour | servo position 1 |
| `stripes` | 3 bands: both edge bands the same colour, middle band another (1 row in 3x3, 2 rows in 4x4) | servo position 2 |
| `corners` | one colour except the 4 corners in another | belt end |
| `other` | any other combination | belt end (with `dudoso`) |

Use the conda env **`Control`** (TensorFlow 2.20, OpenCV with GUI), run from the repo root.

| File | Role |
|---|---|
| `params.json` | every segmentation / crop / decision constant + camera colour settings - shared with the firmware |
| `preprocess.py` | Canny (Y + chroma) → close/fill/open → **square filter** (+ fragment merging) → upright 64×64 face |
| `capture.py` | live dataset capture from the ESP32-CAM, tuning trackbars (incl. camera saturation/brightness/contrast) |
| `recrop.py` | re-run the pipeline on all saved frames after a params change; `--rename old=new` for classes |
| `augment.py` | frame-level scale ×0.7–1.3 + any rotation (camera height / pose) → re-detect; face-level rot90/flip, colour permutation, light, blur, noise, JPEG |
| `train.py` | small CNN (61 k params), blocked K-fold CV + final model → `classifier/runs/<name>/` (report, confusion, errors) |
| `live.py` | live test on the PC: camera → detection → CNN → 5-frame vote → class / dudoso + destination; keys 1..4 save the frame with its true class |
| `tests/` | `python -m unittest discover -s classifier/tests -t .` |

## How the cube is found
Edges → solid mask → each blob's rotated rectangle must be a **square**: sides ratio ≤ `max_side_ratio`,
blob fills ≥ `min_rect_fill` of it, side within `side_cm`, and 4 **corner probes** on the blob (rejects discs,
hands, strips). A face that falls apart into parallel strips (stripes face with an invisible middle band) is
re-joined by **fragment merging** (`merge_gap_cm`, `merge_max_deg`). The square's 4 corners are warped to an
upright 64×64 image of the face only - no background, no rotation, no size (classes do not depend on size).

## Taking the photos (no conveyor yet)
1. Camera: use **VGA** (the crop band 640×160 or full 640×480), not QVGA - at QVGA a cube is only ~20 px wide at
   the processing resolution, which is the lower limit (45°-rotated small cubes get lost). If the camera only gives
   320×240, check the serial monitor for `no PSRAM found` (PSRAM must be enabled in the board menu).
2. Colours: the stream firmware boots with saturation −2; `capture.py` sends `params.json → camera`
   (saturation +2 by default) at start and has live trackbars. Yellow must look clearly different from white.
3. `crop.px_per_cm`: measure it at your camera height (cube side in px / side in cm) - it only affects the
   `side_cm` size filter and the reported size.
4. `python -m classifier.capture --url serial:/dev/ttyUSB0 --session <name> --tune`
   (USB cable, firmware `firmware/esp32_cam_serial/`; `--url http://<cam-ip>` for the WiFi firmware;
   `--crop 0` sends the full 640×480 frame - better for photos on a desk, slower: ~5 fps instead of ~15)
   - 1..4 pick the class, **a** auto-save while slowly rotating/moving the cube, **b** empty-background shots,
     **p** save tuned params.
5. One session per setup (background, light, camera height); the train/test split is by session.
6. Target 150–300 shots per class, both cube sizes, every colour combination you can make.

## Training
`python -m classifier.train --sessions 10` (add more sessions as you collect them). Augmentation: rotations
0/90/180/270, light (brightness, contrast, gamma, white balance, mild saturation/hue) and, by default, a random
colour permutation (`--colour-perm 0.8`; the classes depend on which stickers share a colour, not on the colour).
Validation is blocked cross-validation (contiguous blocks of shots per class) because consecutive shots are
near-duplicates. Session 10 (95 faces): CV accuracy 0.79 with rotation + light only, 0.82 with colour permutation.
