"""
classifier/capture.py - build the dataset with the ESP32-CAM (or tune the segmentation on recordings).

Grabs frames, runs classifier/preprocess.py live and shows every stage, so you see exactly what the
model will get. Saves, per shot:
    dataset/<session>/<class>/<id>_band.png   the raw cut band (lossless copy of the decoded frame)
    dataset/<session>/<class>/<id>_mask.png   the object mask
    dataset/<session>/<class>/<id>_crop.png   the model input (input_size x input_size, BGR)
    dataset/<session>/manifest.jsonl          one JSON line per shot (class, bbox, size, params used, ...)
The raw band is kept so the whole dataset can be re-segmented later if params.json changes
(python -m classifier.preprocess ... or a later re-crop script).

Usage (repo root, conda env "Control"):
    python -m classifier.capture --url http://10.82.234.15 --session bench_white_01 --tune
    python -m classifier.capture --source some_belt_video.mp4 --tune       # tune params on a recording
    python -m classifier.capture --source dataset/bench_white_01/class_a    # re-check saved bands

Keys:  1..9 pick class (from params.json "model.classes")   SPACE save (only if status "ok")
       f force-save the band even if not ok                a  toggle auto-save every --interval s while "ok"
       b save an EMPTY background frame (false-trigger stats)   p write tuned params to params.json
       q / ESC quit
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import requests

from classifier import preprocess as pp

TUNE = [  # (trackbar name, params path, max, scale)
    ("canny lo", ("edges", "lo"), 300, 1),
    ("canny hi", ("edges", "hi"), 400, 1),
    ("chroma x10", ("edges", "chroma_gain"), 40, 10),
    ("min edge px", ("edges", "min_edge_px"), 80, 1),
    ("close r", ("mask", "close_r"), 12, 1),
    ("merge gap x10cm", ("blob", "merge_gap_cm"), 80, 10),
    ("open r", ("mask", "open_r"), 6, 1),
    ("min area", ("blob", "min_area_px"), 2000, 1),
    ("side ratio x100", ("blob", "max_side_ratio"), 200, 100),
    ("rect fill x100", ("blob", "min_rect_fill"), 100, 100),
]
CAM_TUNE = ("saturation", "brightness", "contrast")   # camera /control vars, -2..2 (trackbar 0..4)


# ── frame sources ───────────────────────────────────────────────────────────────────────────
class Esp32Cam:
    def __init__(self, base, crop, cam_settings):
        self.base = base.rstrip("/")
        self.s = requests.Session()
        self.name = self.base
        self.set("crop", int(crop))
        for var, val in cam_settings.items():        # the stream firmware boots with saturation -2 (washed out)
            self.set(var, val)

    def set(self, var, val):
        try:
            r = self.s.get(f"{self.base}/control", params=dict(var=var, val=int(val)), timeout=3)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"warning: could not set {var}={val} on the camera: {e}")

    def read(self):
        r = self.s.get(f"{self.base}/capture", timeout=5)
        r.raise_for_status()
        return cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)


class Files:
    def __init__(self, path):
        exts = (".png", ".jpg", ".jpeg", ".bmp")
        if os.path.isdir(path):
            self.files = sorted(f for f in glob.glob(os.path.join(path, "**", "*"), recursive=True)
                                if f.lower().endswith(exts) and not f.endswith(("_mask.png", "_crop.png")))
        else:
            self.files = sorted(glob.glob(path))
        self.i, self.name = 0, path

    def read(self):                                   # stays on the current image; n / SPACE-less browse with 'n'
        return cv2.imread(self.files[self.i]) if self.files else None


class Video:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
        self.name = str(src)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:                                    # loop a file
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None


# ── saving ──────────────────────────────────────────────────────────────────────────────────
class Writer:
    def __init__(self, root, session, p, source):
        self.dir = os.path.join(root, session)
        os.makedirs(self.dir, exist_ok=True)
        self.manifest = os.path.join(self.dir, "manifest.jsonl")
        self.p, self.source = p, source
        self.n = sum(1 for _ in open(self.manifest)) if os.path.exists(self.manifest) else 0

    def save(self, cls, band, det):
        d = os.path.join(self.dir, cls)
        os.makedirs(d, exist_ok=True)
        sid = f"{time.strftime('%Y%m%d_%H%M%S')}_{self.n:05d}"
        cv2.imwrite(os.path.join(d, f"{sid}_band.png"), band)
        if det is not None and det.mask is not None:
            cv2.imwrite(os.path.join(d, f"{sid}_mask.png"), det.mask)
        if det is not None and det.crop is not None:
            cv2.imwrite(os.path.join(d, f"{sid}_crop.png"), det.crop)
        rec = dict(id=sid, cls=cls, t=time.time(), source=self.source, shape=list(band.shape),
                   status=det.status if det else "background",
                   bbox=list(det.bbox) if det and det.bbox else None,
                   area_px=det.area_px if det else 0, size_cm=list(det.size_cm) if det and det.size_cm else None,
                   angle=det.angle if det else 0.0,
                   corners=det.corners.tolist() if det is not None and det.corners is not None else None,
                   params=self.p)
        with open(self.manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.n += 1
        return sid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="ESP32-CAM base URL, e.g. http://10.82.234.15 (uses /capture and /control)")
    ap.add_argument("--source", help="instead of the camera: image folder / glob, video file, or webcam index")
    ap.add_argument("--crop", type=int, default=1, help="ask the camera for the cut middle band (1) or full frame (0)")
    ap.add_argument("--session", default=time.strftime("session_%Y%m%d_%H%M"),
                    help="one name per sitting: background, light, camera height... (train/test split is by session)")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--interval", type=float, default=0.7, help="auto-save period (s)")
    ap.add_argument("--tune", action="store_true", help="show trackbars for the segmentation params")
    ap.add_argument("--params", default=pp.PARAMS_PATH)
    a = ap.parse_args()

    p = pp.load_params(a.params)
    classes = p["model"]["classes"]
    if a.url:
        src = Esp32Cam(a.url, a.crop, p.get("camera", {}))
    elif a.source and (os.path.isdir(a.source) or any(ch in a.source for ch in "*?")
                       or a.source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))):
        src = Files(a.source)
    elif a.source:
        src = Video(a.source)
    else:
        ap.error("give --url or --source")
    writer = Writer(a.out, a.session, p, src.name)

    win = "capture  (1..9 class, SPACE save, a auto, b background, p params, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if a.tune:
        for name, (sec, key), mx, sc in TUNE:
            cv2.createTrackbar(name, win, int(round(p[sec][key] * sc)), mx, lambda _v: None)
        if isinstance(src, Esp32Cam):
            for var in CAM_TUNE:
                cv2.createTrackbar(f"cam {var}", win, int(p.get("camera", {}).get(var, 0)) + 2, 4, lambda _v: None)

    cls_i, auto, last_save, msg, msg_t = 0, False, 0.0, "", 0.0
    while True:
        try:
            band = src.read()
        except requests.RequestException as e:
            print(f"camera error: {e}")
            time.sleep(1.0)
            continue
        if band is None:
            print("no frame (end of source?)")
            break

        if a.tune:
            for name, (sec, key), _mx, sc in TUNE:
                v = cv2.getTrackbarPos(name, win)
                p[sec][key] = v / sc if sc != 1 else v
            if isinstance(src, Esp32Cam):
                cam = p.setdefault("camera", {})
                for var in CAM_TUNE:
                    v = cv2.getTrackbarPos(f"cam {var}", win) - 2
                    if cam.get(var) != v:
                        cam[var] = v
                        src.set(var, v)

        det = pp.process(band, p, keep_stages=True)
        view = pp.debug_view(band, det, p, width=1100)
        status = (f"class [{cls_i + 1}] {classes[cls_i]}   saved {writer.n}   auto {'ON' if auto else 'off'}"
                  f"   session {a.session}")
        if isinstance(src, Files) and src.files:
            status += f"   img {src.i + 1}/{len(src.files)} (n/N next/prev)"
        bar = np.zeros((44, view.shape[1], 3), np.uint8)
        cv2.putText(bar, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if time.time() - msg_t < 2.0:
            cv2.putText(bar, msg, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, np.vstack([bar, view]))

        def note(s):
            nonlocal msg, msg_t
            msg, msg_t = s, time.time()
            print(s)

        if auto and det.status == "ok" and time.time() - last_save >= a.interval:
            note(f"auto-saved {writer.save(classes[cls_i], band, det)} -> {classes[cls_i]}")
            last_save = time.time()

        k = cv2.waitKey(30 if not isinstance(src, Files) else 50) & 0xFF
        if k in (ord("q"), 27):
            break
        if ord("1") <= k <= ord("9") and k - ord("1") < len(classes):
            cls_i = k - ord("1")
        elif k == ord(" "):
            if det.status == "ok":
                note(f"saved {writer.save(classes[cls_i], band, det)} -> {classes[cls_i]}")
            else:
                note(f"not saved: status {det.status} (use f to force)")
        elif k == ord("f"):
            note(f"force-saved {writer.save(classes[cls_i], band, det)} -> {classes[cls_i]} ({det.status})")
        elif k == ord("a"):
            auto = not auto
        elif k == ord("b"):
            note(f"background saved {writer.save('_background', band, None)}")
        elif k == ord("p"):
            pp.save_params(p, a.params)
            note(f"params written to {a.params}")
        elif isinstance(src, Files) and k in (ord("n"), ord("N")) and src.files:
            src.i = (src.i + (1 if k == ord("n") else -1)) % len(src.files)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
