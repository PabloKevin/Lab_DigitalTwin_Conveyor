"""
classifier/live.py - test the trained model on the PC with the live camera, before putting it on the ESP32.

Same chain as the final system: frame -> preprocess.process() (find the square face, upright 64x64 crop)
-> CNN -> decision rule of params.json "decision" (the vision "debounce" the firmware will use too):
    collect n_frames predictions of the same object; the winner needs >= min_agree votes and mean confidence
    >= tau, and must not be "other" -> class; otherwise DUDOSO. The decision is held until the object leaves.
-> destination from params.json "destinations".

    python -m classifier.live --url serial:/dev/ttyUSB0            # ESP32-CAM on the USB cable
    python -m classifier.live --url http://10.82.234.15            # ESP32-CAM on WiFi
    python -m classifier.live --source 0                           # laptop webcam / --source video.mp4
    python -m classifier.live --url serial:/dev/ttyUSB0 --crop 0   # full frame (cube on a desk)
    --model classifier/runs/<run>/model.keras    (default: the run with the best CV macro-F1)

Keys:  q quit   r reset the vote   SPACE pause
       1..4 save the current frame into dataset/<--session>/<class> (the true class) - collect the cases the model
       gets wrong and retrain with them
"""
import argparse
import glob
import json
import os
import time
from collections import Counter

import cv2
import numpy as np

from classifier import capture as cap
from classifier import preprocess as pp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def best_model():
    """Run folder with the highest CV macro-F1 (report.json), else the newest model.keras."""
    runs = []
    for rep in glob.glob(os.path.join(ROOT, "classifier", "runs", "*", "report.json")):
        d = os.path.dirname(rep)
        if os.path.exists(os.path.join(d, "model.keras")):
            f1 = json.load(open(rep)).get("cv", {}).get("macro_f1", -1)
            runs.append((f1, os.path.getmtime(rep), d))
    if not runs:
        raise SystemExit("no trained model in classifier/runs/ - run python -m classifier.train first")
    return os.path.join(max(runs)[2], "model.keras")


class Voter:
    """Decision rule over consecutive frames of ONE object (ported 1:1 to the firmware later).
    update() per frame -> dict(state, votes, decision, reason)."""

    def __init__(self, classes, n_frames, min_agree, tau, lost_frames=3, jump_px=60):
        self.classes, self.n, self.k, self.tau = classes, n_frames, min_agree, tau
        self.lost_frames, self.jump_px = lost_frames, jump_px
        self.reset()

    def reset(self):
        self.probs, self.center, self.misses = [], None, 0
        self.decision, self.reason = None, ""

    def update(self, ok, probs=None, center=None):
        if not ok:
            self.misses += 1
            if self.misses > self.lost_frames:          # object gone: ready for the next one
                self.reset()
            return self.view()
        self.misses = 0
        if self.center is not None and np.hypot(center[0] - self.center[0], center[1] - self.center[1]) > self.jump_px:
            self.reset()                                # a different object
        self.center = center
        if self.decision is None:
            self.probs.append(np.asarray(probs, float))
            if len(self.probs) >= self.n:
                self._decide()
        return self.view()

    def _decide(self):
        P = np.stack(self.probs[-self.n:])
        votes = P.argmax(1)
        win, cnt = Counter(votes.tolist()).most_common(1)[0]
        conf = float(P[votes == win, win].mean())
        name = self.classes[win]
        if cnt < self.k:
            self.decision, self.reason = "dudoso", f"only {cnt}/{self.n} agree"
        elif conf < self.tau:
            self.decision, self.reason = "dudoso", f"{name} conf {conf:.2f} < {self.tau}"
        elif name == "other":
            self.decision, self.reason = "dudoso", f"other ({cnt}/{self.n}, {conf:.2f})"
        else:
            self.decision, self.reason = name, f"{cnt}/{self.n} votes, conf {conf:.2f}"

    def view(self):
        return dict(votes=[self.classes[int(p.argmax())] for p in self.probs[-self.n:]],
                    decision=self.decision, reason=self.reason)


# ── drawing ─────────────────────────────────────────────────────────────────────────────────
COL = {"solid": (60, 180, 75), "stripes": (230, 140, 30), "corners": (40, 160, 230), "other": (150, 150, 150),
       "dudoso": (60, 60, 230)}


def panel(face, probs, classes, vote, dest, h, fps, ms):
    w = 300
    pnl = np.full((max(h, 330), w, 3), 32, np.uint8)
    if face is not None:
        pnl[8:136, 8:136] = cv2.resize(cv2.cvtColor(face, cv2.COLOR_RGB2BGR), (128, 128), interpolation=cv2.INTER_NEAREST)
    y = 20
    for i, c in enumerate(classes):
        p = float(probs[i]) if probs is not None else 0.0
        cv2.putText(pnl, c, (146, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.rectangle(pnl, (146, y + 14), (146 + int(140 * p), y + 24), COL.get(c, (200, 200, 200)), -1)
        cv2.putText(pnl, f"{p:.2f}", (250, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        y += 30
    cv2.putText(pnl, "votes: " + " ".join(v[:4] for v in vote["votes"]), (8, 158),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    d = vote["decision"]
    if d:
        cv2.rectangle(pnl, (8, 170), (w - 8, 230), COL.get(d, (200, 200, 200)), -1)
        cv2.putText(pnl, d.upper(), (16, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pnl, f"-> {dest}", (16, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pnl, vote["reason"], (8, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    else:
        cv2.putText(pnl, "collecting..." if vote["votes"] else "no cube", (16, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(pnl, f"{fps:4.1f} fps   preprocess+CNN {ms:4.0f} ms", (8, pnl.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return pnl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="serial:/dev/ttyUSB0[@baud] or http://<ip>")
    ap.add_argument("--source", help="webcam index, video file, or image folder")
    ap.add_argument("--crop", type=int, default=1, help="camera: middle band (1) or full frame (0)")
    ap.add_argument("--model", help="model.keras (default: best run in classifier/runs/)")
    ap.add_argument("--session", default=time.strftime("live_%Y%m%d"), help="where keys 1..4 save frames")
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset"))
    a = ap.parse_args()

    p = pp.load_params()
    classes = p["model"]["classes"]
    dec = p["decision"]
    model_path = a.model or best_model()
    print(f"model: {model_path}")
    import keras
    model = keras.models.load_model(model_path)
    n = p["crop"]["input_size"]
    model(np.zeros((1, n, n, 3), np.uint8), training=False)          # warm-up (first call builds the graph)

    if a.url:
        src = (cap.SerialCamSource if a.url.startswith("serial:") else cap.Esp32Cam)(a.url, a.crop, p.get("camera", {}))
    elif a.source and (os.path.isdir(a.source) or a.source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))):
        src = cap.Files(a.source)
    elif a.source:
        src = cap.Video(a.source)
    else:
        ap.error("give --url or --source")
    writer = cap.Writer(a.out, a.session, p, src.name)
    voter = Voter(classes, dec["n_frames"], dec["min_agree"], dec["tau"])

    win = "live classifier  (q quit, r reset, SPACE pause, 1..4 save as true class)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    paused, frame, msg, msg_t, stamps = False, None, "", 0.0, []
    while True:
        if not paused or frame is None:
            try:
                frame = src.read()
            except Exception as e:                     # noqa: BLE001 - camera hiccup: keep the window alive
                print(f"camera: {e}")
                time.sleep(0.5)
                continue
            if frame is None:
                break
            t0 = time.time()
            det = pp.process(frame, p)
            probs = face = None
            if det.status == "ok":
                face = cv2.cvtColor(det.crop, cv2.COLOR_BGR2RGB)
                probs = model(face[None], training=False).numpy()[0]
                x1, y1, x2, y2 = det.bbox
                vote = voter.update(True, probs, ((x1 + x2) / 2, (y1 + y2) / 2))
            else:
                vote = voter.update(False)
            ms = (time.time() - t0) * 1000
            stamps = [s for s in stamps if t0 - s < 2.0] + [t0]
        fps = (len(stamps) - 1) / max(1e-6, stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0

        view = frame.copy()
        if det.corners is not None:
            col = COL.get(vote["decision"] or "", (60, 200, 60))
            cv2.polylines(view, [det.corners.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        elif det.bbox:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), (0, 200, 255), 1)
        dest = p["destinations"].get(vote["decision"] or "", "")
        pn = panel(face, probs, classes, vote, dest, view.shape[0], fps, ms)
        if view.shape[0] < pn.shape[0]:
            view = np.vstack([view, np.zeros((pn.shape[0] - view.shape[0], view.shape[1], 3), np.uint8)])
        out = np.hstack([view, pn])
        if time.time() - msg_t < 2.0:
            cv2.putText(out, msg, (8, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, out)

        k = cv2.waitKey(1 if not isinstance(src, cap.Files) else 30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            voter.reset()
        elif k == ord(" "):
            paused = not paused
        elif ord("1") <= k < ord("1") + len(classes):
            c = classes[k - ord("1")]
            sid = writer.save(c, frame, det)
            msg, msg_t = f"saved {sid} as {c} -> {writer.dir}", time.time()
            print(msg)
        elif isinstance(src, cap.Files) and k in (ord("n"), ord("N")) and src.files:
            src.i = (src.i + (1 if k == ord("n") else -1)) % len(src.files)
            voter.reset()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
