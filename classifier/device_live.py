"""
classifier/device_live.py - watch what the ESP32-CAM classifier (firmware/esp32_cam_classifier) decides.
The model runs ON THE ESP32; the PC only shows the results it sends over the USB serial cable.

    python -m classifier.device_live --url serial:/dev/ttyUSB0             # results + preview frames (2 fps)
    python -m classifier.device_live --url serial:/dev/ttyUSB0 --preview 0 # results only (fastest classifying)
    python -m classifier.device_live --url serial:/dev/ttyUSB0 --crop 0    # full frame (cube on a desk)

Window: preview frame with the face found by the ESP32, the 64x64 face its model saw, class probabilities,
votes, decision + destination, and per-stage times measured on the ESP32 (capture / RGB convert / find face / CNN).
Every decision is also printed (and appended to --log as CSV).
Keys: q quit, r reset the vote on the ESP32, p preview on/off.
"""
import argparse
import csv
import os
import time

import cv2
import numpy as np

from classifier import capture as cap
from classifier import live
from classifier import preprocess as pp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="serial:/dev/ttyUSB0")
    ap.add_argument("--preview", type=int, default=1,
                    help="preview frames per second (0 = off; each one costs the ESP32 ~100 ms of JPEG encoding)")
    ap.add_argument("--crop", type=int, default=1, help="1 = belt band (middle third), 0 = full frame")
    ap.add_argument("--log", default=None, help="CSV file to append every decision to")
    a = ap.parse_args()

    p = pp.load_params()
    classes = p["model"]["classes"]
    port, baud = cap.camserial.parse_url(a.url)
    decisions = []

    def on_result(res):
        if res.get("new"):
            decisions.append(res)
            print(f"{time.strftime('%H:%M:%S')}  #{res['n']:5d}  {res['dec']:>8} -> {res['dest']:<9} {res['why']}"
                  f"   ({sum(res.get('ms', {}).values()):.0f} ms/frame on the ESP32)")
            if a.log:
                new = not os.path.exists(a.log)
                with open(a.log, "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["time", "frame", "decision", "destination", "why", "size_cm"] + classes)
                    w.writerow([time.time(), res["n"], res["dec"], res["dest"], res["why"], res.get("cm")] + res.get("p", []))

    settings = {"crop": a.crop, **p.get("camera", {})}
    cam = cap.camserial.SerialCam(port, baud, settings=settings, fps=max(1, a.preview), stream=a.preview > 0,
                                  on_result=on_result, on_log=lambda m: print(f"  [esp32] {m}"))
    win = "ESP32-CAM classifier  (q quit, r reset vote, p preview on/off)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    preview, img, stamps, last_n = a.preview > 0, None, [], None
    while True:
        res = cam.result
        if cam.frame is not None and preview:
            img = cam.frame.decode()
        view = img.copy() if img is not None else np.zeros((480, 640, 3), np.uint8)
        if res is not None:
            if res.get("n") != last_n:
                last_n = res.get("n")
                stamps = [s for s in stamps if res["arrival"] - s < 3.0] + [res["arrival"]]
            col = live.COL.get(res.get("dec") or "", (60, 200, 60))
            if res.get("c"):
                pts = np.array(res["c"], np.float32).reshape(4, 2).astype(np.int32)
                cv2.polylines(view, [pts], True, col, 2, cv2.LINE_AA)
            elif res.get("box"):
                x1, y1, x2, y2 = res["box"]
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 200, 255), 1)
            if not preview:
                cv2.putText(view, "preview off (p) - boxes drawn on a blank frame", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        fps = (len(stamps) - 1) / max(1e-6, stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        vote = dict(votes=(res or {}).get("votes", []), decision=(res or {}).get("dec"), reason=(res or {}).get("why", ""))
        probs = np.array(res["p"]) if res and res.get("p") else None
        ms = sum((res or {}).get("ms", {}).values()) if res else 0
        pn = live.panel(cam.face if res and res.get("st") == "ok" else None, probs, classes, vote,
                        (res or {}).get("dest", ""), view.shape[0], fps, ms)
        if res and res.get("ms"):
            t = res["ms"]
            cv2.putText(pn, "ESP32 ms: " + "  ".join(f"{k} {v:.0f}" for k, v in t.items()),
                        (8, pn.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 220, 180), 1, cv2.LINE_AA)
        cv2.putText(pn, f"link {cam.status}   decisions {len(decisions)}", (8, pn.shape[0] - 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        hh = max(view.shape[0], pn.shape[0])
        pad = lambda im: np.vstack([im, np.zeros((hh - im.shape[0], im.shape[1], 3), np.uint8)])
        cv2.imshow(win, np.hstack([pad(view), pad(pn)]))
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("r"):
            cam.command("reset")
        elif k == ord("p"):
            preview = not preview
            cam.command(f"preview {int(preview)}")
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
