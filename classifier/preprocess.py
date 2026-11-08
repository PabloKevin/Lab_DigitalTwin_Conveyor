"""
classifier/preprocess.py - object segmentation + model input, shared by dataset capture, training and
(ported 1:1 to C++) the ESP32-CAM firmware.

Pipeline (all parameters in classifier/params.json):
  band (cut frame, e.g. 640x160)  ->  optional window  ->  downsample x2 (block mean)
  -> 3 channels: Y = (R+2G+B)/4, RG = R-G, BY = B-(R+G)/2      (chroma catches colour-only edges)
  -> 5x5 binomial blur -> Sobel per channel, L1 magnitude, chroma scaled by chroma_gain
  -> per pixel keep the channel with the strongest gradient -> non-max suppression
  -> hysteresis (lo/hi) + drop edge fragments shorter than min_edge_px  (belt texture, glare)
  -> dilate(close_r) -> fill holes (top/bottom-seeded) -> erode(close_r) (edges -> solid mask)
  -> connected components; reject belt seam / too small / too big / too sparse / too elongated
  -> keep the largest valid blob; status "partial" if it touches the window's left/right edge
  -> crop a square of input_size*k px around it (fixed px/cm scale), pixels outside the mask -> fill
  -> k x k block mean -> input_size x input_size BGR

Everything up to the crop is integer arithmetic on purpose, so the C++ port can reproduce it exactly.
Images are BGR uint8 (OpenCV convention); the model input order is converted to RGB in training/export.

CLI (from the repo root, conda env "Control"):
  python -m classifier.preprocess img1.png img2.png --show        # see every stage
  python -m classifier.preprocess dataset/x/*.png --out /tmp/crops
"""
import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.json")


def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return json.load(f)


def save_params(p, path=PARAMS_PATH):
    with open(path, "w") as f:
        json.dump(p, f, indent=2)
        f.write("\n")


@dataclass
class Detection:
    status: str                        # "ok" | "partial" (touches window edge) | "none"
    reason: str = ""                   # why "none" (for the overlay)
    bbox: tuple = None                 # (x1, y1, x2, y2) in full-image px, x2/y2 exclusive
    mask: np.ndarray = None            # full-image uint8 mask (0/255) of the chosen blob
    crop: np.ndarray = None            # input_size x input_size x 3 BGR model input (only if status == "ok")
    area_px: int = 0                   # blob area, full-image px
    size_cm: tuple = None              # (width, height) of the bbox in cm
    scale_k: int = 0                   # block size used for the crop (> crop.scale_k => object was too big)
    blobs: list = field(default_factory=list)   # every component: dict(bbox, area, why) for the overlay
    stages: dict = field(default_factory=dict)  # small debug images (edges, mask, ...)


# ── integer image primitives (mirrored in C++) ──────────────────────────────────────────────
def downsample(img, f):
    """Block mean f x f with rounding. Crops to a multiple of f."""
    if f == 1:
        return img.copy()
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    blk = img[:h, :w].astype(np.int32).reshape(h // f, f, w // f, f, -1).sum(axis=(1, 3))
    return ((blk + f * f // 2) // (f * f)).astype(np.uint8)


def channels(small_bgr):
    b, g, r = [small_bgr[..., i].astype(np.int32) for i in range(3)]
    y = (r + 2 * g + b + 2) >> 2
    rg = r - g
    by = b - ((r + g) >> 1)
    return [y, rg, by]


_BINOM = (1, 4, 6, 4, 1)


def blur5(ch):
    """Separable 5x5 binomial blur, replicate border, integer: (sum + 128) >> 8 (floor shift)."""
    p = np.pad(ch, 2, mode="edge")
    h, w = ch.shape
    t = sum(k * p[:, i:i + w] for i, k in enumerate(_BINOM))
    s = sum(k * t[i:i + h, :] for i, k in enumerate(_BINOM))
    return (s + 128) >> 8


def sobel(ch):
    """3x3 Sobel, replicate border. Returns int32 gx, gy (gy positive downwards)."""
    p = np.pad(ch, 1, mode="edge")
    h, w = ch.shape
    P = lambda dy, dx: p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    gx = (P(-1, 1) + 2 * P(0, 1) + P(1, 1)) - (P(-1, -1) + 2 * P(0, -1) + P(1, -1))
    gy = (P(1, -1) + 2 * P(1, 0) + P(1, 1)) - (P(-1, -1) + 2 * P(-1, 0) + P(-1, 1))
    return gx, gy


def gradient(small_bgr, chroma_gain):
    """Per pixel, the gradient of the channel with the largest (gain-weighted) L1 magnitude."""
    gq = [16, int(round(chroma_gain * 16)), int(round(chroma_gain * 16))]   # fixed point, /16
    best_m = best_gx = best_gy = None
    for ch, q in zip(channels(small_bgr), gq):
        gx, gy = sobel(blur5(ch))
        m = ((np.abs(gx) + np.abs(gy)) * q) >> 4
        if best_m is None:
            best_m, best_gx, best_gy = m, gx, gy
        else:
            sel = m > best_m                       # strict: ties keep the earlier channel (Y first)
            best_m = np.where(sel, m, best_m)
            best_gx = np.where(sel, gx, best_gx)
            best_gy = np.where(sel, gy, best_gy)
    return best_m, best_gx, best_gy


def nonmax(m, gx, gy):
    """Canny non-maximum suppression, 4 directions; keep m > n1 and m >= n2 (as OpenCV). 1-px border = 0."""
    h, w = m.shape
    out = np.zeros_like(m)
    if h < 3 or w < 3:
        return out
    c = m[1:-1, 1:-1]
    ax, ay = np.abs(gx[1:-1, 1:-1]), np.abs(gy[1:-1, 1:-1])
    N = lambda dy, dx: m[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
    horiz = ay * 1000 < ax * 414                   # gradient ~ along x -> compare left/right
    vert = ax * 1000 < ay * 414                    # gradient ~ along y -> compare up/down
    same = (gx[1:-1, 1:-1] * gy[1:-1, 1:-1]) > 0   # diagonal "\" (down-right) vs "/"
    n1 = np.where(horiz, N(0, -1), np.where(vert, N(-1, 0), np.where(same, N(-1, -1), N(-1, 1))))
    n2 = np.where(horiz, N(0, 1), np.where(vert, N(1, 0), np.where(same, N(1, 1), N(1, -1))))
    keep = (c > n1) & (c >= n2)
    out[1:-1, 1:-1] = np.where(keep, c, 0)
    return out


def thresholds(nms, e):
    if e["mode"] == "percentile":
        v = np.sort(nms[nms > 0])
        if v.size == 0:
            return 1 << 30, 1 << 30
        hi = int(v[max(0, math.ceil(e["hi_percentile"] / 100.0 * v.size) - 1)])
        hi = max(hi, int(e["lo"]))                 # never below the fixed floor (empty, flat belt)
        return int(hi * e["lo_ratio"]), hi
    return int(e["lo"]), int(e["hi"])


def hysteresis(nms, lo, hi, min_px):
    """Keep 8-connected weak (>= lo) edge chains that contain a strong (>= hi) pixel and have >= min_px pixels."""
    cand = (nms >= lo) & (nms > 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(nms.shape, np.uint8)
    strong = np.zeros(n, bool)
    strong[np.unique(lab[nms >= hi])] = True
    keep = strong & (stats[:, cv2.CC_STAT_AREA] >= min_px)
    keep[0] = False
    return (keep[lab] * 255).astype(np.uint8)


def dilate(mask, r):
    return cv2.dilate(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) if r > 0 else mask.copy()


def erode(mask, r):
    # out-of-image counts as foreground (OpenCV default), so blobs touching the border are not eaten
    return cv2.erode(mask, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) if r > 0 else mask.copy()


def fill_holes(mask):
    """Background pixels not 4-connected to the TOP or BOTTOM image border become foreground.
    The left/right borders act as walls on purpose: objects enter the band from the side, and an object cut by
    the side border has no edge there - treating the border as a wall closes its outline so it is filled (and
    reported as "partial") instead of staying an open contour."""
    bg = (mask == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    border = np.zeros(n, bool)
    for edge in (lab[0, :], lab[-1, :]):
        border[edge] = True
    border[0] = True                               # label 0 = foreground pixels
    return np.where(border[lab], mask, 255).astype(np.uint8)


# ── pipeline ────────────────────────────────────────────────────────────────────────────────
def segment(small_bgr, p):
    """-> (edges, mask, components) at the downsampled resolution."""
    e = p["edges"]
    m, gx, gy = gradient(small_bgr, e["chroma_gain"])
    nms = nonmax(m, gx, gy)
    lo, hi = thresholds(nms, e)
    edges = hysteresis(nms, lo, hi, e["min_edge_px"])
    r = p["mask"]["close_r"]
    mask = erode(fill_holes(dilate(edges, r)), r)

    b = p["blob"]
    H, W = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        fill = area / float(w * h)
        aspect = max(w, h) / float(max(1, min(w, h)))
        why = ""
        if w <= b["seam_max_w_px"] and h >= b["seam_min_h_frac"] * H:
            why = "seam"
        elif area < b["min_area_px"]:
            why = "small"
        elif area > b["max_area_px"]:
            why = "big"
        elif fill < b["min_fill"]:
            why = "sparse"
        elif aspect > b["max_aspect"]:
            why = "thin"
        touches = x <= b["edge_margin_px"] or x + w >= W - b["edge_margin_px"]
        comps.append(dict(label=i, bbox=(x, y, x + w, y + h), area=area, fill=fill, why=why, touches=touches))
    return edges, mask, lab, comps


def make_crop(img, blob_mask, bbox, p):
    """Square crop at a fixed px/cm scale around bbox, background -> fill, k x k block mean -> input_size."""
    c = p["crop"]
    n = c["input_size"]
    x1, y1, x2, y2 = bbox
    f = p["image"]["downsample"]
    k = c["scale_k"]
    while n * k < max(x2 - x1, y2 - y1) + 2 * f:   # object bigger than the square: shrink (flagged by k)
        k += 1
    side = n * k
    sx, sy = (x1 + x2) // 2 - side // 2, (y1 + y2) // 2 - side // 2
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = np.array(c["fill_bgr"], np.uint8)
    H, W = img.shape[:2]
    ix1, iy1, ix2, iy2 = max(0, sx), max(0, sy), min(W, sx + side), min(H, sy + side)
    if ix2 > ix1 and iy2 > iy1:
        src = img[iy1:iy2, ix1:ix2]
        m = blob_mask[iy1:iy2, ix1:ix2] > 0
        dst = canvas[iy1 - sy:iy2 - sy, ix1 - sx:ix2 - sx]
        dst[m] = src[m]
    return downsample(canvas, k), k


def process(img, p=None, keep_stages=False):
    """Full pipeline on one BGR image (the cut band). Returns a Detection."""
    p = p or load_params()
    f = p["image"]["downsample"]
    H, W = img.shape[:2]
    wx1, wy1, wx2, wy2 = p["image"]["window"] or (0, 0, W, H)
    wx1, wy1, wx2, wy2 = max(0, wx1), max(0, wy1), min(W, wx2), min(H, wy2)
    win = img[wy1:wy2, wx1:wx2]
    small = downsample(win, f)
    edges, mask, lab, comps = segment(small, p)

    to_full = lambda bb: (wx1 + bb[0] * f, wy1 + bb[1] * f, wx1 + bb[2] * f, wy1 + bb[3] * f)
    blobs = [dict(bbox=to_full(cp["bbox"]), area=cp["area"] * f * f, why=cp["why"] or ("edge" if cp["touches"] else ""))
             for cp in comps]
    stages = dict(small=small, edges=edges, mask=mask) if keep_stages else {}

    valid = [cp for cp in comps if not cp["why"]]
    if not valid:
        return Detection("none", reason="no object" if not comps else "rejected", blobs=blobs, stages=stages)
    best = max(valid, key=lambda cp: cp["area"])

    blob_small = (lab == best["label"]).astype(np.uint8) * 255
    full_mask = np.zeros((H, W), np.uint8)
    up = np.repeat(np.repeat(blob_small, f, axis=0), f, axis=1)
    full_mask[wy1:wy1 + up.shape[0], wx1:wx1 + up.shape[1]] = up
    bbox = to_full(best["bbox"])
    ppcm = p["crop"]["px_per_cm"]
    det = Detection("partial" if best["touches"] else "ok", bbox=bbox, mask=full_mask, area_px=best["area"] * f * f,
                    size_cm=((bbox[2] - bbox[0]) / ppcm, (bbox[3] - bbox[1]) / ppcm), blobs=blobs, stages=stages)
    if det.status == "ok":
        det.crop, det.scale_k = make_crop(img, full_mask, bbox, p)
    return det


# ── debug view ──────────────────────────────────────────────────────────────────────────────
def debug_view(img, det, p=None, width=960):
    """Stacked image: band with boxes | edges | mask, plus the model input zoomed on the right."""
    p = p or load_params()
    view = img.copy()
    win = p["image"]["window"]
    if win:
        cv2.rectangle(view, tuple(win[:2]), (win[2] - 1, win[3] - 1), (200, 140, 20), 1)
    for b in det.blobs:
        x1, y1, x2, y2 = b["bbox"]
        if b["why"]:
            cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), (200, 0, 200), 1)
            cv2.putText(view, b["why"], (x1, min(y2 + 12, view.shape[0] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 0, 200), 1, cv2.LINE_AA)
        else:                                      # valid but not the largest: not classified this frame
            cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), (60, 200, 60), 1)
    col = {"ok": (60, 200, 60), "partial": (0, 200, 255)}.get(det.status, (0, 0, 255))
    if det.bbox:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), col, 2)
    txt = det.status if det.status != "none" else f"none ({det.reason})"
    if det.size_cm:
        txt += f"  {det.size_cm[0]:.1f}x{det.size_cm[1]:.1f} cm  k={det.scale_k}"
    cv2.putText(view, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    rows = [view]
    for key in ("edges", "mask"):
        if key in det.stages:
            s = cv2.resize(det.stages[key], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            rows.append(cv2.cvtColor(s, cv2.COLOR_GRAY2BGR))
    left = np.vstack(rows)
    side = left.shape[0]
    crop = det.crop if det.crop is not None else np.full((8, 8, 3), 40, np.uint8)
    right = cv2.resize(crop, (min(side, 256), min(side, 256)), interpolation=cv2.INTER_NEAREST)
    right = np.pad(right, ((0, side - right.shape[0]), (0, 0), (0, 0)))
    out = np.hstack([left, right])
    s = width / out.shape[1]
    return cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run the segmentation pipeline on image files.")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--params", default=PARAMS_PATH)
    ap.add_argument("--show", action="store_true", help="show each stage (any key = next, q = quit)")
    ap.add_argument("--out", help="write <name>_crop.png and <name>_dbg.png here")
    a = ap.parse_args()
    p = load_params(a.params)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
    for path in a.images:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {path}: not an image")
            continue
        det = process(img, p, keep_stages=True)
        print(f"{path}: {det.status} {det.reason} bbox={det.bbox} area={det.area_px} size_cm={det.size_cm}")
        dbg = debug_view(img, det, p)
        if a.out:
            stem = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0])
            cv2.imwrite(stem + "_dbg.png", dbg)
            if det.crop is not None:
                cv2.imwrite(stem + "_crop.png", det.crop)
        if a.show:
            cv2.imshow("preprocess", dbg)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()
