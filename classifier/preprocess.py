"""
classifier/preprocess.py - find the cube face (a square) and turn it into the model input. Shared by dataset
capture, augmentation, training and (ported 1:1 to C++) the ESP32-CAM firmware.

Pipeline (all parameters in classifier/params.json):
  frame / cut band -> optional window -> downsample (block mean)
  -> 3 channels: Y = (R+2G+B)/4, RG = R-G, BY = B-(R+G)/2      (chroma catches colour-only edges)
  -> 5x5 binomial blur -> Sobel per channel, L1 magnitude, chroma scaled by chroma_gain
  -> per pixel keep the channel with the strongest gradient -> non-max suppression
  -> hysteresis (lo/hi) + drop edge fragments shorter than min_edge_px  (belt texture, glare)
  -> dilate(close_r) -> fill holes (top/bottom-seeded) -> erode(close_r) (edges -> solid mask)
  -> open(open_r): erase 1-px edge lines that were never closed (they also glue objects together)
  -> connected components -> rotated bounding rectangle (min-area rect) of each one
  -> SQUARE FILTER: sides ratio <= max_side_ratio, area / rect area >= min_rect_fill, side within side_cm,
     and the 4 corner probes (corner_probe_frac inside each corner) on the object
     (rejects rectangles, discs, strips, hands...). Blobs touching the left/right edge -> "partial".
  -> FRAGMENT MERGING: pieces (strips, stickers, crosses, small squares) that are close and aligned are grouped;
     the group's convex hull is tested as a square too (a face whose stickers partly have no edge against the
     background: stripes -> 2 strips, corners on a dark background -> 4 corner stickers). Largest square wins.
  -> largest square -> its 4 corners (shrunk by inset_frac) -> affine warp of the FULL-RES image
     -> upright input_size x input_size face (no background, no rotation; 90-degree ambiguity is irrelevant
        for the classes and covered by augmentation)

Everything up to the components is integer arithmetic on purpose, so the C++ port can reproduce it exactly.
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
    bbox: tuple = None                 # axis-aligned (x1, y1, x2, y2) in full-image px, x2/y2 exclusive
    corners: np.ndarray = None         # 4x2 float32 face corners in full-image px (tl, tr, br, bl), shrunk by inset
    angle: float = 0.0                 # rotation of the face, degrees in [0, 90)
    mask: np.ndarray = None            # full-image uint8 mask (0/255) of the chosen blob
    crop: np.ndarray = None            # input_size x input_size x 3 BGR upright face (only if status == "ok")
    area_px: int = 0                   # blob area, full-image px
    size_cm: tuple = None              # (side_1, side_2) of the rotated rectangle, cm
    blobs: list = field(default_factory=list)   # every component: dict(bbox, area, why, box) for the overlay
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
def square_test(pts, area, b, ppcm, inside, min_inset_px=0.0):
    """Rotated min-area rect of the points + the square filter. -> (rect, why); why == "" means it is a square.
    inside(x, y) -> bool tells whether a pixel belongs to the object (corner probes)."""
    rect = cv2.minAreaRect(pts.astype(np.float32))
    s1, s2 = rect[1][0] + 1.0, rect[1][1] + 1.0    # pixel centres -> pixel extent
    if max(s1, s2) / max(1.0, min(s1, s2)) > b["max_side_ratio"]:
        return rect, "rect"
    fill = area / (s1 * s2)
    if fill < b["min_rect_fill"]:
        return rect, "shape"
    if not (b["side_cm"][0] <= (s1 + s2) / 2 / ppcm <= b["side_cm"][1]):
        return rect, "size"
    # corner probes: a square has its 4 corners filled; a disc / hand / triangle does not
    # inset along each side = max(frac * side, min_inset_px): the morphology rounds corners by a fixed number of px
    frac = min(0.25, max(b["corner_probe_frac"], min_inset_px / min(s1, s2)))
    c = np.array(rect[0])
    hits = 0
    for q in cv2.boxPoints(rect):
        x, y = c + (q - c) * (1.0 - 2.0 * frac)
        hits += bool(inside(int(round(x)), int(round(y))))
    # min_corner_hits = 3: one corner may be missing (a corner sticker with a weak edge); a disc misses all 4,
    # a triangle at least 2
    return rect, ("" if hits >= b["min_corner_hits"] else "corners")


def axis_deg90(rect):
    """Orientation of a rotated rect modulo 90 degrees (squares have no long axis), in [0, 90)."""
    return rect[2] % 90.0


def merge_fragments(comps, pix, b, ppcm, p_open_r=0):
    """A face can fall apart into pieces when some of its stickers have no edge against the background:
    "stripes" -> 2 strips, "corners" on a dark background -> 4 corner stickers or a cross + stickers.
    Group every complete-size-checked component (rect computed, not touching the edge) that is close to another
    (bbox gap <= merge_gap_cm) and aligned with it (orientations within merge_max_deg, modulo 90), and test the
    convex hull of each group of >= 2 as a square. The caller keeps the LARGEST valid square, so the whole face
    wins over one of its stickers.
    -> list of new candidate components (mask given by their hull)."""
    idx = [k for k, c in enumerate(comps) if c["rect"] is not None and not c["touches"]]
    gap = b["merge_gap_cm"] * ppcm
    parent = {k: k for k in idx}

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k

    for i_, i in enumerate(idx):
        for j in idx[i_ + 1:]:
            (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = comps[i]["bbox"], comps[j]["bbox"]
            g = max(max(ax1, bx1) - min(ax2, bx2), max(ay1, by1) - min(ay2, by2), 0)
            d = abs(axis_deg90(comps[i]["rect"]) - axis_deg90(comps[j]["rect"]))
            if g <= gap and min(d, 90.0 - d) <= b["merge_max_deg"]:
                parent[root(j)] = root(i)
    groups = {}
    for k in idx:
        groups.setdefault(root(k), []).append(k)
    out = []
    for members in groups.values():
        if len(members) < 2:
            continue
        hull = cv2.convexHull(np.vstack([pix[k] for k in members]).astype(np.int32))
        area = int(cv2.contourArea(hull)) + 1
        rect, why = square_test(hull.reshape(-1, 2), area, b, ppcm,
                                lambda x, y, h=hull: cv2.pointPolygonTest(h, (float(x), float(y)), False) >= 0,
                                min_inset_px=p_open_r + 1)
        xs, ys = hull[:, 0, 0], hull[:, 0, 1]
        out.append(dict(label=None, hull=hull, bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                        area=area, why=why, touches=False, rect=rect))
    return out


def segment(small_bgr, p):
    """-> (edges, mask, labels, components) at the downsampled resolution."""
    e = p["edges"]
    m, gx, gy = gradient(small_bgr, e["chroma_gain"])
    nms = nonmax(m, gx, gy)
    lo, hi = thresholds(nms, e)
    edges = hysteresis(nms, lo, hi, e["min_edge_px"])
    r, ro = p["mask"]["close_r"], p["mask"]["open_r"]
    mask = erode(fill_holes(dilate(edges, r)), r)
    if ro > 0:
        mask = dilate(erode(mask, ro), ro)

    b = p["blob"]
    f = p["image"]["downsample"]
    ppcm = p["crop"]["px_per_cm"] / f            # px per cm at the downsampled resolution
    H, W = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps, pix = [], []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        touches = x <= b["edge_margin_px"] or x + w >= W - b["edge_margin_px"]
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        pts = np.column_stack([xs + x, ys + y])
        rect, why = None, ""
        if area < b["min_area_px"]:
            why = "small"
            if area >= b["merge_min_px"] and not touches:   # can still be a piece of a face (sticker)
                rect = cv2.minAreaRect(pts.astype(np.float32))
        elif area > b["max_area_px"]:
            why = "big"
        elif not touches:                          # a cube entering from the side is not square yet
            rect, why = square_test(pts, area, b, ppcm,
                                    lambda x, y, i=i: 0 <= y < H and 0 <= x < W and lab[y, x] == i,
                                    min_inset_px=ro + 1)
        comps.append(dict(label=i, hull=None, bbox=(x, y, x + w, y + h), area=area, why=why, touches=touches,
                          rect=rect))
        pix.append(pts)
    if b.get("merge_gap_cm", 0) > 0:
        comps += merge_fragments(comps, pix, b, ppcm, ro)
    return edges, mask, lab, comps


def order_corners(pts):
    """4 points -> (tl, tr, br, bl): start at the min x+y corner, go clockwise on screen (y down)."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def rect_corners_full(rect, f, off, inset):
    """Rotated rect at the downsampled resolution -> 4 ordered corners in full-res px, shrunk by inset."""
    (cx, cy), (w, h), a = rect
    rect = ((cx, cy), (w + 1.0, h + 1.0), a)       # include the pixels' extent, not just their centres
    pts = cv2.boxPoints(rect).astype(np.float64)
    c = pts.mean(axis=0)
    pts = c + (pts - c) * (1.0 - 2.0 * inset)
    pts = (pts + 0.5) * f - 0.5 + np.array(off, np.float64)   # block centre -> full-res pixel coords
    return order_corners(pts).astype(np.float32)


def rectify(img, corners, n):
    """Affine warp: corners (tl, tr, br, bl) -> upright n x n image (bilinear, replicate border)."""
    src = corners[:3].astype(np.float32)
    dst = np.array([[-0.5, -0.5], [n - 0.5, -0.5], [n - 0.5, n - 0.5]], np.float32)
    A = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(img, A, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def process(img, p=None, keep_stages=False):
    """Full pipeline on one BGR image (frame or cut band). Returns a Detection."""
    p = p or load_params()
    f = p["image"]["downsample"]
    H, W = img.shape[:2]
    wx1, wy1, wx2, wy2 = p["image"]["window"] or (0, 0, W, H)
    wx1, wy1, wx2, wy2 = max(0, wx1), max(0, wy1), min(W, wx2), min(H, wy2)
    win = img[wy1:wy2, wx1:wx2]
    small = downsample(win, f)
    edges, mask, lab, comps = segment(small, p)

    to_full = lambda bb: (wx1 + bb[0] * f, wy1 + bb[1] * f, wx1 + bb[2] * f, wy1 + bb[3] * f)
    blobs = []
    for cp in comps:
        box = rect_corners_full(cp["rect"], f, (wx1, wy1), 0.0) if cp["rect"] is not None else None
        blobs.append(dict(bbox=to_full(cp["bbox"]), area=cp["area"] * f * f, box=box,
                          why=cp["why"] or ("edge" if cp["touches"] else "")))
    stages = dict(small=small, edges=edges, mask=mask) if keep_stages else {}

    valid = [cp for cp in comps if not cp["why"]]
    if not valid:
        return Detection("none", reason="no object" if not comps else "rejected", blobs=blobs, stages=stages)
    best = max(valid, key=lambda cp: (not cp["touches"], cp["area"]))   # prefer a complete square

    if best["hull"] is not None:                   # merged fragments: the hull is the object
        blob_small = np.zeros(mask.shape, np.uint8)
        cv2.fillConvexPoly(blob_small, best["hull"], 255)
    else:
        blob_small = (lab == best["label"]).astype(np.uint8) * 255
    full_mask = np.zeros((H, W), np.uint8)
    up = np.repeat(np.repeat(blob_small, f, axis=0), f, axis=1)
    full_mask[wy1:wy1 + up.shape[0], wx1:wx1 + up.shape[1]] = up
    det = Detection("partial" if best["touches"] else "ok", bbox=to_full(best["bbox"]), mask=full_mask,
                    area_px=best["area"] * f * f, blobs=blobs, stages=stages)
    if det.status == "ok":
        rect = best["rect"]
        ppcm = p["crop"]["px_per_cm"] / f
        det.size_cm = ((rect[1][0] + 1) / ppcm, (rect[1][1] + 1) / ppcm)
        det.angle = float(rect[2] % 90.0)
        det.corners = rect_corners_full(rect, f, (wx1, wy1), p["crop"]["inset_frac"])
        det.crop = rectify(img, det.corners, p["crop"]["input_size"])
    return det


# ── debug view ──────────────────────────────────────────────────────────────────────────────
def debug_view(img, det, p=None, width=960):
    """Stacked image: frame with boxes | edges | mask, plus the model input zoomed on the right."""
    p = p or load_params()
    view = img.copy()
    win = p["image"]["window"]
    if win:
        cv2.rectangle(view, tuple(win[:2]), (win[2] - 1, win[3] - 1), (200, 140, 20), 1)
    for b in det.blobs:
        x1, y1, x2, y2 = b["bbox"]
        if b["why"]:
            if b["box"] is not None:
                cv2.polylines(view, [b["box"].astype(np.int32)], True, (200, 0, 200), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), (200, 0, 200), 1)
            cv2.putText(view, b["why"], (x1, min(y2 + 12, view.shape[0] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 0, 200), 1, cv2.LINE_AA)
        elif b["box"] is not None:                 # valid square but not the chosen one
            cv2.polylines(view, [b["box"].astype(np.int32)], True, (60, 200, 60), 1, cv2.LINE_AA)
    col = {"ok": (60, 200, 60), "partial": (0, 200, 255)}.get(det.status, (0, 0, 255))
    if det.corners is not None:
        cv2.polylines(view, [det.corners.astype(np.int32)], True, col, 2, cv2.LINE_AA)
        cv2.circle(view, tuple(int(v) for v in det.corners[0]), 3, (0, 0, 255), -1)   # tl = crop's top-left
    elif det.bbox:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(view, (x1, y1), (x2 - 1, y2 - 1), col, 2)
    txt = det.status if det.status != "none" else f"none ({det.reason})"
    if det.size_cm:
        txt += f"  {det.size_cm[0]:.1f}x{det.size_cm[1]:.1f} cm  {det.angle:.0f} deg"
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
    sz = min(side, max(img.shape[0], 256))
    right = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_NEAREST)
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
