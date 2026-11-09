"""
classifier/augment.py - data augmentation in two stages.

1) FRAME level (before detection) - simulates camera height and object pose, and tests the detector at the same time:
     scale x[scale_lo, scale_hi] (camera closer / farther), rotation 0-360 deg, small shift, all about the object;
     the frame is then run through preprocess.process() again -> new upright face. If the detector loses the cube
     the sample is dropped (counted: it is a robustness number for the report).
2) FACE level (on the 64x64 upright face):
     rot90 / flip (the classes do not care about the face orientation),
     colour permutation: the classes only depend on WHICH stickers share a colour, never on the colour itself, so
       swapping colours keeps the label (BGR channel shuffle + hue rotation),
     brightness / contrast / gamma, mild saturation change (kept >= 0.85: yellow must not become white),
     small corner jitter (imperfect detection), motion blur (belt), noise, JPEG artefacts (camera stream).

    python -m classifier.augment dataset --preview /tmp/aug.jpg        # visual check
    (classifier/train.py calls make_dataset())
"""
import glob
import json
import os

import cv2
import numpy as np

from classifier import preprocess as pp

CFG = dict(scale=(0.7, 1.3), shift_px=6, sat=(0.85, 1.2), hue_deg=12, bright=(-25, 25), contrast=(0.75, 1.25),
           gamma=(0.8, 1.25), jitter_frac=0.04, p_permute=0.6, p_blur=0.3, blur_len=(3, 7), noise_std=(0, 6),
           p_jpeg=0.5, jpeg_q=(35, 90))


# ── frame level ─────────────────────────────────────────────────────────────────────────────
def augment_frame(frame, center, rng, cfg=CFG):
    """Scale + rotate + shift the whole frame about the object centre (reflected border keeps the texture)."""
    s = rng.uniform(*cfg["scale"])
    a = rng.uniform(0, 360)
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), a, s)
    M[:, 2] += rng.uniform(-cfg["shift_px"], cfg["shift_px"], 2)
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


# ── face level ──────────────────────────────────────────────────────────────────────────────
def permute_colours(face, rng, cfg=CFG):
    face = face[..., rng.permutation(3)]
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + int(rng.integers(0, 180))) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def photometric(face, rng, cfg=CFG):
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-cfg["hue_deg"], cfg["hue_deg"]) / 2) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(*cfg["sat"]), 0, 255)
    face = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    face = (face - 128) * rng.uniform(*cfg["contrast"]) + 128 + rng.uniform(*cfg["bright"])
    face = 255 * (np.clip(face, 0, 255) / 255) ** rng.uniform(*cfg["gamma"])
    return np.clip(face, 0, 255).astype(np.uint8)


def corner_jitter(face, rng, cfg=CFG):
    n = face.shape[0]
    src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
    dst = src + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * cfg["jitter_frac"] * n
    return cv2.warpPerspective(face, cv2.getPerspectiveTransform(src, dst), (n, n), borderMode=cv2.BORDER_REPLICATE)


def degrade(face, rng, cfg=CFG):
    if rng.random() < cfg["p_blur"]:
        k = int(rng.integers(cfg["blur_len"][0], cfg["blur_len"][1] + 1))
        ker = np.zeros((k, k), np.float32)
        ker[k // 2, :] = 1.0 / k
        R = cv2.getRotationMatrix2D(((k - 1) / 2, (k - 1) / 2), rng.uniform(0, 180), 1.0)
        ker = cv2.warpAffine(ker, R, (k, k))
        face = cv2.filter2D(face, -1, ker / max(ker.sum(), 1e-6))
    std = rng.uniform(*cfg["noise_std"])
    if std > 0:
        face = np.clip(face + rng.normal(0, std, face.shape), 0, 255).astype(np.uint8)
    if rng.random() < cfg["p_jpeg"]:
        q = int(rng.integers(*cfg["jpeg_q"]))
        face = cv2.imdecode(cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return face


def augment_face(face, rng, cfg=CFG):
    face = np.rot90(face, int(rng.integers(4)))
    if rng.random() < 0.5:
        face = face[:, ::-1]
    face = np.ascontiguousarray(face)
    if rng.random() < cfg["p_permute"]:
        face = permute_colours(face, rng, cfg)
    face = corner_jitter(face, rng, cfg)
    face = photometric(face, rng, cfg)
    return degrade(face, rng, cfg)


# ── dataset ─────────────────────────────────────────────────────────────────────────────────
def records(root):
    """All labelled shots: dict(session, cls, band_path, center)."""
    out = []
    for man in sorted(glob.glob(os.path.join(root, "**", "manifest.jsonl"), recursive=True)):
        sdir = os.path.dirname(man)
        for line in open(man):
            r = json.loads(line)
            if r["cls"].startswith("_") or r.get("status") != "ok":
                continue
            x1, y1, x2, y2 = r["bbox"]
            out.append(dict(session=os.path.relpath(sdir, root), cls=r["cls"], id=r["id"],
                            band_path=os.path.join(sdir, r["cls"], f"{r['id']}_band.png"),
                            center=((x1 + x2) / 2, (y1 + y2) / 2)))
    return out


def samples_for(rec, n_aug, rng, p, cfg=CFG):
    """-> (faces list, n_frame_attempts, n_frame_lost). First face = the un-augmented one."""
    band = cv2.imread(rec["band_path"])
    det = pp.process(band, p)
    faces, lost = ([det.crop] if det.crop is not None else []), 0
    for _ in range(n_aug):
        d = pp.process(augment_frame(band, rec["center"], rng, cfg), p)
        if d.crop is None:
            lost += 1
            continue
        faces.append(augment_face(d.crop, rng, cfg))
    return faces, n_aug, lost


def make_dataset(root, classes, n_aug=10, seed=0, p=None, cfg=CFG):
    """-> X (N, n, n, 3) uint8 RGB, y (N,) int, groups (N,) session names, original (N,) bool, stats dict."""
    p = p or pp.load_params()
    rng = np.random.default_rng(seed)
    X, y, g, orig = [], [], [], []
    tried = lost = 0
    for rec in records(root):
        if rec["cls"] not in classes:
            continue
        faces, t, l = samples_for(rec, n_aug, rng, p, cfg)
        tried, lost = tried + t, lost + l
        for k, f in enumerate(faces):
            X.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            y.append(classes.index(rec["cls"]))
            g.append(rec["session"])
            orig.append(k == 0)
    X = np.stack(X) if X else np.zeros((0, p["crop"]["input_size"], p["crop"]["input_size"], 3), np.uint8)
    return X, np.array(y), np.array(g), np.array(orig), dict(frame_aug_tried=tried, frame_aug_lost=lost)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--preview", required=True, help="output image: one row per shot, original + augmentations")
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--n", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = pp.load_params()
    rng = np.random.default_rng(a.seed)
    recs = records(a.root)
    pick = [recs[i] for i in rng.choice(len(recs), min(a.rows, len(recs)), replace=False)]
    rows, tried, lost = [], 0, 0
    for r in pick:
        faces, t, l = samples_for(r, a.n, rng, p)
        tried, lost = tried + t, lost + l
        tiles = [cv2.resize(f, (96, 96), interpolation=cv2.INTER_NEAREST) for f in faces]
        tiles += [np.zeros((96, 96, 3), np.uint8)] * (a.n + 1 - len(tiles))
        cv2.putText(tiles[0], r["cls"], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.hstack(tiles))
    cv2.imwrite(a.preview, np.vstack(rows))
    print(f"{a.preview}: {len(rows)} shots; frame-level augmentations lost by the detector: {lost}/{tried}")


if __name__ == "__main__":
    main()
