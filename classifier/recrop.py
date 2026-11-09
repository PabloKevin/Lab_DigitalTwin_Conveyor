"""
classifier/recrop.py - re-run preprocess.py on every saved raw frame (<id>_band.png) of the dataset, e.g. after
params.json changed. Rewrites <id>_mask.png / <id>_crop.png and the manifest (backup kept as manifest.jsonl.bak).
Optionally renames classes (moves the folders and fixes the manifest).

    python -m classifier.recrop dataset                                  # all sessions
    python -m classifier.recrop dataset/00 --rename class_a=solid class_b=stripes
"""
import argparse
import glob
import json
import os
import shutil

import cv2

from classifier import preprocess as pp


def sessions(root):
    if os.path.exists(os.path.join(root, "manifest.jsonl")):
        return [root]
    return sorted(os.path.dirname(m) for m in glob.glob(os.path.join(root, "*", "manifest.jsonl")))


def recrop_session(sdir, p, rename):
    man = os.path.join(sdir, "manifest.jsonl")
    shutil.copyfile(man, man + ".bak")
    for old, new in rename.items():
        src, dst = os.path.join(sdir, old), os.path.join(sdir, new)
        if os.path.isdir(src):
            if os.path.exists(dst):
                raise SystemExit(f"{dst} already exists - merge by hand")
            os.rename(src, dst)
    recs, stats = [], {}
    for line in open(man):
        r = json.loads(line)
        r["cls"] = rename.get(r["cls"], r["cls"])
        d = os.path.join(sdir, r["cls"])
        band_path = os.path.join(d, f"{r['id']}_band.png")
        band = cv2.imread(band_path)
        if band is None:
            print(f"  missing {band_path}, dropped from manifest")
            continue
        if r["cls"] != "_background":
            det = pp.process(band, p)
            for kind, img in (("mask", det.mask), ("crop", det.crop)):
                path = os.path.join(d, f"{r['id']}_{kind}.png")
                if img is not None:
                    cv2.imwrite(path, img)
                elif os.path.exists(path):
                    os.remove(path)                          # stale output of older params
            r.update(status=det.status, bbox=list(det.bbox) if det.bbox else None, area_px=det.area_px,
                     size_cm=list(det.size_cm) if det.size_cm else None, angle=det.angle,
                     corners=det.corners.tolist() if det.corners is not None else None, params=p)
            r.pop("scale_k", None)
            stats[(r["cls"], det.status)] = stats.get((r["cls"], det.status), 0) + 1
        recs.append(r)
    with open(man, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root or one session folder")
    ap.add_argument("--rename", nargs="*", default=[], metavar="OLD=NEW")
    ap.add_argument("--params", default=pp.PARAMS_PATH)
    a = ap.parse_args()
    p = pp.load_params(a.params)
    rename = dict(x.split("=", 1) for x in a.rename)
    for s in sessions(a.root):
        st = recrop_session(s, p, rename)
        print(s, " ".join(f"{c}:{k}={n}" for (c, k), n in sorted(st.items())))


if __name__ == "__main__":
    main()
