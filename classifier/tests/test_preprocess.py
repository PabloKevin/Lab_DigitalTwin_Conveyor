"""Synthetic checks of classifier/preprocess.py.   Run:  python -m unittest classifier.tests.test_preprocess -v"""
import unittest

import cv2
import numpy as np

from classifier import preprocess as pp

H, W = 160, 640


def belt(seed=0, seam=True, glare=True):
    """Non-homogeneous belt: grey base + fine noise + soft low-frequency texture + seam + soft glare spots."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), (70, 75, 80), np.float32)
    low = cv2.resize(rng.normal(0, 10, (8, 32)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img += low[..., None]
    img += rng.normal(0, 6, (H, W, 3))
    if seam:
        img[:, 430:450] += 60
    if glare:
        yy, xx = np.mgrid[0:H, 0:W]
        for cx, cy, s in ((200, 40, 18), (520, 110, 25)):
            img += (70 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


def cube(img, cx, cy, side, angle, n=3, colours=None, gap=(25, 25, 25)):
    """Draw a rotated n x n cube face (black plastic gaps between stickers). colours[r][c] = BGR.
    Returns the ground-truth mask."""
    colours = colours or [[(30, 120, 230)] * n for _ in range(n)]
    face = np.zeros((side, side, 3), np.uint8)
    face[:] = gap
    cell = side / n
    for r in range(n):
        for c in range(n):
            x0, y0 = int(c * cell + 3), int(r * cell + 3)
            x1, y1 = int((c + 1) * cell - 3), int((r + 1) * cell - 3)
            face[y0:y1, x0:x1] = colours[r][c]
    M = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    M[:, 2] += (cx - side / 2, cy - side / 2)
    warped = cv2.warpAffine(face, M, (img.shape[1], img.shape[0]))
    gt = cv2.warpAffine(np.full((side, side), 255, np.uint8), M, (img.shape[1], img.shape[0]))
    img[gt > 127] = warped[gt > 127]
    return gt


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max(1, (a | b).sum())


def precision(mask, gt):
    """Fraction of the mask that is really the object (= no background leaking into the crop)."""
    m = mask > 0
    return (m & (gt > 0)).sum() / max(1, m.sum())


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.p = pp.load_params()
        self.p["image"]["window"] = None
        # synthetic scene: moderate thresholds (the real params are tuned on the camera)
        self.p["edges"].update(chroma_gain=1.5, lo=40, hi=100, min_edge_px=12)
        self.p["mask"].update(close_r=2, open_r=2)
        self.p["blob"].update(min_area_px=120, side_cm=[3.0, 10.0])
        self.p["crop"]["px_per_cm"] = 10.0

    def test_empty_belt_no_detection(self):
        for seed in range(5):
            det = pp.process(belt(seed), self.p)
            self.assertEqual(det.status, "none", f"seed {seed}: false trigger {det.bbox}")

    def test_rotated_cube_found_and_rectified(self):
        R, B = (40, 40, 220), (220, 90, 30)
        img = belt(1)
        gt = cube(img, 300, 80, 60, 25, n=3, colours=[[R, R, R], [B, B, B], [R, R, R]])   # "stripes"
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertGreaterEqual(precision(det.mask, gt), 0.97)
        self.assertGreaterEqual(iou(det.mask, gt), 0.75)
        self.assertEqual(det.crop.shape, (64, 64, 3))
        self.assertAlmostEqual(det.size_cm[0], 6.0, delta=0.6)
        # upright face: middle row blue, top/bottom rows red (or the same rotated 90 degrees -> columns)
        mid = det.crop[24:40, 24:40].reshape(-1, 3).mean(0)
        top = det.crop[4:16, 24:40].reshape(-1, 3).mean(0)
        side = det.crop[24:40, 4:16].reshape(-1, 3).mean(0)
        self.assertGreater(mid[0], mid[2])                          # blue in the middle
        self.assertTrue(top[2] > top[0] or side[2] > side[0])        # red at one pair of edges

    def test_low_luminance_contrast_colour_cube(self):
        img = belt(2)
        gt = cube(img, 290, 80, 64, 10, colours=[[(120, 60, 40)] * 3 for _ in range(3)])
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertGreaterEqual(precision(det.mask, gt), 0.97)
        self.assertGreaterEqual(iou(det.mask, gt), 0.75)

    def test_rectangle_and_disc_rejected(self):
        img = belt(3)
        cv2.rectangle(img, (100, 50), (200, 95), (40, 40, 220), -1)      # 2.2:1 strip
        cv2.circle(img, (330, 80), 30, (30, 200, 230), -1)              # disc: fills pi/4 of its square
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "none")
        whys = sorted(b["why"] for b in det.blobs if b["why"])
        self.assertIn("rect", whys)
        self.assertIn("corners", whys)                               # disc passes the fill test, not the probes

    def test_rotated_45_cube_found(self):
        """45 degrees: the mask corners get rounded by the pixel grid - must still pass. Working range: side >= ~25 px
        at the downsampled resolution (a 5.7 cm cube on the VGA band at 10 px/cm is ~28 px); below that the 5x5
        opening destroys a 45-degree diamond."""
        for side in (52, 60, 80):
            img = belt(8)
            cube(img, 300, 80, side, 45)
            det = pp.process(img, self.p)
            self.assertEqual(det.status, "ok", f"side {side}: {[b['why'] for b in det.blobs]}")

    def test_cube_touching_a_line_is_separated(self):
        """A thin edge line glued to the cube (desk edge) must not merge with it (open_r)."""
        img = belt(4, seam=False)
        cube(img, 300, 85, 60, 0)
        cv2.line(img, (0, 52), (639, 56), (200, 200, 200), 1)
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertLess(det.bbox[2] - det.bbox[0], 80)

    def test_face_split_in_strips_is_rejoined(self):
        """Like real frame 00061: the middle stripe and the plastic frame barely differ from the background, so
        the edges only outline the two outer strips -> 2 strips (rejected as rect); fragment merging must
        re-join them into one square."""
        img = belt(6, seam=False, glare=False)
        bg = tuple(int(v) for v in img[80, 300])
        O = (60, 110, 240)
        cube(img, 300, 80, 60, 0, colours=[[O, bg, O]] * 3, gap=bg)
        q = {**self.p, "blob": {**self.p["blob"], "merge_gap_cm": 0}}
        self.assertEqual(pp.process(img, q).status, "none")          # without merging: lost
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertAlmostEqual(det.size_cm[0], 5.6, delta=0.8)

    def test_corners_face_on_dark_background(self):
        """'corners' face whose centre cross is the same colour as the background (blue cube on dark cloth):
        only the 4 corner stickers give edges. They must be merged into the whole face, and the result must be the
        face, not one sticker."""
        img = belt(9, seam=False, glare=False)
        bg = tuple(int(v) for v in img[80, 300])
        Y = (80, 220, 230)
        cube(img, 300, 80, 66, 15, colours=[[Y, bg, Y], [bg, bg, bg], [Y, bg, Y]], gap=bg)
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertGreater(det.size_cm[0], 5.0)                     # the face (~6 cm), not a sticker (~2 cm)

    def test_far_strips_not_merged(self):
        """Two parallel strips far apart (> merge_gap_cm) must not become a fake square."""
        img = belt(7, seam=False, glare=False)
        for x in (150, 330):
            cv2.rectangle(img, (x, 50), (x + 18, 110), (60, 110, 240), -1)
        self.assertEqual(pp.process(img, self.p).status, "none")

    def test_cube_at_edge_is_partial(self):
        img = belt(5)
        cube(img, 10, 80, 60, 0)                                       # half outside on the left
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "partial")
        self.assertIsNone(det.crop)

    def test_integer_primitives(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4, 1).repeat(3, axis=2)
        self.assertEqual(pp.downsample(img, 2)[0, 0, 0], 3)        # (0+1+4+5+2)//4
        flat = np.full((10, 10), 77, np.int32)
        self.assertTrue((pp.blur5(flat) == 77).all())


if __name__ == "__main__":
    unittest.main()
