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


def iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max(1, (a | b).sum())


class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.p = pp.load_params()
        self.p["image"]["window"] = None

    def _object(self, img, draw):
        gt = np.zeros((H, W), np.uint8)
        draw(gt, 255)
        col = np.zeros_like(img)
        draw(col, None)
        img[gt > 0] = col[gt > 0]
        return gt

    def test_empty_belt_no_detection(self):
        for seed in range(5):
            det = pp.process(belt(seed), self.p)
            self.assertEqual(det.status, "none", f"seed {seed}: false trigger {det.bbox}")

    def test_orange_disc(self):
        img = belt(1)
        def draw(t, v):
            cv2.circle(t, (300, 80), 28, v if v is not None else (30, 120, 230), -1)
        gt = self._object(img, draw)
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertGreaterEqual(iou(det.mask, gt), 0.9)
        self.assertEqual(det.crop.shape, (64, 64, 3))

    def test_low_luminance_contrast_colour_object(self):
        """Dark blue box on the dark belt: little Y contrast, found through the chroma channels."""
        img = belt(2)
        def draw(t, v):
            cv2.rectangle(t, (250, 50), (330, 115), v if v is not None else (120, 60, 40), -1)
        gt = self._object(img, draw)
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertGreaterEqual(iou(det.mask, gt), 0.9)

    def test_textured_object_with_hole_is_filled(self):
        img = belt(3)
        def draw(t, v):
            cv2.ellipse(t, (150, 80), (45, 30), 20, 0, 360, v if v is not None else (200, 200, 210), -1)
        gt = self._object(img, draw)
        cv2.circle(img, (150, 80), 8, (60, 60, 60), -1)          # dark print in the middle of the object
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "ok")
        self.assertGreaterEqual(iou(det.mask, gt), 0.9)

    def test_object_at_edge_is_partial(self):
        img = belt(4)
        def draw(t, v):
            cv2.rectangle(t, (0, 40), (40, 110), v if v is not None else (40, 200, 40), -1)
        self._object(img, draw)
        det = pp.process(img, self.p)
        self.assertEqual(det.status, "partial")
        self.assertIsNone(det.crop)

    def test_fixed_scale(self):
        """Same object, two sizes -> the crop keeps the size difference (no stretching)."""
        areas = []
        for r in (15, 30):
            img = belt(5)
            def draw(t, v, r=r):
                cv2.circle(t, (300, 80), r, v if v is not None else (30, 120, 230), -1)
            self._object(img, draw)
            det = pp.process(img, self.p)
            self.assertEqual(det.status, "ok")
            fill = np.array(self.p["crop"]["fill_bgr"], np.uint8)
            areas.append(int((np.abs(det.crop.astype(int) - fill).sum(axis=2) > 30).sum()))
        self.assertGreater(areas[1] / areas[0], 3.0)                # radius x2 -> area ~x4

    def test_integer_primitives(self):
        img = np.arange(16, dtype=np.uint8).reshape(4, 4, 1).repeat(3, axis=2)
        self.assertEqual(pp.downsample(img, 2)[0, 0, 0], 3)        # (0+1+4+5+2)//4
        flat = np.full((10, 10), 77, np.int32)
        self.assertTrue((pp.blur5(flat) == 77).all())


if __name__ == "__main__":
    unittest.main()
