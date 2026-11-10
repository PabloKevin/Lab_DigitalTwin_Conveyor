"""C++ facefind.cpp (ESP32 firmware) vs Python preprocess.py on the real dataset frames + synthetic scenes.
Needs g++.   python -m unittest classifier.tests.test_facefind_parity -v"""
import ctypes
import glob
import os
import tempfile
import unittest

import cv2
import numpy as np

from classifier import fwgen
from classifier import preprocess as pp
from classifier.tests.test_preprocess import belt, cube

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATUS = {0: "none", 1: "partial", 2: "ok"}


def run_cpp(lib, img, p):
    img = np.ascontiguousarray(img)
    res = fwgen.FFResult()
    n = p["crop"]["input_size"]
    crop = np.zeros((n, n, 3), np.uint8)
    lib.ff_process(img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], ctypes.byref(fwgen.ff_params(p)),
                   ctypes.byref(res), crop.ctypes.data)
    return res, crop


class TestParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.lib = fwgen.build_pc_lib(cls.tmp)
        cls.p = pp.load_params()

    def compare(self, img, p, name):
        det = pp.process(img, p)
        res, crop = run_cpp(self.lib, img, p)
        self.assertEqual(STATUS[res.status], det.status, f"{name}: status")
        if det.status != "none":
            self.assertEqual(list(res.bbox), list(det.bbox), f"{name}: bbox")
        if det.status == "ok":
            cc = np.array(res.corners).reshape(4, 2)
            self.assertLess(np.abs(cc - det.corners).max(), 0.6, f"{name}: corners {cc} vs {det.corners}")
            diff = np.abs(crop.astype(int) - det.crop.astype(int))
            self.assertLess(diff.mean(), 2.0, f"{name}: crop mean abs diff {diff.mean():.2f}")
            self.assertAlmostEqual(sorted(res.size_cm)[0], sorted(det.size_cm)[0], delta=0.05)
        return det.status

    def test_dataset_frames(self):
        files = sorted(glob.glob(os.path.join(ROOT, "dataset", "*", "*", "*_band.png")))
        if not files:
            self.skipTest("no dataset")
        seen = {}
        for f in files:
            st = self.compare(cv2.imread(f), self.p, os.path.relpath(f, ROOT))
            seen[st] = seen.get(st, 0) + 1
        print(f"\n  parity on {len(files)} dataset frames: {seen}")

    def test_synthetic(self):
        q = dict(self.p)
        q["edges"] = {**self.p["edges"], "chroma_gain": 1.5, "lo": 40, "hi": 100, "min_edge_px": 12}
        q["mask"] = {**self.p["mask"], "close_r": 2, "open_r": 2}
        q["blob"] = {**self.p["blob"], "min_area_px": 120}
        for seed, ang in ((1, 0), (2, 25), (3, 45), (4, 70)):
            img = belt(seed)
            cube(img, 300, 80, 60, ang)
            self.compare(img, q, f"synthetic cube {ang} deg")
        img = belt(5)
        cube(img, 10, 80, 60, 0)
        self.assertEqual(self.compare(img, q, "edge cube"), "partial")
        self.assertEqual(self.compare(belt(6), q, "empty belt"), "none")


if __name__ == "__main__":
    unittest.main()
