"""NumPy CNN (classifier/npmodel.py, used by the twin app) vs Keras.   python -m unittest classifier.tests.test_npmodel"""
import glob
import os
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNpModel(unittest.TestCase):
    def test_matches_keras(self):
        try:
            import keras
        except ImportError:
            self.skipTest("needs Keras (conda env Control)")
        from classifier import npmodel, preprocess as pp
        from classifier.train import load
        runs = sorted(glob.glob(os.path.join(ROOT, "classifier", "runs", "*", "model.keras")))
        if not runs:
            self.skipTest("no trained model")
        p = pp.load_params()
        X, y, _, _ = load(os.path.join(ROOT, "dataset"), ["10"], p["model"]["classes"])
        for path in runs[:3]:
            km = keras.models.load_model(path)
            npz = os.path.join(tempfile.mkdtemp(), "m.npz")
            npmodel.to_npz(km, npz, p["model"]["classes"])
            nm = npmodel.NpModel(npz)
            pk = km.predict(X, verbose=0)
            pn = np.stack([nm.predict(x) for x in X])
            self.assertLess(np.abs(pk - pn).max(), 1e-3, path)          # float32 / GPU (TF32) rounding
            self.assertTrue((pk.argmax(1) == pn.argmax(1)).all(), path)


if __name__ == "__main__":
    unittest.main()
