"""
classifier/npmodel.py - the trained CNN (train.build_model) re-implemented in plain NumPy, so the twin app can classify
without TensorFlow (its conda env "conveyor-twin" and the Orange Pi do not have it). ~2-5 ms per face.

    m = NpModel("classifier/models/cube_classifier.npz")
    probs = m.predict(face_rgb)          # face: 64x64x3 uint8 RGB  ->  (n_classes,) probabilities

The .npz is written by  python -m classifier.export  (batch-norm folded into the convolutions);
classifier/tests/test_npmodel.py checks it against Keras on the dataset.
"""
import json

import numpy as np


def to_npz(model, path, classes):
    """Keras model (train.build_model) -> npz with folded conv weights. Runs where Keras is installed."""
    arrays, convs = {}, 0
    layers = model.layers
    for i, layer in enumerate(layers):
        kind = layer.__class__.__name__
        if kind == "Conv2D":
            W = layer.get_weights()[0].astype(np.float64)                     # kh, kw, cin, cout
            b = layer.get_weights()[1] if layer.use_bias else np.zeros(W.shape[-1])
            bn = layers[i + 1]
            assert bn.__class__.__name__ == "BatchNormalization", "expected Conv2D -> BatchNormalization"
            gamma, beta, mean, var = [w.astype(np.float64) for w in bn.get_weights()]
            s = gamma / np.sqrt(var + bn.epsilon)
            arrays[f"conv{convs}_w"] = (W * s).astype(np.float32)
            arrays[f"conv{convs}_b"] = ((b - mean) * s + beta).astype(np.float32)
            convs += 1
        elif kind == "Dense":
            arrays["dense_w"], arrays["dense_b"] = [w.astype(np.float32) for w in layer.get_weights()]
    arrays["meta"] = np.array(json.dumps(dict(classes=classes, n_conv=convs, input=list(model.input_shape[1:]))))
    np.savez_compressed(path, **arrays)


class NpModel:
    def __init__(self, path):
        d = np.load(path)
        self.meta = json.loads(str(d["meta"]))
        self.classes = self.meta["classes"]
        self.convs = [(d[f"conv{k}_w"], d[f"conv{k}_b"]) for k in range(self.meta["n_conv"])]
        self.dense = (d["dense_w"], d["dense_b"])

    @staticmethod
    def _conv3x3(x, W, b):
        """'same' 3x3 convolution, stride 1. x: (h, w, cin) -> (h, w, cout)."""
        p = np.pad(x, ((1, 1), (1, 1), (0, 0)))
        win = np.lib.stride_tricks.sliding_window_view(p, (3, 3), axis=(0, 1))   # h, w, cin, 3, 3
        return np.einsum("hwcij,ijco->hwo", win, W, optimize=True) + b

    @staticmethod
    def _maxpool2(x):
        h, w, c = x.shape
        return x[:h // 2 * 2, :w // 2 * 2].reshape(h // 2, 2, w // 2, 2, c).max(axis=(1, 3))

    def predict(self, face_rgb):
        x = face_rgb.astype(np.float32) / 255.0
        for W, b in self.convs:
            x = self._maxpool2(np.maximum(self._conv3x3(x, W, b), 0.0))
        x = x.mean(axis=(0, 1))
        z = x @ self.dense[0] + self.dense[1]
        e = np.exp(z - z.max())
        return e / e.sum()
