"""
classifier/train.py - train the cube-face classifier (small CNN, Keras) on the crops of the dataset.

Data:   dataset/<session>/<class>/<id>_crop.png  (upright 64x64 faces made by preprocess.py), classes from params.json.
Split:  BLOCKED cross-validation - each class's shots, in capture order, are cut into K contiguous blocks; fold k
        tests on block k of every class. Consecutive shots are often the same cube in a slightly different pose,
        so a random split would put near-duplicates in train and test and report a score that is too good.
        After CV, the final model is trained on ALL the data (CV = the honest estimate of how good it is).
Augmentation (on the fly, training only):
        rotation 0 / 90 / 180 / 270 degrees (the classes do not depend on the face orientation)
        light: brightness, contrast, gamma, per-channel gain (white balance), mild saturation and hue
        --colour-perm P: with probability P, swap the RGB channels and rotate the hue at random. The classes only
        depend on WHICH stickers share a colour, never on the colour itself, so this keeps the label and stops
        the model from memorising colour combinations (with ~25 photos per class it cannot see them all).
Output: classifier/runs/<name>/  model.keras, report.json (per-fold + overall metrics), confusion.png,
        errors.png (every misclassified CV sample with true/pred), aug_preview.png

    python -m classifier.train --sessions 10
    python -m classifier.train --sessions 10 01 --epochs 80 --folds 5
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from classifier import preprocess as pp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIGHT = dict(brightness=0.12, contrast=(0.75, 1.3), gamma=(0.75, 1.3), gain=(0.9, 1.1), saturation=(0.85, 1.2),
             hue=0.03)


# ── data ────────────────────────────────────────────────────────────────────────────────────
def load(dataset, sessions, classes, recrop=False, p=None):
    """-> X (N,n,n,3) uint8 RGB, y (N,), order (N,) capture order within its class, ids list."""
    X, y, t, ids = [], [], [], []
    for s in sessions:
        man = os.path.join(dataset, s, "manifest.jsonl")
        for line in open(man):
            r = json.loads(line)
            if r["cls"] not in classes or r.get("status") != "ok":
                continue
            d = os.path.join(dataset, s, r["cls"])
            if recrop:
                det = pp.process(cv2.imread(os.path.join(d, f"{r['id']}_band.png")), p)
                img = det.crop
            else:
                img = cv2.imread(os.path.join(d, f"{r['id']}_crop.png"))
            if img is None:
                continue
            X.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            y.append(classes.index(r["cls"]))
            t.append(r["t"])
            ids.append(f"{s}/{r['cls']}/{r['id']}")
    return np.stack(X), np.array(y), np.array(t), ids


def blocked_folds(y, t, k):
    """fold index per sample: each class sorted by time, cut into k contiguous blocks."""
    fold = np.zeros(len(y), int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = idx[np.argsort(t[idx])]
        for f, chunk in enumerate(np.array_split(idx, k)):
            fold[chunk] = f
    return fold


# ── augmentation (tf ops, runs inside tf.data) ──────────────────────────────────────────────
PERMS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]


def colour_permute(x):
    import tensorflow as tf
    perm = tf.gather(tf.constant(PERMS), tf.random.uniform([], 0, 6, dtype=tf.int32))
    x = tf.gather(x, perm, axis=-1)
    return tf.image.adjust_hue(x, tf.random.uniform([], -0.5, 0.5))


def augment(img, colour_perm=0.0):
    import tensorflow as tf
    x = tf.image.convert_image_dtype(img, tf.float32)
    x = tf.image.rot90(x, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    if colour_perm > 0:
        x = tf.cond(tf.random.uniform([]) < colour_perm, lambda: colour_permute(x), lambda: x)
    x = tf.image.random_saturation(x, *LIGHT["saturation"])
    x = tf.image.random_hue(x, LIGHT["hue"])
    x = tf.image.random_contrast(x, *LIGHT["contrast"])
    x = tf.image.random_brightness(x, LIGHT["brightness"])
    x = x * tf.random.uniform([1, 1, 3], *LIGHT["gain"])
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = tf.pow(x, tf.random.uniform([], *LIGHT["gamma"]))
    return tf.cast(tf.round(x * 255.0), tf.uint8)


def dataset(X, y, batch, train, colour_perm=0.0):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if train:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True).repeat(8)   # 8 augmented copies per epoch
        ds = ds.map(lambda a, b: (augment(a, colour_perm), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ── model ───────────────────────────────────────────────────────────────────────────────────
def build_model(n, n_classes, width=16):
    """Small CNN for the ESP32 (int8 later): 4 conv blocks, global average pooling, dense. Input uint8 0..255."""
    import keras
    from keras import layers
    inp = keras.Input((n, n, 3), dtype="uint8")
    x = layers.Rescaling(1.0 / 255)(inp)
    for i, f in enumerate((width, width * 2, width * 4, width * 4)):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def class_weights(y, n):
    cnt = np.bincount(y, minlength=n).astype(float)
    return {i: len(y) / (n * max(c, 1)) for i, c in enumerate(cnt)}


def fit(Xtr, ytr, n_classes, epochs, batch, seed, colour_perm=0.0, width=16):
    import keras
    keras.utils.set_random_seed(seed)
    m = build_model(Xtr.shape[1], n_classes, width)
    m.fit(dataset(Xtr, ytr, batch, True, colour_perm), epochs=epochs, class_weight=class_weights(ytr, n_classes), verbose=0)
    return m


# ── reports ─────────────────────────────────────────────────────────────────────────────────
def metrics(y, pred, classes):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", labels=range(len(classes)), zero_division=0)),
                confusion=confusion_matrix(y, pred, labels=range(len(classes))).tolist())


def save_confusion(cm, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=30)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sheet(images, labels, path, size=96, cols=10):
    if not images:
        return
    tiles = []
    for img, lab in zip(images, labels):
        t = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (size, size), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, lab, (2, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, lab, (2, size - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    cv2.imwrite(path, np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(ROOT, "dataset"))
    ap.add_argument("--sessions", nargs="+", default=["10"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=16,
                    help="filters of the first conv (x1, x2, x4, x4); 8 = ~4x fewer operations on the ESP32")
    ap.add_argument("--colour-perm", type=float, default=0.8,
                    help="probability of a random colour permutation (0 = only rotation + light)")
    ap.add_argument("--recrop", action="store_true", help="re-run preprocess.py on the bands instead of saved crops")
    ap.add_argument("--no-cv", action="store_true", help="only train the final model")
    ap.add_argument("--out", default=os.path.join(ROOT, "classifier", "runs", time.strftime("%Y%m%d_%H%M%S")))
    a = ap.parse_args()

    p = pp.load_params()
    classes = p["model"]["classes"]
    X, y, t, ids = load(a.dataset, a.sessions, classes, a.recrop, p)
    os.makedirs(a.out, exist_ok=True)
    counts = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    print(f"{len(X)} faces {X.shape[1:]} from sessions {a.sessions}: {counts}")

    import tensorflow as tf
    print("GPU:", [d.name for d in tf.config.list_physical_devices("GPU")] or "none (CPU)")
    sample = np.stack([augment(X[i], a.colour_perm).numpy() for i in np.random.default_rng(0).choice(len(X), 20)])
    sheet(list(X[:10]) + list(sample[:10]), ["orig"] * 10 + ["aug"] * 10, os.path.join(a.out, "aug_preview.png"))

    report = dict(sessions=a.sessions, classes=classes, counts=counts, epochs=a.epochs, folds=a.folds,
                  light_aug=LIGHT, rotations=[0, 90, 180, 270], colour_perm=a.colour_perm, width=a.width)
    if not a.no_cv:
        fold = blocked_folds(y, t, a.folds)
        pred = np.full(len(y), -1)
        prob = np.zeros((len(y), len(classes)))
        per_fold = []
        for k in range(a.folds):
            te = fold == k
            t0 = time.time()
            m = fit(X[~te], y[~te], len(classes), a.epochs, a.batch, a.seed + k, a.colour_perm, a.width)
            prob[te] = m.predict(X[te], verbose=0)
            pred[te] = prob[te].argmax(1)
            acc = float((pred[te] == y[te]).mean())
            per_fold.append(dict(fold=k, n_test=int(te.sum()), accuracy=acc))
            print(f"fold {k}: test {te.sum():3d}  accuracy {acc:.3f}  ({time.time() - t0:.0f} s)")
        cv = metrics(y, pred, classes)
        conf = prob.max(1)
        cv["per_fold"] = per_fold
        cv["mean_confidence_correct"] = float(conf[pred == y].mean()) if (pred == y).any() else None
        cv["mean_confidence_wrong"] = float(conf[pred != y].mean()) if (pred != y).any() else None
        report["cv"] = cv
        print(f"CV accuracy {cv['accuracy']:.3f}  macro-F1 {cv['macro_f1']:.3f}")
        print("confusion (rows = true, cols = predicted, order", classes, ")")
        for c, row in zip(classes, cv["confusion"]):
            print(f"  {c:>8} {row}")
        save_confusion(cv["confusion"], classes, os.path.join(a.out, "confusion.png"),
                       f"blocked {a.folds}-fold CV  acc {cv['accuracy']:.2f}  macro-F1 {cv['macro_f1']:.2f}")
        wrong = np.where(pred != y)[0]
        sheet([X[i] for i in wrong], [f"{classes[y[i]][:4]}>{classes[pred[i]][:4]} {conf[i]:.2f}" for i in wrong],
              os.path.join(a.out, "errors.png"))
        report["cv_errors"] = [dict(id=ids[i], true=classes[y[i]], pred=classes[pred[i]], conf=float(conf[i]))
                               for i in wrong]

    m = fit(X, y, len(classes), a.epochs, a.batch, a.seed, a.colour_perm, a.width)
    m.save(os.path.join(a.out, "model.keras"))
    report["params"] = int(m.count_params())
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"final model ({report['params']} params) -> {a.out}")


if __name__ == "__main__":
    main()
