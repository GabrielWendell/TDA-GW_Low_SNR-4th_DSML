# src/ml/io.py
import os, json, csv
import numpy as np
from typing import Dict, Tuple

class SplitData:
    def __init__(self, X, y, ids, meta):
        self.X = X
        self.y = y
        self.ids = ids
        self.meta = meta  # list of dict rows

class FeatureSet:
    def __init__(self, train: SplitData, val: SplitData, test: SplitData, columns):
        self.train, self.val, self.test = train, val, test
        self.columns = columns


def _load_columns(dirpath: str):
    with open(os.path.join(dirpath, "columns.json"), "r") as f:
        return json.load(f)

def _load_index(dirpath: str):
    rows = []
    with open(os.path.join(dirpath, "index.csv"), "r") as f:
        rd = csv.DictReader(f)
        for r in rd: rows.append(r)
    return rows


def _pack_split(dirpath: str, split: str, rows):
    X = np.load(os.path.join(dirpath, f"features_{split}.npy"))
    # Rows in index.csv are in dataset order used when saving features
    # Select indices for this split preserving order
    idx = [i for i, r in enumerate(rows) if r["split"] == split]
    ids = [rows[i]["id"] for i in idx]
    y = np.array([int(rows[i]["label"]) for i in idx], dtype=int)
    meta = [rows[i] for i in idx]
    # Safety
    if X.shape[0] != len(idx):
        raise RuntimeError(f"Row count mismatch in {dirpath}/{split}: X = {X.shape[0]} vs index = {len(idx)}")
    if not np.isfinite(X).all():
        raise RuntimeError(f"Non‑finite values in {dirpath}/{split} features")
    return SplitData(X, y, ids, meta)


def load_feature_set(dirpath: str) -> FeatureSet:
    """Load features and aligned metadata for a vectorization directory (pi/pl/bc/baseline)."""
    cols = _load_columns(dirpath)
    rows = _load_index(dirpath)
    train = _pack_split(dirpath, "train", rows)
    val   = _pack_split(dirpath, "val", rows)
    test  = _pack_split(dirpath, "test", rows)
    return FeatureSet(train, val, test, cols)


def dataset_summary(fs: FeatureSet) -> Dict:
    return {
        "n_features": int(len(fs.columns)),
        "n_train": int(fs.train.X.shape[0]),
        "n_val": int(fs.val.X.shape[0]),
        "n_test": int(fs.test.X.shape[0]),
        "pos_train": int(fs.train.y.sum()),
        "pos_val": int(fs.val.y.sum()),
        "pos_test": int(fs.test.y.sum()),
    }