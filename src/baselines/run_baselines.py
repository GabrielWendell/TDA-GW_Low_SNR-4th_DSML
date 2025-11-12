# src/baseline/run_baselines.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import os, csv, json, numpy as np
from typing import List, Dict
from ..embed.takens import standardize  # reuse standardization if desired
from .features import compute_baseline_features

def run_baselines(project_root: str = ".", standardize_input: bool = False):
    data_dir = os.path.join(project_root, "data", "synthetic")
    meta_path = os.path.join(data_dir, "metadata.csv")
    out_dir = os.path.join(project_root, "data", "baseline")
    os.makedirs(out_dir, exist_ok=True)

    # First pass: collect features, record columns
    indices: List[Dict] = []
    feats_all: Dict[str, list] = {}
    sample_count = 0
    with open(meta_path, "r") as fmeta:
        reader = csv.DictReader(fmeta)
        for row in reader:
            uid = row["id"]; split = row["split"]; path = row["path"]
            fs = float(row["fs"])
            label = int(row["label"])
            snr_target = float(row["snr_target"])
            noise_type = row["noise_type"]

            x = np.load(path)
            if standardize_input:
                x = standardize(x)

            feats = compute_baseline_features(x, fs)

            # 1) Ensure any NEW feature key is backfilled to current sample_count
            for k in feats.keys():
                if k not in feats_all:
                    feats_all[k] = [0.0] * sample_count

            # 2) Append this sample's value for EVERY known key (default 0)
            for k in feats_all.keys():
                feats_all[k].append(feats.get(k, 0.0))

            indices.append({
                "id": uid, "split": split, "label": label,
                "snr_target": snr_target, "noise_type": noise_type
            })
            sample_count += 1

    # Optional sanity check: all columns must have len == sample_count
    for k, v in feats_all.items():
        if len(v) != sample_count:
            raise RuntimeError(f"Column length mismatch for {k}: {len(v)} vs {sample_count}")

    # Convert to arrays and split by 'split'
    cols = sorted(feats_all.keys())
    X = np.vstack([np.array(feats_all[c], dtype=float) for c in cols]).T  # shape (M, D)

    # write columns.json
    with open(os.path.join(out_dir, "columns.json"), "w") as f:
        json.dump(cols, f, indent=2)

    # write index CSV
    index_path = os.path.join(out_dir, "features_index.csv")
    with open(index_path, "w", newline="") as fidx:
        w = csv.DictWriter(fidx, fieldnames=["id","split","label","snr_target","noise_type"])
        w.writeheader()
        for r in indices:
            w.writerow(r)

    # split and save matrices
    splits = {"train": [], "val": [], "test": []}
    for i, r in enumerate(indices):
        splits[r["split"]].append(i)

    for sp, idxs in splits.items():
        idxs = np.array(idxs, dtype=int)
        Xsp = X[idxs] if idxs.size else np.zeros((0, X.shape[1]))
        np.save(os.path.join(out_dir, f"features_{sp}.npy"), Xsp)

    print("Baseline features saved under data/baseline/:",
          "columns.json, features_index.csv, and features_[train|val|test].npy")

if __name__ == "__main__":
    run_baselines(project_root=".", standardize_input=False)
