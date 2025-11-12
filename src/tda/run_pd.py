# src/tda/run_pd.py
import os, csv, json, time, numpy as np
from typing import Optional
from ..embed.takens import takens_embed, standardize
from .pd import vr_persistence_diagrams

def _load_params(params_path: str):
    # id -> record dict
    params = {}
    with open(params_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            params[rec["id"]] = rec
    return params

def run_pd(
    project_root: str = ".",
    store_pointclouds: bool = False,
    max_points: int = 3000,
    use_backend: Optional[str] = None,
    seed: int = 2025,
    make_qc_plots: bool = True
):
    data_dir = os.path.join(project_root, "data", "synthetic")
    meta_path = os.path.join(data_dir, "metadata.csv")
    embed_dir = os.path.join(project_root, "data", "embeddings")
    params_path = os.path.join(embed_dir, "params.jsonl")

    out_dir = os.path.join(project_root, "data", "pd")
    os.makedirs(out_dir, exist_ok = True)
    qc_dir = os.path.join(project_root, "results", "pd_qc")
    os.makedirs(qc_dir, exist_ok = True)

    # Load tau*, m*
    params = _load_params(params_path)

    index_rows = []
    with open(meta_path, "r") as fmeta:
        reader = csv.DictReader(fmeta)
        for row in reader:
            uid = row["id"]; split = row["split"]; path = row["path"]
            fs = float(row["fs"])
            label = int(row["label"])
            snr_target = float(row["snr_target"])
            noise_type = row["noise_type"]

            # Standardize raw series, re-embed using tau*, m*
            x = np.load(path)
            x_std = standardize(x)

            if uid not in params:
                # If missing, skip or set defaults
                print(f"[WARN] Missing embedding params for {uid}, skipping.")
                continue
            tau_star = int(params[uid]["tau_star"])
            m_star = int(params[uid]["m_star"])

            X = takens_embed(x_std, m_star, tau_star)

            # Compute PDs
            t0 = time.time()
            (pd_dict, info) = vr_persistence_diagrams(
                X, maxdim = 1, max_points = max_points, seed = seed, use_backend = use_backend,
                max_edge_percentile=None  # Optionally set e.g. 90.0
            )
            dt_ms = int(1000.0 * (time.time() - t0))

            # Save per-sample PD
            pd_path = os.path.join(out_dir, f"{uid}_pd.npz")
            np.savez_compressed(pd_path, H0 = pd_dict["H0"], H1 = pd_dict["H1"])

            # Optional QC: plot a few PDs
            if make_qc_plots and (hash(uid) % 25 == 0):
                try:
                    import matplotlib.pyplot as plt
                    import smplotlib
                    for k in ("H0","H1"):
                        D = pd_dict[k]
                        if D.size == 0:
                            continue
                        b, d = D[:,0], D[:,1]
                        plt.figure()
                        plt.scatter(b, d, s = 6, alpha = 0.7)
                        lim = max(float(np.max(d)), float(np.max(b))) if D.size else 1.0
                        plt.plot([0, lim], [0, lim], "k--", linewidth = 1)
                        plt.xlabel("birth"); plt.ylabel("death"); plt.title(f"{uid} $-$ {k}")
                        plt.tight_layout()
                        plt.savefig(os.path.join(qc_dir, f"{uid}_{k}.png"))
                        plt.close()
                except Exception:
                    pass

            # Summary metrics
            H1 = pd_dict["H1"]
            max_pers_H1 = float(np.max(H1[:,1] - H1[:,0])) if H1.size else 0.0

            index_rows.append({
                "id": uid, "split": split, "label": label,
                "snr_target": snr_target, "noise_type": noise_type,
                "tau_star": tau_star, "m_star": m_star,
                "n_H0": int(pd_dict["H0"].shape[0]),
                "n_H1": int(pd_dict["H1"].shape[0]),
                "max_persistence_H1": max_pers_H1,
                "runtime_ms": dt_ms,
                "backend": info["backend"],
                "max_points_used": info["max_points_used"],
                "pd_path": pd_path
            })

    # Write index CSV
    idx_path = os.path.join(out_dir, "index.csv")
    with open(idx_path, "w", newline="") as fidx:
        cols = ["id","split","label","snr_target","noise_type","tau_star","m_star",
                "n_H0","n_H1","max_persistence_H1","runtime_ms","backend","max_points_used","pd_path"]
        w = csv.DictWriter(fidx, fieldnames=cols)
        w.writeheader()
        for r in index_rows:
            w.writerow(r)

    print(f"PDs written to {out_dir}. Index: {idx_path}")

if __name__ == "__main__":
    run_pd(project_root = ".", store_pointclouds = False, max_points = 3000, use_backend = None, seed = 2025)
