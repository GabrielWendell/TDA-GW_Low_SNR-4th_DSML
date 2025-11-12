# src/embed/run_embeddings.py
import os, json, csv, numpy as np
from typing import Optional
from ami import choose_tau
from fnn import choose_m
from takens import takens_embed, standardize
from qc import save_ami_curve, save_fnn_curve, save_pca_scatter

def run_embeddings(project_root: str = ".", store_pointclouds: bool = False,
                   qc_plots: bool = True, seed: int = 2025):
    data_dir = os.path.join(project_root, "data", "synthetic")
    meta_path = os.path.join(data_dir, "metadata.csv")
    out_dir = os.path.join(project_root, "data", "embeddings")
    qc_dir = os.path.join(project_root, "results", "embedding_qc")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    params_path = os.path.join(out_dir, "params.jsonl")
    rng = np.random.default_rng(seed)

    with open(meta_path, "r") as fmeta, open(params_path, "w") as fout:
        reader = csv.DictReader(fmeta)
        for row in reader:
            uid = row["id"]; split = row["split"]; path = row["path"]
            fs = int(row["fs"]); N = int(float(row["N"]))
            # load series
            x = np.load(path)
            x_std = standardize(x)  # improves AMI/FNN stability

            # Select tau via AMI
            tau_star, ami_info = choose_tau(x_std)

            # Select m via FNN
            m_star, fnn_info = choose_m(x_std, tau_star, m_min=2, m_max=12, eps=0.01, downsample=2)

            # Optional: compute embedding and store
            X = None
            embed_path = None
            if store_pointclouds:
                X = takens_embed(x_std, m_star, tau_star)
                embed_dir = os.path.join(out_dir, split)
                os.makedirs(embed_dir, exist_ok=True)
                embed_path = os.path.join(embed_dir, f"{uid}_m{m_star}_t{tau_star}.npy")
                np.save(embed_path, X)

            # QC plots for a small, stratified subset (here: 1% or at least a few)
            if qc_plots and (rng.random() < 0.02 or uid.endswith("0")):
                # AMI curve
                save_ami_curve(os.path.join(qc_dir, f"{uid}_ami.png"), np.array(ami_info["ami_curve"]))
                # FNN curve
                save_fnn_curve(os.path.join(qc_dir, f"{uid}_fnn.png"), fnn_info["fnn_curve"])
                # PCA scatter if embedding computed or compute ad-hoc
                if X is None:
                    X = takens_embed(x_std, m_star, tau_star)
                save_pca_scatter(os.path.join(qc_dir, f"{uid}_embed_pca.png"), X, n_comp=min(3, m_star))

            # Write sidecar JSON line
            record = {
                "id": uid,
                "split": split,
                "fs": fs,
                "N": N,
                "tau_star": int(tau_star),
                "m_star": int(m_star),
                "ami_bins": ami_info["ami_bins"],
                "ami_tau_max": ami_info["tau_max"],
                "ami_first_local_min_idx0": ami_info["first_local_min_idx0"],
                "qc_ami_no_local_min": ami_info["qc_ami_no_local_min"],
                "fnn_curve": fnn_info["fnn_curve"],
                "qc_fnn_never_below_eps": fnn_info["qc_fnn_never_below_eps"],
                "embed_path": embed_path
            }
            fout.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    run_embeddings(project_root=".", store_pointclouds=False, qc_plots=True, seed=2025)
    print("Embedding parameter sweep completed. See data/embeddings/params.jsonl and results/embedding_qc/")
