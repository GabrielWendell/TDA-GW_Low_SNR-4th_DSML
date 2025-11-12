# src/embed/qc.py
import os, json, numpy as np
import matplotlib.pyplot as plt
import smplotlib
from typing import Dict

def save_ami_curve(figpath: str, I: np.ndarray):
    plt.figure()
    plt.plot(np.arange(1, len(I)+1), I)
    plt.xlabel("$\\tau$")
    plt.ylabel("AMI")
    plt.title("AMI $(\\tau)$")
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

def save_fnn_curve(figpath: str, curve):
    plt.figure()
    ms = [c["m"] for c in curve]
    fs = [c["fnn"] for c in curve]
    plt.plot(ms, fs, marker="o")
    plt.xlabel("m")
    plt.ylabel("FNN fraction")
    plt.title("FNN (m)")
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

def save_pca_scatter(figpath: str, X: np.ndarray, n_comp: int = 3):
    from sklearn.decomposition import PCA
    k = min(n_comp, X.shape[1])
    P = PCA(n_components=k).fit_transform(X)
    plt.figure()
    if k == 2:
        plt.scatter(P[:,0], P[:,1], s=4, alpha=0.6)
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Embedding PCA (2D)")
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        ax = plt.figure().add_subplot(111, projection="3d")
        ax.scatter(P[:,0], P[:,1], P[:,2], s=4, alpha=0.6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title("Embedding PCA (3D)")
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()
