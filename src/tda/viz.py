# src/tda/viz.py
import os, numpy as np, matplotlib.pyplot as plt, smplotlib

def plot_pd(pd: np.ndarray, title: str, path: str):
    if pd.size == 0:
        return
    b, d = pd[:,0], pd[:,1]
    lim = max(float(np.max(d)), float(np.max(b)))
    plt.figure()
    plt.scatter(b, d, s = 8, alpha = 0.7)
    plt.plot([0, lim], [0, lim], "k--", linewidth = 1)
    plt.xlabel("birth"); plt.ylabel("death"); plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok = True)
    plt.savefig(path); plt.close()
