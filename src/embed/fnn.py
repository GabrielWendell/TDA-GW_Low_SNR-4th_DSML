# src/embed/fnn.py
import numpy as np
from typing import Tuple, Dict
from takens import takens_embed  # keep as-is if you run the script from src/embed

def _nearest_neighbors_idx(X: np.ndarray) -> np.ndarray:
    """
    For each row in X, return the index of its nearest neighbor (exclude self).
    Naive O(N^2) for robustness.
    """
    N = X.shape[0]
    idx_nn = np.empty(N, dtype=int)
    for i in range(N):
        d2 = np.sum((X - X[i])**2, axis=1)
        d2[i] = np.inf
        j = int(np.argmin(d2))
        if not np.isfinite(d2[j]):
            # Degenerate: all identical; pick self-proxy and handle upstream
            j = i  # will be caught by downstream test
        idx_nn[i] = j
    return idx_nn

def fnn_fraction(x: np.ndarray, tau: int, m: int,
                 R_tol: float = 10.0, A_tol: float = 2.0,
                 downsample: int = 1) -> float:
    """
    Fraction of false nearest neighbors at dimension m using (R_tol, A_tol).
    Downsampling is applied consistently to Xm and Xmp1, then both are truncated
    to a common length to avoid index mismatches.
    """
    Xm = takens_embed(x, m, tau)
    Xmp1 = takens_embed(x, m+1, tau)

    # Consistent downsampling
    if downsample > 1:
        Xm_ds = Xm[::downsample]
        Xmp1_ds = Xmp1[::downsample]
    else:
        Xm_ds = Xm
        Xmp1_ds = Xmp1

    # Align lengths — Xmp1 is usually shorter by tau before downsampling
    Np = min(Xm_ds.shape[0], Xmp1_ds.shape[0])
    if Np < 2:
        # Not enough points to form meaningful NN statistics
        return 1.0  # pessimistic fallback; encourages higher m

    Xm_ds = Xm_ds[:Np]
    Xmp1_ds = Xmp1_ds[:Np]

    # std for A criterion
    sigma = x.std() if x.std() > 0 else 1.0

    # Nearest neighbors computed on the truncated Xm_ds
    idx_nn = _nearest_neighbors_idx(Xm_ds)

    false_count = 0
    for i in range(Np):
        j = idx_nn[i]
        if j == i:
            # Degenerate NN case → treat as false
            false_count += 1
            continue

        d2_m = np.sum((Xm_ds[i] - Xm_ds[j])**2)
        if d2_m <= 0.0 or not np.isfinite(d2_m):
            false_count += 1
            continue

        d2_mp1 = np.sum((Xmp1_ds[i] - Xmp1_ds[j])**2)
        R = (d2_mp1 - d2_m) / d2_m

        # Amplitude test using the additional coordinate in m+1 (last column)
        A = abs(Xmp1_ds[i, -1] - Xmp1_ds[j, -1]) / sigma

        if (R > R_tol) or (A > A_tol):
            false_count += 1

    return false_count / float(Np)

def choose_m(x: np.ndarray, tau: int, m_min: int = 2, m_max: int = 12,
             R_tol: float = 10.0, A_tol: float = 2.0, eps: float = 0.01,
             downsample: int = 1) -> Tuple[int, Dict]:
    curve = []
    m_star = m_max
    hit = False
    for m in range(m_min, m_max + 1):
        fnn = fnn_fraction(x, tau, m, R_tol=R_tol, A_tol=A_tol, downsample=downsample)
        curve.append((m, fnn))
        if not hit and fnn < eps:
            m_star = m
            hit = True
    info = {
        "fnn_curve": [{"m": int(m), "fnn": float(f)} for m, f in curve],
        "m_min": int(m_min), "m_max": int(m_max),
        "R_tol": float(R_tol), "A_tol": float(A_tol),
        "eps": float(eps),
        "downsample": int(downsample),
        "qc_fnn_never_below_eps": (not hit)
    }
    return m_star, info
