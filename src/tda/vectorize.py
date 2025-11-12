# src/tda/vectorize.py
import math
import os, json, csv
import numpy as np
from typing import Dict, Tuple, List

EPS = 1e-12

# ---------- Utilities ----------

def _finite_guard(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a[~np.isfinite(a)] = 0.0
    return a

def _finite_minmax(a: np.ndarray) -> tuple:
    """Return (amin, amax) over finite entries only; None if empty."""
    a = np.asarray(a, dtype = float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return float(a.min()), float(a.max())

def _load_pd(pd_path: str) -> Dict[str, np.ndarray]:
    z = np.load(pd_path)
    H0 = z["H0"] if "H0" in z else np.zeros((0,2))
    H1 = z["H1"] if "H1" in z else np.zeros((0,2))
    return {"H0": np.asarray(H0, float), "H1": np.asarray(H1, float)}

def _bp_bounds(pd_list: List[Dict[str, np.ndarray]], pad_frac: float = 0.02) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    """
    Compute pooled bounds on birth b and persistence p over finite entries only.
    Returns ((bmin,bmax), (pmin,pmax)).
    """
    bs, ps = [], []
    for PD in pd_list:
        for H in ("H0", "H1"):
            D = PD.get(H, None)
            if D is None or D.size == 0:
                continue
            b = D[:, 0]
            d = D[:, 1]
            p = np.maximum(d - b, 0.0)
            # Keep only finite values
            b = b[np.isfinite(b)]
            p = p[np.isfinite(p)]
            if b.size:
                bs.append(b)
            if p.size:
                ps.append(p)
    if not bs or not ps:
        return (0.0, 1.0), (0.0, 1.0)
    bcat = np.concatenate(bs)
    pcat = np.concatenate(ps)
    bmin, bmax = float(bcat.min()), float(bcat.max())
    pmin, pmax = float(max(pcat.min(), 0.0)), float(pcat.max())
    # Pad
    bw = bmax - bmin
    pw = pmax - pmin
    bmin -= pad_frac * (bw if bw > 0 else 1.0)
    bmax += pad_frac * (bw if bw > 0 else 1.0)
    pmin = 0.0
    pmax += pad_frac * (pw if pw > 0 else 1.0)
    # Final finite guards
    if not np.isfinite([bmin, bmax, pmin, pmax]).all():
        return (0.0, 1.0), (0.0, 1.0)
    return (bmin, bmax), (pmin, pmax)

# ---------- Persistence Image (PI) ----------

def _pi_single(D: np.ndarray, grid_b: np.ndarray, grid_p: np.ndarray, sigma: float, weight: str = "p") -> np.ndarray:
    """
    Build a PI for one diagram D (Nx2 birth, death) over a (len(grid_b), len(grid_p)) grid in (b,p).
    """
    B = len(grid_b); P = len(grid_p)
    if D.size == 0:
        return np.zeros((B,P), dtype = float)
    b = D[:,0]; d = D[:,1]; p = np.maximum(d - b, 0.0)
    if weight == "p":
        w = p
    elif weight == "1":
        w = np.ones_like(p)
    else:
        # Linear in p by default
        w = p
    # Precompute centers
    BB, PP = np.meshgrid(grid_b, grid_p, indexing = "ij")  # BxP
    I = np.zeros((B,P), dtype = float)
    sigma = max(float(sigma), 1e-8)
    inv2s2 = 1.0 / (2.0 * sigma * sigma + EPS)
    # Vectorized accumulation via broadcasting
    # shape: (N, B, P)
    db = (b[:,None,None] - BB[None,:,:])
    dp = (p[:,None,None] - PP[None,:,:])
    G = np.exp(-(db*db + dp*dp) * inv2s2)
    I = (w[:,None,None] * G).sum(axis = 0)
    return I

def vectorize_PI(pd_index_path: str, out_dir: str, grid_size: Tuple[int,int] = (50,50), sigma_frac: float=0.05, weight: str="p"):
    """
    Build PI features for both H0 and H1, concatenate [H0 | H1].
    """
    os.makedirs(out_dir, exist_ok = True)
    # Load PD index to get all rows and PDs
    rows = []
    PDS = []
    with open(pd_index_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            PDS.append(_load_pd(r["pd_path"]))
    # Bounds and grid
    (bmin,bmax), (pmin,pmax) = _bp_bounds(PDS, pad_frac = 0.02)
    B, P = grid_size
    grid_b = np.linspace(bmin, bmax, num=B)
    grid_p = np.linspace(pmin, pmax, num=P)
    side_b = (bmax - bmin) if (bmax > bmin) else 1.0
    side_p = (pmax - pmin) if (pmax > pmin) else 1.0
    sigma = float(sigma_frac * max(side_b, side_p))

    # Feature assembly
    feats = []
    index_out = []
    for r, PD in zip(rows, PDS):
        I0 = _pi_single(PD["H0"], grid_b, grid_p, sigma, weight=weight)
        I1 = _pi_single(PD["H1"], grid_b, grid_p, sigma, weight=weight)
        v = np.concatenate([I0.ravel(order="C"), I1.ravel(order="C")], axis=0)
        feats.append(_finite_guard(v))
        index_out.append({"id": r["id"], "split": r["split"], "label": r["label"],
                          "snr_target": r["snr_target"], "noise_type": r["noise_type"], "pd_path": r["pd_path"]})

    X = np.vstack(feats)
    # columns
    cols = []
    for h, tag in zip([("H0",0), ("H1",1)], ["H0","H1"]):
        for i in range(B):
            for j in range(P):
                cols.append(f"{tag}_PI_r{i}_c{j}")
    # Save per-split
    _save_by_split(out_dir, X, cols, index_out,
                   config={"grid_size": grid_size, "sigma_frac": sigma_frac, "weight": weight,
                           "bounds":{"bmin":bmin,"bmax":bmax,"pmin":pmin,"pmax":pmax}})

# ---------- Persistence Landscapes (PL) ----------

def _landscape_values(D: np.ndarray, tgrid: np.ndarray, K: int=5) -> np.ndarray:
    """
    Return array of shape (K, len(tgrid)) with k-th landscape values.
    """
    T = tgrid.size
    if D.size == 0:
        return np.zeros((K,T), dtype=float)
    b = D[:,0]; d = D[:,1]
    # For each (b,d), tent at t: max(0, min(t-b, d-t))
    # Vectorize: (N,T)
    Tmat = tgrid[None,:]
    tent = np.minimum(Tmat - b[:,None], d[:,None] - Tmat)
    tent = np.maximum(tent, 0.0)
    # At each t, sort descending and take first K
    tent.sort(axis=0)
    tent = tent[::-1,:]  # descending
    if tent.shape[0] < K:
        pad = np.zeros((K - tent.shape[0], T), dtype=float)
        tent = np.vstack([tent, pad])
    else:
        tent = tent[:K,:]
    return tent

def vectorize_PL(pd_index_path: str, out_dir: str, K: int = 5, T_samples: int = 256):
    os.makedirs(out_dir, exist_ok = True)
    rows = []
    PDS = []
    with open(pd_index_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            PDS.append(_load_pd(r["pd_path"]))
    # Pooled t-range from finite births/deaths
    bs, ds = [], []
    for PD in PDS:
        for H in ("H0", "H1"):
            D = PD[H]
            if D.size == 0:
                continue
            b = D[:, 0]
            d = D[:, 1]
            b = b[np.isfinite(b)]
            d = d[np.isfinite(d)]
            if b.size:
                bs.append(b)
            if d.size:
                ds.append(d)
    if not bs or not ds:
        tmin, tmax = 0.0, 1.0
    else:
        tmin = float(np.min(np.concatenate(bs)))
        tmax = float(np.max(np.concatenate(ds)))
        if not np.isfinite(tmin) or not np.isfinite(tmax) or tmax <= tmin:
            tmin, tmax = 0.0, 1.0
    tgrid = np.linspace(tmin, tmax, num = T_samples)

    feats = []
    index_out = []
    for r, PD in zip(rows, PDS):
        L0 = _landscape_values(PD["H0"], tgrid, K = K)  # (K,T)
        L1 = _landscape_values(PD["H1"], tgrid, K = K)
        v = np.concatenate([L0.ravel(order = "C"), L1.ravel(order = "C")], axis = 0)
        feats.append(_finite_guard(v))
        index_out.append({"id": r["id"], "split": r["split"], "label": r["label"],
                          "snr_target": r["snr_target"], "noise_type": r["noise_type"], "pd_path": r["pd_path"]})
    X = np.vstack(feats)
    # Columns
    cols = []
    for tag in ["H0","H1"]:
        for k in range(1, K+1):
            for i in range(T_samples):
                cols.append(f"{tag}_PL_k{k}_t{i}")
    _save_by_split(out_dir, X, cols, index_out,
                   config = {"K":K, "T_samples":T_samples, "tmin":tmin, "tmax":tmax})

# ---------- Betti Curves (BC) ----------

def _betti_curve(D: np.ndarray, egrid: np.ndarray) -> np.ndarray:
    """
    beta(e) = # of intervals with b <= e < d
    """
    if D.size == 0:
        return np.zeros_like(egrid)
    b = D[:,0]; d = D[:,1]
    # For each e, count how many intervals cover it
    # Vectorized using broadcasting (N,E)
    E = egrid[None,:]
    mask = (b[:,None] <= E) & (E < d[:,None])
    return mask.sum(axis = 0).astype(float)

def vectorize_BC(pd_index_path: str, out_dir: str, E_samples: int=200):
    os.makedirs(out_dir, exist_ok = True)
    rows = []
    PDS = []
    with open(pd_index_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            PDS.append(_load_pd(r["pd_path"]))
    # Pooled epsilon range over finite births/deaths
    bs, ds = [], []
    for PD in PDS:
        for H in ("H0", "H1"):
            D = PD[H]
            if D.size == 0:
                continue
            b = D[:, 0]
            d = D[:, 1]
            b = b[np.isfinite(b)]
            d = d[np.isfinite(d)]
            if b.size:
                bs.append(b)
            if d.size:
                ds.append(d)
    if not bs or not ds:
        emin, emax = 0.0, 1.0
    else:
        emin = float(np.min(np.concatenate(bs)))
        emax = float(np.max(np.concatenate(ds)))
        if not np.isfinite(emin) or not np.isfinite(emax) or emax <= emin:
            emin, emax = 0.0, 1.0
    egrid = np.linspace(emin, emax, num = E_samples)

    feats = []
    index_out = []
    for r, PD in zip(rows, PDS):
        B0 = _betti_curve(PD["H0"], egrid)
        B1 = _betti_curve(PD["H1"], egrid)
        v = np.concatenate([B0, B1], axis=0)
        feats.append(_finite_guard(v))
        index_out.append({"id": r["id"], "split": r["split"], "label": r["label"],
                          "snr_target": r["snr_target"], "noise_type": r["noise_type"], "pd_path": r["pd_path"]})
    X = np.vstack(feats)
    # columns
    cols = [f"H0_BC_e{i}" for i in range(E_samples)] + [f"H1_BC_e{i}" for i in range(E_samples)]
    _save_by_split(out_dir, X, cols, index_out,
                   config={"E_samples":E_samples, "emin":emin, "emax":emax})

# ---------- Saving helper ----------

def _save_by_split(out_dir: str, X: np.ndarray, cols: List[str], index_rows: List[Dict], config: Dict):
    # Write columns.json and config.json
    with open(os.path.join(out_dir, "columns.json"), "w") as f:
        json.dump(cols, f, indent = 2, allow_nan = False)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent = 2, allow_nan = False)
    # Write index.csv
    idx_path = os.path.join(out_dir, "index.csv")
    with open(idx_path, "w", newline = "") as fidx:
        w = csv.DictWriter(fidx, fieldnames = ["id","split","label","snr_target","noise_type","pd_path"])
        w.writeheader()
        for r in index_rows: w.writerow(r)
    # Split matrices
    splits = {"train": [], "val": [], "test": []}
    for i, r in enumerate(index_rows):
        splits[r["split"]].append(i)
    for sp, idxs in splits.items():
        idxs = np.array(idxs, dtype = int)
        Xsp = X[idxs] if idxs.size else np.zeros((0, X.shape[1]))
        np.save(os.path.join(out_dir, f"features_{sp}.npy"), Xsp)