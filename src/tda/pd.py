# src/tda/pd.py
import numpy as np
from typing import Dict, Tuple, Optional

def _try_import_ripser():
    try:
        from ripser import ripser
        return ripser
    except Exception:
        return None

def _try_import_gudhi():
    try:
        import gudhi as gd
        return gd
    except Exception:
        return None

def _subsample_points(X: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if X.shape[0] <= max_points:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_points, replace = False)
    return X[idx]

def vr_persistence_diagrams(
    X: np.ndarray,
    maxdim: int = 1,
    max_points: int = 3000,
    seed: int = 2025,
    use_backend: Optional[str] = None,
    max_edge_percentile: Optional[float] = None
) -> Dict[str, np.ndarray]:
    """
    Compute VR persistence diagrams up to H1 from point cloud X.
    Returns dict with 'H0' and 'H1' arrays of shape (n_k, 2) (birth, death).
    """
    X = np.asarray(X, dtype = float)
    X = _subsample_points(X, max_points = max_points, seed = seed)

    backend = use_backend
    ripser = _try_import_ripser()
    gd = _try_import_gudhi()

    if backend is None:
        if ripser is not None:
            backend = "ripser"
        elif gd is not None:
            backend = "gudhi"
        else:
            raise ImportError("Neither ripser nor gudhi is available. Please install one of them.")

    if backend == "ripser":
        # Determine numeric 'thresh' for ripser
        # Default: unbounded filtration radius
        thresh = float("inf")
        if max_edge_percentile is not None:
            # Estimate a numeric cap from sampled pairwise distances
            rng = np.random.default_rng(seed)
            M = X.shape[0]
            if M >= 2:
                s = min(5000, M*(M-1)//2)
                i_idx = rng.integers(0, M, size=s)
                j_idx = rng.integers(0, M, size=s)
                mask = i_idx != j_idx
                i_idx, j_idx = i_idx[mask], j_idx[mask]
                if i_idx.size > 0:
                    d = np.linalg.norm(X[i_idx] - X[j_idx], axis=1)
                    thresh = float(np.percentile(d, max_edge_percentile))
                else:
                    thresh = float("inf")
            else:
                # Not enough points to estimate; fall back to inf
                thresh = float("inf")

        res = ripser(X, maxdim = maxdim, thresh = thresh)
        dgms = res["dgms"]
        out = {"H0": np.array(dgms[0], dtype = float)}
        if maxdim >= 1:
            out["H1"] = np.array(dgms[1], dtype = float)
        else:
            out["H1"] = np.zeros((0, 2), dtype = float)
        info = {"backend": "ripser", "max_points_used": int(X.shape[0])}
        return out, info

    elif backend == "gudhi":
        # GUDHI VR complex
        R = gd.RipsComplex(points = X.tolist(), max_edge_length = None)
        st = R.create_simplex_tree(max_dimension = maxdim+1)
        st.compute_persistence()
        H0 = []
        H1 = []
        for dim, pair in st.persistence():
            b, d = pair
            if dim == 0:
                H0.append((float(b), float(d)))
            elif dim == 1:
                H1.append((float(b), float(d)))
        out = {"H0": np.array(H0, dtype = float), "H1": np.array(H1, dtype = float)}
        info = {"backend": "gudhi", "max_points_used": int(X.shape[0])}
        return out, info

    else:
        raise ValueError(f"Unknown backend: {backend}")