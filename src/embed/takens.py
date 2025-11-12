# src/embed/takens.py
import numpy as np
from typing import Tuple

def takens_embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Delay-embedding of a 1D series x into R^m with delay tau.
    Returns array of shape (N_embed, m).
    """
    if m < 2 or tau < 1:
        raise ValueError("Require m>=2 and tau>=1.")
    N = x.size
    N_embed = N - (m - 1) * tau
    if N_embed <= 0:
        raise ValueError("Not enough points for given (m, tau).")
    X = np.zeros((N_embed, m), dtype=float)
    for k in range(m):
        X[:, k] = x[(m - 1 - k) * tau : (m - 1 - k) * tau + N_embed]
    return X

def standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std()
    sd = 1.0 if sd == 0 else sd
    return (x - mu) / sd
