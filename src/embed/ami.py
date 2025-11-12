# src/embed/ami.py
import numpy as np
from typing import Tuple, Dict

def freedman_diaconis_bins(x: np.ndarray, max_bins: int = 128) -> int:
    x = np.asarray(x)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    n = x.size
    bw = 2 * iqr * (n ** (-1/3))
    if bw <= 0:
        return min(32, max_bins)
    bins = int(np.ceil((x.max() - x.min()) / bw))
    return max(16, min(bins, max_bins))

def mutual_information_delay(x: np.ndarray, tau: int, bins: int) -> float:
    """
    Histogram-based MI estimator between x_t and x_{t-tau}.
    """
    if tau <= 0 or tau >= x.size:
        return 0.0
    x0 = x[tau:]
    x1 = x[:-tau]
    H, xe, ye = np.histogram2d(x0, x1, bins = bins)
    Pxy = H / H.sum()
    Px = Pxy.sum(axis = 1, keepdims = True)
    Py = Pxy.sum(axis = 0, keepdims = True)
    with np.errstate(divide = "ignore", invalid = "ignore"):
        I = Pxy * (np.log(Pxy + 1e-12) - np.log(Px + 1e-12) - np.log(Py + 1e-12))
    return float(np.nansum(I))

def first_local_minimum(y: np.ndarray) -> int:
    """
    Return index of first local minimum in y[1:len-2] (strict on both sides).
    If none, return -1.
    """
    for i in range(1, len(y) - 1):
        if y[i] < y[i-1] and y[i] < y[i+1]:
            return i
    return -1

def choose_tau(x: np.ndarray, tau_max: int = None, bins: int = None) -> Tuple[int, Dict]:
    N = x.size
    if tau_max is None:
        tau_max = max(5, min(200, N // 10))
    if bins is None:
        bins = freedman_diaconis_bins(x, max_bins = 128)

    I = np.zeros(tau_max, dtype = float)
    for tau in range(1, tau_max + 1):
        I[tau - 1] = mutual_information_delay(x, tau, bins = bins)

    idx = first_local_minimum(I)
    qc_flag_no_local_min = False
    if idx < 0:
        # Fallback: choose smallest tau with I(tau) <= median(I)
        medI = np.median(I)
        candidates = np.where(I <= medI)[0]
        if candidates.size > 0:
            idx = int(candidates[0])
            qc_flag_no_local_min = True
        else:
            idx = int(np.argmin(I))
            qc_flag_no_local_min = True

    tau_star = idx + 1
    info = {
        "ami_curve": I.tolist(),
        "ami_bins": int(bins),
        "tau_max": int(tau_max),
        "first_local_min_idx0": int(idx),
        "qc_ami_no_local_min": bool(qc_flag_no_local_min)
    }
    return tau_star, info
