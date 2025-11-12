# src/data_gen/chirp.py
import numpy as np

def linear_chirp(fs: int, T: float, f0: float, f1: float, phi0: float = None) -> np.ndarray:
    """
    Energy-normalized linear chirp: ||s||_2 = 1.
    Parameters
    ----------
    fs : sampling rate [Hz]
    T  : duration [s]
    f0 : start frequency [Hz]
    f1 : end frequency [Hz]
    phi0 : initial phase [rad] (random if None)
    """
    N = int(round(fs * T))
    t = np.arange(N) / fs
    if phi0 is None:
        phi0 = 2.0 * np.pi * np.random.rand()
    alpha = (f1 - f0) / T
    phase = 2.0 * np.pi * (f0 * t + 0.5 * alpha * t**2) + phi0
    s = np.cos(phase)
    # Energy normalize
    norm = np.linalg.norm(s)
    if norm == 0:
        raise ValueError("Generated zero-norm chirp.")
    return s / norm
