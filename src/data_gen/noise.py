# src/data_gen/noise.py
import numpy as np

def white_gaussian(N: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, sigma, size=N)

def colored_1overf(N: int, alpha: float, rng: np.random.Generator, eps: float = 1e-6) -> np.ndarray:
    """
    Generate 1/f^alpha colored noise by spectral shaping of white noise.
    """
    # White noise
    w = rng.normal(0.0, 1.0, size=N)
    # FFT
    W = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(N, d=1.0)  # assumes unit sampling interval; scale doesn't matter for shape
    # Avoid f=0 blow-up; regularize low frequencies
    mag = (freqs + eps) ** (-alpha / 2.0)
    # Shape
    C = W * mag
    c = np.fft.irfft(C, n=N)
    # Zero-mean
    c -= c.mean()
    return c

def ar1(N: int, phi: float, rng: np.random.Generator, sigma_eps: float = 1.0) -> np.ndarray:
    """
    AR(1): x_t = phi*x_{t-1} + eps_t, eps ~ N(0, sigma_eps^2)
    """
    x = np.zeros(N)
    for n in range(1, N):
        x[n] = phi * x[n-1] + rng.normal(0.0, sigma_eps)
    x -= x.mean()
    return x
