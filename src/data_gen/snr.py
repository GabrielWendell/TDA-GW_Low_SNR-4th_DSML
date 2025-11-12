# src/data_gen/snr.py
import numpy as np

def sigma_for_target_snr(N: int, target_snr: float, signal_l2: float = 1.0) -> float:
    """
    Given ||s||_2 and N, return noise std so that rho = ||s||_2 / (sigma * sqrt(N)) equals target_snr.
    """
    return signal_l2 / (target_snr * np.sqrt(N))

def empirical_snr(x: np.ndarray, s: np.ndarray) -> float:
    """
    Empirical time-domain SNR defined by ||s||_2 / (std(noise) * sqrt(N)).
    """
    n = x - s
    N = x.size
    sigma_hat = n.std(ddof=0)
    s_l2 = np.linalg.norm(s)
    return s_l2 / (sigma_hat * np.sqrt(N))

def scale_noise_to_sigma(noise: np.ndarray, sigma: float, eps: float = 1e-12) -> np.ndarray:
    var = noise.var() + eps
    return noise * (sigma / np.sqrt(var))
