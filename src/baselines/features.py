# src/baseline/features.py
import numpy as np
from typing import Dict
from .stats import zero_crossing_rate, hjorth_params, autocorr_peak_lag, envelope_stats
from .spectral import spectral_features, spectral_flux

def ar_yule_walker(x: np.ndarray, order: int = 4) -> np.ndarray:
    # Yule–Walker via autocorrelation, Toeplitz solve
    x = np.asarray(x) - np.mean(x)
    N = x.size
    nfft = 1 << (2*N-1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    S = X * np.conj(X)
    r = np.fft.irfft(S, n=nfft)[:order+1] / (np.arange(N, N-order-1, -1))
    R = np.lib.stride_tricks.sliding_window_view(r[:-1], order)[0]
    # Toeplitz R: r[0], r[1],...,r[p-1]
    # Build Toeplitz explicitly for stability
    T = np.empty((order, order))
    for i in range(order):
        for j in range(order):
            T[i, j] = r[abs(i-j)]
    a = np.linalg.solve(T, r[1:order+1])
    return a

def compute_time_features(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x)
    mu = float(np.mean(x)); sd = float(np.std(x)); rms = float(np.sqrt(np.mean(x**2))); energy = float(np.sum(x**2))
    # numerical guards
    sd_safe = sd if sd > 0 else 1.0
    skew = float(np.mean(((x - mu)/sd_safe)**3))
    kurt = float(np.mean(((x - mu)/sd_safe)**4) - 3.0)
    peak = float(np.max(np.abs(x)))
    ptr = float(peak / (rms if rms>0 else 1.0))
    crest = float(peak / sd_safe)
    zcr = zero_crossing_rate(x)
    act, mob, comp = hjorth_params(x)
    k_lag, k_val = autocorr_peak_lag(x)
    env_mean, env_std, env_max = envelope_stats(x)
    feats = {
        "mean": mu, "std": sd, "rms": rms, "energy": energy,
        "skewness": skew, "kurtosis": kurt, "peak": peak,
        "peak_to_rms": ptr, "crest_factor": crest, "zcr": zcr,
        "hjorth_activity": float(act), "hjorth_mobility": float(mob), "hjorth_complexity": float(comp),
        "autocorr_peak_lag": float(k_lag), "autocorr_peak_val": float(k_val),
        "env_mean": env_mean, "env_std": env_std, "env_max": env_max
    }
    # AR(4)
    try:
        a = ar_yule_walker(x, order=4)
        for i, ai in enumerate(a, start=1):
            feats[f"ar4_a{i}"] = float(ai)
    except Exception:
        for i in range(1,5):
            feats[f"ar4_a{i}"] = 0.0
    return feats

def compute_baseline_features(x: np.ndarray, fs: float) -> Dict[str, float]:
    feats = {}
    feats.update(compute_time_features(x))
    feats.update(spectral_features(x, fs))
    feats["spec_flux"] = spectral_flux(x, fs)
    return feats
