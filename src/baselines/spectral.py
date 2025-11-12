# src/baseline/spectral.py
import numpy as np
from typing import Tuple

def _welch_psd(x: np.ndarray, fs: float):
    try:
        from scipy.signal import welch, get_window
        N = x.size
        # pick nperseg sensibly
        p2 = 1 << (N-1).bit_length()
        nperseg = min(1024, max(256, p2//2))
        f, Pxx = welch(x, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg//2, detrend="constant", return_onesided=True, scaling="density")
        return f, Pxx
    except Exception:
        # Fallback: periodogram with Hann window
        w = np.hanning(x.size)
        X = np.fft.rfft((x - x.mean()) * w)
        P = (np.abs(X)**2) / (np.sum(w**2))
        f = np.fft.rfftfreq(x.size, 1.0/fs)
        return f, P

def spectral_features(x: np.ndarray, fs: float, band_edges=(0,50,100,200)) -> dict:
    f, S = _welch_psd(x, fs)
    S = np.maximum(S, 1e-12)
    Sf = S.sum()
    feats = {}
    # peak
    ipeak = int(np.argmax(S)); feats["f_peak"] = float(f[ipeak]); feats["S_peak"] = float(S[ipeak])
    # centroid & bandwidth
    mu_f = float((f*S).sum() / Sf); feats["spec_centroid"] = mu_f
    feats["spec_bandwidth"] = float(np.sqrt(((f - mu_f)**2 * S).sum() / Sf))
    # rolloff 95%
    cumsum = np.cumsum(S)
    thr = 0.95 * Sf
    idx = int(np.searchsorted(cumsum, thr))
    if idx >= f.size: idx = f.size - 1
    feats["spec_rolloff95"] = float(f[idx])
    # flatness GM/AM
    gm = float(np.exp(np.mean(np.log(S))))
    am = float(np.mean(S))
    feats["spec_flatness"] = gm / am
    # entropy
    p = S / Sf
    feats["spec_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    # slope in log-log (interior)
    lo = max(1, int(0.02 * f.size)); hi = int(0.90 * f.size)
    logf = np.log(f[lo:hi] + 1e-12); logm = np.log(np.sqrt(S[lo:hi]) + 1e-12)
    A = np.vstack([logf, np.ones_like(logf)]).T
    slope, intercept = np.linalg.lstsq(A, logm, rcond=None)[0]
    feats["spec_slope"] = float(slope)
    # bandpowers
    edges = [e for e in band_edges if e < fs/2] + [fs/2]
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (f >= a) & (f < b)
        feats[f"bandpower_{a:.0f}_{b:.0f}"] = float(S[mask].sum() / Sf) if mask.any() else 0.0
    return feats

def spectral_flux(x: np.ndarray, fs: float) -> float:
    # Simple flux with a fixed short frame; normalized magnitude spectra
    try:
        from scipy.signal import stft, get_window
        f, t, Z = stft(x, fs=fs, window="hann", nperseg=256, noverlap=128, detrend=False, boundary=None)
        M = np.abs(Z)
        M = M / (M.sum(axis=0, keepdims=True) + 1e-12)
        d = np.diff(M, axis=1)
        return float(np.mean(np.sqrt((d**2).sum(axis=0))))
    except Exception:
        return 0.0
