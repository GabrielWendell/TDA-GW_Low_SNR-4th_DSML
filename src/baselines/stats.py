# src/baseline/stats.py
import numpy as np
from scipy.signal import hilbert

def zero_crossing_rate(x: np.ndarray) -> float:
    x = np.asarray(x)
    s = np.signbit(x)
    return np.mean(s[:-1] != s[1:])

def hjorth_params(x: np.ndarray):
    x = np.asarray(x)
    var_x = np.var(x)
    if var_x == 0: var_x = 1.0
    dx = np.diff(x, prepend=x[0])
    var_dx = np.var(dx)
    mob = np.sqrt(var_dx / var_x)
    ddx = np.diff(dx, prepend=dx[0])
    var_ddx = np.var(ddx)
    mob_dx = np.sqrt(var_ddx / (var_dx if var_dx>0 else 1.0))
    comp = mob_dx / mob if mob>0 else 0.0
    return var_x, mob, comp

def autocorr_peak_lag(x: np.ndarray, max_lag: int = None):
    x = np.asarray(x) - np.mean(x)
    N = x.size
    if max_lag is None: max_lag = min(N-1, 4096)
    # unbiased autocorr via FFT
    nfft = 1 << (N*2-1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    S = X * np.conj(X)
    r = np.fft.irfft(S, n=nfft)[:N] / (np.arange(N,0,-1))
    # ignore lag 0
    if max_lag >= N: max_lag = N-1
    k = 1 + np.argmax(r[1:max_lag+1])
    return int(k), float(r[k])

def envelope_stats(x: np.ndarray):
    a = np.abs(hilbert(x))
    return float(np.mean(a)), float(np.std(a)), float(np.max(a))
