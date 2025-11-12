# src/data_gen/utils.py
import os, csv, uuid, numpy as np
from dataclasses import dataclass

@dataclass
class SamplePath:
    base_dir: str
    split: str
    noise_dir: str
    uid: str

    def npy_path(self) -> str:
        os.makedirs(os.path.join(self.base_dir, self.split, self.noise_dir), exist_ok=True)
        return os.path.join(self.base_dir, self.split, self.noise_dir, f"{self.uid}.npy")

def make_uid() -> str:
    return uuid.uuid4().hex

def write_metadata_row(writer, row: dict):
    writer.writerow(row)

def psd_slope_loglog(x: np.ndarray) -> float:
    """
    Very light PSD slope estimator: use rfft magnitude vs frequency on log-log region
    ignoring the lowest ~1% and the highest ~5% bins.
    """
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(x.size, d=1.0)
    mag = np.abs(X) + 1e-12
    # Keep interior band
    lo = int(0.01 * f.size)
    hi = int(0.95 * f.size)
    f_sel = f[lo:hi]; m_sel = mag[lo:hi]
    logf = np.log(f_sel + 1e-12)
    logm = np.log(m_sel)
    A = np.vstack([logf, np.ones_like(logf)]).T
    slope, _ = np.linalg.lstsq(A, logm, rcond=None)[0]
    # For 1/f^alpha noise, |X(f)| ~ f^{-alpha/2} => slope ≈ -alpha/2
    return slope
