# src/data_gen/make_dataset.py
import os, csv, math, numpy as np
from typing import List, Tuple
from chirp import linear_chirp
from noise import white_gaussian, colored_1overf, ar1
from snr import sigma_for_target_snr, scale_noise_to_sigma, empirical_snr
from utils import SamplePath, make_uid, write_metadata_row, psd_slope_loglog

def param_grid():
    # Example small grid (expand freely)
    fs_choices = [1024, 2048]
    T_choices = [1.0, 2.0]
    f0_choices = [20.0, 30.0, 40.0]
    f1_choices = [80.0, 120.0, 180.0]
    return fs_choices, T_choices, f0_choices, f1_choices

def make_splits(num_total: int, seed: int, ratios=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    idx = np.arange(num_total); rng.shuffle(idx)
    n_train = int(ratios[0]*num_total)
    n_val   = int(ratios[1]*num_total)
    train = idx[:n_train]
    val   = idx[n_train:n_train+n_val]
    test  = idx[n_train+n_val:]
    return train, val, test

def build_dataset(
    out_dir: str,
    n_signal: int = 2000,
    n_noiseonly: int = 2000,
    snr_levels: List[float] = (2,3,4,5,6,8,10),
    noise_kinds: List[str] = ("white", "1overf_0.5", "1overf_1.0", "1overf_1.5"),
    seed: int = 12345,
    psd_slope_tol: float = 0.3  # loose tolerance for quick QC
):
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(out_dir, "data", "synthetic")
    os.makedirs(data_dir, exist_ok=True)
    meta_path = os.path.join(data_dir, "metadata.csv")

    rng_master = np.random.default_rng(seed)

    # Split indices over the union of samples:
    total = n_signal + n_noiseonly
    train_idx, val_idx, test_idx = make_splits(total, seed)

    # Open metadata writer
    fieldnames = [
        "id","split","label","snr_target","snr_empirical",
        "fs","N","T","f0","f1","phi0",
        "noise_type","alpha","ar1_phi",
        "seed","path"
    ]
    with open(meta_path, "w", newline="") as fmeta:
        writer = csv.DictWriter(fmeta, fieldnames=fieldnames)
        writer.writeheader()

        # Helper to map global index -> split
        def which_split(i):
            if i in train_idx: return "train"
            if i in val_idx:   return "val"
            return "test"

        # Build SIGNAL+NOISE samples
        fs_choices, T_choices, f0_choices, f1_choices = param_grid()

        # 1) SIGNAL-PRESENT
        for i in range(n_signal):
            uid = make_uid()
            split = which_split(i)
            # Draw parameters
            fs = int(rng_master.choice(fs_choices))
            T = float(rng_master.choice(T_choices))
            f0 = float(rng_master.choice(f0_choices))
            f1 = float(rng_master.choice(f1_choices))
            # Guard f1>f0
            if f1 <= f0:
                f1 = f0 + 40.0
            phi0 = float(2*np.pi*rng_master.random())
            s = linear_chirp(fs, T, f0, f1, phi0=phi0)  # ||s||_2 = 1
            N = s.size

            snr = float(rng_master.choice(snr_levels))
            sigma = sigma_for_target_snr(N, snr, signal_l2=1.0)

            # Pick noise type
            noise_kind = rng_master.choice(noise_kinds)
            if noise_kind == "white":
                noise_dir = "white"
                noise = np.random.default_rng(rng_master.integers(0, 2**31-1)).normal(0.0, 1.0, size=N)
                noise = scale_noise_to_sigma(noise, sigma)
                alpha = np.nan; ar1_phi = np.nan
            elif noise_kind.startswith("1overf_"):
                alpha = float(noise_kind.split("_")[1])
                noise_dir = f"colored_1overf_alpha{alpha}"
                gen = np.random.default_rng(rng_master.integers(0, 2**31-1))
                noise = colored_1overf(N, alpha=alpha, rng=gen)
                # scale to target sigma
                noise = scale_noise_to_sigma(noise, sigma)
                # QC: PSD slope (expect ~ -alpha/2)
                slope = psd_slope_loglog(noise)
                # No hard assert, but can log if needed
                ar1_phi = np.nan
            else:
                raise ValueError(f"Unknown noise_kind: {noise_kind}")

            x = s + noise
            snr_emp = empirical_snr(x, s)
            # QC SNR tolerance
            if abs(snr_emp/snr - 1.0) > 0.025:
                # Small corrective scaling of noise
                noise = scale_noise_to_sigma(x - s, sigma)
                x = s + noise
                snr_emp = empirical_snr(x, s)

            spath = SamplePath(base_dir=data_dir, split=split, noise_dir=noise_dir, uid=uid)
            npy_path = spath.npy_path()
            np.save(npy_path, x)

            write_metadata_row(writer, {
                "id": uid,
                "split": split,
                "label": 1,
                "snr_target": snr,
                "snr_empirical": snr_emp,
                "fs": fs, "N": N, "T": T,
                "f0": f0, "f1": f1, "phi0": phi0,
                "noise_type": "white" if noise_kind=="white" else "1overf",
                "alpha": alpha, "ar1_phi": ar1_phi,
                "seed": seed,
                "path": npy_path
            })

        # 2) NOISE-ONLY controls (label=0)
        for j in range(n_noiseonly):
            uid = make_uid()
            split = which_split(n_signal + j)

            # For noise-only, still pick fs,T for consistency
            fs = int(rng_master.choice(fs_choices))
            T = float(rng_master.choice(T_choices))
            N = int(round(fs*T))

            # Draw a “dummy” snr_target to mirror distribution (not used for scaling signal)
            snr = float(rng_master.choice(snr_levels))
            sigma = 1.0 / (snr * np.sqrt(N))  # just to keep “comparable” amplitude scale

            noise_kind = rng_master.choice(noise_kinds)
            if noise_kind == "white":
                noise_dir = "white"
                noise = np.random.default_rng(rng_master.integers(0, 2**31-1)).normal(0.0, 1.0, size=N)
                alpha = np.nan; ar1_phi = np.nan
            elif noise_kind.startswith("1overf_"):
                alpha = float(noise_kind.split("_")[1])
                noise_dir = f"colored_1overf_alpha{alpha}"
                gen = np.random.default_rng(rng_master.integers(0, 2**31-1))
                noise = colored_1overf(N, alpha=alpha, rng=gen)
                ar1_phi = np.nan
            else:
                raise ValueError(f"Unknown noise_kind: {noise_kind}")

            # Scale by sigma to keep amplitude ranges consistent with signal-present cases
            noise = scale_noise_to_sigma(noise, sigma)
            x = noise
            snr_emp = 0.0  # undefined; store 0

            spath = SamplePath(base_dir=data_dir, split=split, noise_dir=noise_dir, uid=uid)
            npy_path = spath.npy_path()
            np.save(npy_path, x)

            write_metadata_row(writer, {
                "id": uid,
                "split": split,
                "label": 0,
                "snr_target": snr,
                "snr_empirical": snr_emp,
                "fs": fs, "N": N, "T": T,
                "f0": math.nan, "f1": math.nan, "phi0": math.nan,
                "noise_type": "white" if noise_kind=="white" else "1overf",
                "alpha": alpha, "ar1_phi": ar1_phi,
                "seed": seed,
                "path": npy_path
            })

if __name__ == "__main__":
    # Example CLI usage
    build_dataset(out_dir=".", n_signal=500, n_noiseonly=500, seed=2025)
    print("Synthetic dataset generated under ./data/synthetic with metadata.csv")
