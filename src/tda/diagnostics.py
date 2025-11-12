# src/tda/diagnostics.py
import os, csv, json, time
import numpy as np
import matplotlib.pyplot as plt
import smplotlib
from typing import Dict, List, Tuple, Optional

# --- Reuse from previous steps ---
from ..embed.takens import takens_embed, standardize
from .pd import vr_persistence_diagrams

# Try optional packages
def _try_persim():
    try:
        from persim import bottleneck
        return bottleneck
    except Exception:
        return None

BOTTLE = _try_persim()

# ---------- I/O helpers ----------

def _load_pd_npz(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path)
    H0 = np.asarray(z["H0"] if "H0" in z else np.zeros((0,2)), float)
    H1 = np.asarray(z["H1"] if "H1" in z else np.zeros((0,2)), float)
    return {"H0": H0, "H1": H1}

def _load_params(params_path: str) -> Dict[str, dict]:
    out = {}
    with open(params_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out

# ---------- Stats over a single diagram ----------

def _pd_stats(D: np.ndarray) -> Dict[str, float]:
    if D.size == 0:
        return dict(n = 0, p_mean = 0.0, p_med = 0.0, p95 = 0.0, p_max = 0.0)
    b = D[:,0]; d = D[:,1]
    p = np.maximum(d - b, 0.0)
    p_sorted = np.sort(p)
    q95 = float(p_sorted[int(0.95*(p_sorted.size-1))]) if p_sorted.size else 0.0
    return dict(
        n = int(p.size),
        p_mean = float(np.mean(p)),
        p_med  = float(np.median(p)),
        p95    = q95,
        p_max  = float(np.max(p))
    )

# ---------- Aggregate SNR summaries ----------

def summarize_by_snr(pd_index_csv: str, out_csv: str) -> None:
    rows = []
    with open(pd_index_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    out_rows = []
    for r in rows:
        PD = _load_pd_npz(r["pd_path"])
        for Hname in ("H0","H1"):
            S = _pd_stats(PD[Hname])
            out_rows.append({
                "id": r["id"], "split": r["split"], "label": int(r["label"]),
                "snr_target": float(r["snr_target"]), "noise_type": r["noise_type"],
                "homology": Hname, **S
            })

    # Group by (split, homology, SNR)
    # Write simple long CSV; plotting/aggregation can be done later
    with open(out_csv, "w", newline = "") as fout:
        cols = ["id","split","label","snr_target","noise_type","homology",
                "n","p_mean","p_med","p95","p_max"]
        w = csv.DictWriter(fout, fieldnames = cols)
        w.writeheader()
        for rr in out_rows:
            w.writerow(rr)

# ---------- Distribution plots ----------

def _collect_values_by_snr(pd_index_csv: str, hom: str, field: str = "p") -> Dict[str, np.ndarray]:
    """
    Field in {"b","d","p"} to collect births, deaths, or persistence by SNR bucket (stringified snr_target).
    Returns dict snr_str -> 1D array of values.
    """
    by = {}
    with open(pd_index_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            snr_key = f"{float(r['snr_target']):.1f}"
            PD = _load_pd_npz(r["pd_path"])[hom]
            if PD.size == 0:
                continue
            b = PD[:,0]; d = PD[:,1]; p = np.maximum(d-b, 0.0)
            arr = {"b": b, "d": d, "p": p}[field]
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            by.setdefault(snr_key, []).append(arr)
    # Concat per key
    for k in list(by.keys()):
        by[k] = np.concatenate(by[k]) if len(by[k]) else np.zeros(0)
    return by

def plot_histograms_by_snr(pd_index_csv: str, out_dir: str, hom: str, nbins: int = 60):
    os.makedirs(out_dir, exist_ok=True)
    for field, title in [("b","Births"), ("d","Deaths"), ("p","Persistence")]:
        by = _collect_values_by_snr(pd_index_csv, hom = hom, field = field)
        if not by:
            continue
        # Global bins for comparability
        cat = np.concatenate([v for v in by.values()]) if by else np.zeros(0)
        if cat.size == 0:
            continue
        cat = cat[np.isfinite(cat)]
        lo, hi = float(np.min(cat)), float(np.max(cat))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
        bins = np.linspace(lo, hi, nbins+1)

        plt.figure()
        for snr_key in sorted(by.keys(), key = lambda s: float(s)):
            v = by[snr_key]
            plt.hist(v, bins = bins, alpha = 0.5, label = f"SNR {snr_key}", density = True, ec = 'k')
        plt.xlabel(title)
        plt.ylabel("Density")
        plt.title(f"{hom} : {title} by SNR")
        plt.legend(loc = "best", fontsize = 8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pd_hist_{hom}_{field}.png"))
        plt.close()

# ---------- PD overlay for representatives ----------

def plot_pd_examples(pd_index_csv: str, out_dir: str, k_each: int = 1):
    os.makedirs(out_dir, exist_ok = True)
    # Pick reps: for label = 1 (signal), lowest/mid/high SNR; for label = 0, pick white and 1/f if present
    rows = []
    with open(pd_index_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows: return
    # Candidates
    sig = [r for r in rows if int(r["label"]) == 1]
    bkg = [r for r in rows if int(r["label"]) == 0]
    # Sort by SNR
    sig_sorted = sorted(sig, key = lambda r: float(r["snr_target"]))
    picks = []
    if len(sig_sorted) >= 3:
        picks += [sig_sorted[0], sig_sorted[len(sig_sorted)//2], sig_sorted[-1]]
    else:
        picks += sig_sorted
    # Background reps by noise_type
    types = {}
    for r in bkg:
        types.setdefault(r["noise_type"], []).append(r)
    for nt, lst in types.items():
        picks += lst[:k_each]

    for r in picks:
        PD = _load_pd_npz(r["pd_path"])
        for H in ("H0","H1"):
            D = PD[H]
            plt.figure()
            if D.size > 0:
                b, d = D[:,0], D[:,1]
                lim = float(max(np.max(b), np.max(d)))
                plt.scatter(b, d, s = 12, alpha = 0.5)
            else:
                lim = 1.0
            plt.plot([0, lim], [0, lim], "k--", lw = 1)
            plt.xlabel("Birth"); plt.ylabel("Death")
            plt.title(f"{r['id']} \n {H} (SNR = {float(r['snr_target']):.1f}, label = {r['label']}, noise = {r['noise_type']})")
            plt.tight_layout()
            fn = os.path.join(out_dir, f"pd_scatter_{r['id']}_{H}.png")
            plt.savefig(fn); plt.close()

# ---------- Stability via small-noise perturbation ----------

def _median_pairwise_dist(X: np.ndarray, max_pairs: int = 10000, seed: int = 2025) -> float:
    n = X.shape[0]
    if n < 2: return 0.0
    rng = np.random.default_rng(seed)
    s = min(max_pairs, n*(n-1)//2)
    i = rng.integers(0, n, size = s)
    j = rng.integers(0, n, size = s)
    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0: return 0.0
    d = np.linalg.norm(X[i] - X[j], axis=1)
    return float(np.median(d))

def _bottleneck_fallback(D1: np.ndarray, D2: np.ndarray) -> float:
    """
    Approximate bottleneck with L∞ NN matching allowing the diagonal.
    For a point (b,d), distance to diagonal in L∞ is (d-b)/2.
    """
    if D1.size == 0 and D2.size == 0: return 0.0
    def diag_dist(pt):  # L∞ distance to diagonal
        b, d = pt
        return 0.5 * max(d - b, 0.0)
    def set_to_set(A, B):
        if A.size == 0:
            return max(diag_dist(pt) for pt in B) if B.size else 0.0
        if B.size == 0:
            return max(diag_dist(pt) for pt in A) if A.size else 0.0
        # for each a in A, min over B ∪ {diag}
        m = []
        for a in A:
            abd = abs(a[0]-B[:,0]); add = abs(a[1]-B[:,1])
            dmin = float(np.min(np.maximum(abd, add)))  # L∞ to nearest point
            ddiag = diag_dist(a)
            m.append(min(dmin, ddiag))
        return max(m) if m else 0.0
    return max(set_to_set(D1, D2), set_to_set(D2, D1))

def stability_for_sample(uid: str,
                         x: np.ndarray,
                         m_star: int, tau_star: int,
                         repeats: int = 5,
                         alpha: float = 1e-3,
                         max_points: int = 2000,
                         seed: int = 2025,
                         use_backend: Optional[str] = None) -> Dict[str, float]:
    """
    Re-embed standardized x with (m*,tau*), compute PD once (baseline),
    then R times with Gaussian jitter N(0, (alpha * med_dist)^2) added to the embedding.
    Return average distances (bottleneck preferred) for H0 and H1.
    """
    x_std = standardize(x)
    X = takens_embed(x_std, m_star, tau_star)
    # Baseline PD
    PD0, _ = vr_persistence_diagrams(X, maxdim = 1, max_points = max_points, seed = seed, use_backend = use_backend)
    # Jitter scale from median pairwise distance
    med = _median_pairwise_dist(X, seed=seed)
    sigma = alpha * (med if med > 0 else 1.0)
    rng = np.random.default_rng(seed)

    dists_H0, dists_H1 = [], []
    for r in range(repeats):
        noise = rng.normal(0.0, sigma, size = X.shape)
        Xp = X + noise
        PDp, _ = vr_persistence_diagrams(Xp, maxdim = 1, max_points = max_points, seed = seed + 113*r, 
                                         use_backend = use_backend)
        # distances
        if BOTTLE is not None:
            dH0 = float(BOTTLE(PD0["H0"], PDp["H0"]))
            dH1 = float(BOTTLE(PD0["H1"], PDp["H1"]))
        else:
            dH0 = _bottleneck_fallback(PD0["H0"], PDp["H0"])
            dH1 = _bottleneck_fallback(PD0["H1"], PDp["H1"])
        dists_H0.append(dH0); dists_H1.append(dH1)

    return {
        "uid": uid,
        "sigma": float(sigma),
        "dH0_mean": float(np.mean(dists_H0)), "dH0_std": float(np.std(dists_H0)),
        "dH1_mean": float(np.mean(dists_H1)), "dH1_std": float(np.std(dists_H1))
    }

def run_diagnostics(project_root: str = ".",
                    repeats: int = 3,
                    alpha: float = 1e-3,
                    max_points: int = 2000,
                    seed: int = 2025,
                    use_backend: Optional[str] = None):
    # Paths
    pd_dir = os.path.join(project_root, "data", "pd")
    pd_index = os.path.join(pd_dir, "index.csv")

    embed_dir = os.path.join(project_root, "data", "embeddings")
    params_path = os.path.join(embed_dir, "params.jsonl")

    data_dir = os.path.join(project_root, "data", "synthetic")
    meta_path = os.path.join(data_dir, "metadata.csv")

    out_dir = os.path.join(project_root, "results", "diag")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Per-SNR summaries
    summarize_by_snr(pd_index, os.path.join(out_dir, "pd_stats_by_snr.csv"))

    # 2) Histograms for H0/H1: births, deaths, persistence
    for hom in ("H0","H1"):
        plot_histograms_by_snr(pd_index, out_dir, hom = hom, nbins = 60)

    # 3) Representative PD overlays
    plot_pd_examples(pd_index, out_dir, k_each = 1)

    # 4) Stability analysis (stratified subset)
    # Load metadata and params
    params = _load_params(params_path)
    # Choose ~N reps across SNRs and labels for speed
    rows = []
    with open(pd_index, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    # Stratify: 3 with label=1 across SNRs, and 2 with label=0 across noise types
    sig = [r for r in rows if int(r["label"]) == 1]
    bkg = [r for r in rows if int(r["label"]) == 0]
    sig_sorted = sorted(sig, key = lambda r: float(r["snr_target"]))
    picks = []
    picks += sig_sorted[:1] + sig_sorted[len(sig_sorted)//2:len(sig_sorted)//2+1] + sig_sorted[-1:] if sig_sorted else []
    # Add up to 2 backgrounds of different noise types
    by_noise = {}
    for r in bkg:
        by_noise.setdefault(r["noise_type"], []).append(r)
    for nt, lst in by_noise.items():
        if len(picks) >= 5: break
        picks.append(lst[0])

    # Load raw series table for x paths
    path_by_id = {}
    with open(os.path.join(data_dir, "metadata.csv"), "r") as f:
        rd = csv.DictReader(f)
        for rr in rd:
            path_by_id[rr["id"]] = rr["path"]

    stab_rows = []
    for r in picks:
        uid = r["id"]
        if uid not in params or uid not in path_by_id:
            continue
        x = np.load(path_by_id[uid])
        tau_star = int(params[uid]["tau_star"])
        m_star   = int(params[uid]["m_star"])
        res = stability_for_sample(uid, x, m_star, tau_star,
                                   repeats = repeats, alpha = alpha,
                                   max_points = max_points, seed = seed,
                                   use_backend = use_backend)
        res.update({
            "split": r["split"], "label": int(r["label"]),
            "snr_target": float(r["snr_target"]), "noise_type": r["noise_type"]
        })
        stab_rows.append(res)

    # Write stability index
    stab_cols = ["uid","split","label","snr_target","noise_type","sigma",
                 "dH0_mean","dH0_std","dH1_mean","dH1_std"]
    with open(os.path.join(out_dir, "stability_index.csv"), "w", newline = "") as f:
        w = csv.DictWriter(f, fieldnames = stab_cols); w.writeheader()
        for rr in stab_rows: w.writerow(rr)

    # Boxplot of H1 stability by SNR (if enough data)
    if stab_rows:
        snrs = sorted({ rr["snr_target"] for rr in stab_rows })
        data = []
        for s in snrs:
            vals = [rr["dH1_mean"] for rr in stab_rows if rr["snr_target"] == s]
            if vals: data.append(vals)
        if data:
            plt.figure()
            plt.boxplot(data, labels = [f"{s:.1f}" for s in snrs], showfliers = False)
            plt.xlabel("SNR"); plt.ylabel("Bottleneck distance (H1)")
            plt.title("Stability under small embedding jitter")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "stability_boxplot.png"))
            plt.close()

    print(f"[DIAG] Wrote summaries & plots to {out_dir}")