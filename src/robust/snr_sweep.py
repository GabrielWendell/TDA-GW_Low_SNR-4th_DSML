import os, json, joblib, csv, numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from ..ml.io import load_feature_set

# Optional aesthetics (do not fail if missing)
try:
    import smplotlib  # noqa: F401
except Exception:
    pass

# ---------- Helpers ----------

def _probs(pipe, X):
    if hasattr(pipe, "predict_proba"): return pipe.predict_proba(X)[:,1]
    s = pipe.decision_function(X); smin, smax = float(np.min(s)), float(np.max(s))
    return (s - smin) / (smax - smin + 1e-12)

_def_invalid = set('<>:"/\\|?*')

def _sanitize_filename(s: str) -> str:
    return ''.join('_' if c in _def_invalid else c for c in s)


def _metrics(y, p, thresh=0.5):
    y = np.asarray(y, int); p = np.asarray(p, float)
    yhat = (p >= thresh).astype(int)
    uniq = np.unique(y)
    auc = float("nan") if uniq.size < 2 else float(roc_auc_score(y, p))
    ap  = float("nan") if uniq.size < 2 else float(average_precision_score(y, p))
    f1  = float("nan") if uniq.size < 2 else float(f1_score(y, yhat))
    br  = float(brier_score_loss(y, p))
    return dict(auc = auc, ap = ap, f1 = f1, brier = br)


def _strat_boot_idx(y, rng):
    y = np.asarray(y, int)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    ib = np.concatenate([
        rng.choice(pos, size = pos.size, replace = True) if pos.size else np.array([], int),
        rng.choice(neg, size = neg.size, replace = True) if neg.size else np.array([], int)
    ])
    return ib


def metric_ci(y, p, B = 500, seed = 2025):
    rng = np.random.default_rng(seed)
    aucs, aps, f1s, brs = [], [], [], []
    for _ in range(B):
        idx = _strat_boot_idx(y, rng)
        m = _metrics(y[idx], p[idx])
        if np.isfinite(m['auc']): aucs.append(m['auc'])
        if np.isfinite(m['ap']):  aps.append(m['ap'])
        if np.isfinite(m['f1']):  f1s.append(m['f1'])
        brs.append(m['brier'])
    def q(v):
        return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if len(v) else [np.nan, np.nan]
    return dict(auc_ci = q(aucs), ap_ci = q(aps), f1_ci = q(f1s), brier_ci = q(brs))

# ---------- Core ----------

def train_high_snr_clone(project_root: str, feature_tag: str, model_name: str, snr_min: float):
    """Clone the saved pipeline (same hyperparams), refit on TRAIN+VAL restricted to SNR>=snr_min."""
    # Load features
    feat_dir = os.path.join(project_root, 'data', feature_tag if feature_tag != 'baseline' else 'baseline')
    fs = load_feature_set(feat_dir)
    # Load baseline fitted model to extract hyperparams
    base_path = os.path.join(project_root, 'models', feature_tag, f'{model_name}.joblib')
    pipe_fitted = joblib.load(base_path)
    pipe = clone(pipe_fitted)  # Unfitted
    # Filter train+val by SNR
    snr_tr = np.array([float(m['snr_target']) for m in fs.train.meta])
    snr_va = np.array([float(m['snr_target']) for m in fs.val.meta])
    mask_tr = snr_tr >= snr_min
    mask_va = snr_va >= snr_min
    Xtv = np.vstack([fs.train.X[mask_tr], fs.val.X[mask_va]])
    ytv = np.concatenate([fs.train.y[mask_tr], fs.val.y[mask_va]])
    pipe.fit(Xtv, ytv)
    # Save (Windows-safe)
    out_dir = os.path.join(project_root, 'models_snr', feature_tag)
    os.makedirs(out_dir, exist_ok = True)
    tag = f"{model_name}_trainSNR_ge_{snr_min:.1f}"
    tag = _sanitize_filename(tag)
    joblib.dump(pipe, os.path.join(out_dir, f'{tag}.joblib'))
    return pipe


def eval_per_snr(pipe, Xte, yte, snr_te, snr_levels):
    rows = []
    for s in snr_levels:
        # Float-safe mask
        mask = np.isclose(snr_te, s, atol = 1e-6)
        if not np.any(mask):
            continue
        p = _probs(pipe, Xte[mask])
        m = _metrics(yte[mask], p)
        ci = metric_ci(yte[mask], p, B = 500, seed = 2025)
        m.update(ci)
        m.update(dict(snr = float(s), n = int(mask.sum())))
        rows.append(m)
    return rows


def compare_baseline_vs_high(project_root: str, feature_tag: str, model_name: str, snr_thr: float, snr_levels):
    # Load data
    feat_dir = os.path.join(project_root, 'data', feature_tag if feature_tag != 'baseline' else 'baseline')
    fs = load_feature_set(feat_dir)
    snr_te = np.array([float(m['snr_target']) for m in fs.test.meta])

    # Load models: baseline (train+val on full SNR) and high-SNR refit
    base_path = os.path.join(project_root, 'models', feature_tag, f'{model_name}.joblib')
    base_pipe = joblib.load(base_path)
    high_path = os.path.join(project_root, 'models_snr', feature_tag, f'{model_name}_trainSNR_ge_{snr_thr:.1f}.joblib')
    high_pipe = joblib.load(high_path)

    base_rows = eval_per_snr(base_pipe, fs.test.X, fs.test.y, snr_te, snr_levels)
    high_rows = eval_per_snr(high_pipe, fs.test.X, fs.test.y, snr_te, snr_levels)

    # Join by snr
    by_snr = { r['snr']: { 'baseline': r } for r in base_rows }
    for r in high_rows:
        by_snr.setdefault(r['snr'], {})['high'] = r

    # Write csvs
    out_dir = os.path.join(project_root, 'results', 'robust', 'snr_sweep')
    os.makedirs(out_dir, exist_ok = True)
    comp_path = os.path.join(out_dir, f'{feature_tag}_{model_name}_comparison_{snr_thr:.1f}.csv')
    with open(comp_path, 'w', newline = '') as f:
        cols = [
            'snr','n',
            'auc_base','auc_high','delta_auc',
            'ap_base','ap_high','delta_ap',
            'brier_base','brier_high','delta_brier',
            'f1_base','f1_high','delta_f1'
        ]
        w = csv.DictWriter(f, fieldnames = cols); w.writeheader()
        for s, d in sorted(by_snr.items()):
            b = d.get('baseline', {}); h = d.get('high', {})
            row = dict(
                snr = float(s), n = int(b.get('n', 0) or h.get('n', 0)),
                auc_base = b.get('auc', np.nan), auc_high = h.get('auc', np.nan), delta_auc = (h.get('auc', np.nan) - b.get('auc', np.nan)),
                ap_base = b.get('ap', np.nan), ap_high = h.get('ap', np.nan),   delta_ap = (h.get('ap', np.nan) - b.get('ap', np.nan)),
                brier_base = b.get('brier', np.nan), brier_high = h.get('brier', np.nan), delta_brier = (h.get('brier', np.nan) - b.get('brier', np.nan)),
                f1_base = b.get('f1', np.nan), f1_high = h.get('f1', np.nan), delta_f1 = (h.get('f1', np.nan) - b.get('f1', np.nan))
            )
            w.writerow(row)

    # Per-model per-snr metrics (high-SNR model only)
    met_path = os.path.join(out_dir, f'{feature_tag}_{model_name}_trainSNR_ge_{snr_thr:.1f}_test_metrics.csv')
    with open(met_path, 'w', newline = '') as f:
        cols = ['snr','n','auc','auc_ci_lo','auc_ci_hi','ap','ap_ci_lo','ap_ci_hi','brier','brier_ci_lo','brier_ci_hi','f1','f1_ci_lo','f1_ci_hi']
        w = csv.DictWriter(f, fieldnames = cols); w.writeheader()
        for r in sorted(high_rows, key = lambda z: z['snr']):
            w.writerow(dict(
                snr = r['snr'], n = r['n'],
                auc = r['auc'], auc_ci_lo = r['auc_ci'][0], auc_ci_hi = r['auc_ci'][1],
                ap = r['ap'], ap_ci_lo = r['ap_ci'][0], ap_ci_hi = r['ap_ci'][1],
                brier = r['brier'], brier_ci_lo = r['brier_ci'][0], brier_ci_hi = r['brier_ci'][1],
                f1 = r['f1'], f1_ci_lo = r['f1_ci'][0], f1_ci_hi = r['f1_ci'][1]
            ))

    return comp_path, met_path

# ---------- Plotting ----------

def plot_curves(project_root: str, feature_tag: str, model_name: str, snr_thr: float):
    import pandas as pd
    out_dir = os.path.join(project_root, 'results', 'robust', 'snr_sweep', 'plots')
    os.makedirs(out_dir, exist_ok = True)
    comp_path = os.path.join(project_root, 'results', 'robust', 'snr_sweep', f'{feature_tag}_{model_name}_comparison_{snr_thr:.1f}.csv')
    df = pd.read_csv(comp_path)
    # AUC vs SNR (baseline vs high)
    plt.figure(figsize = (6,4))
    plt.plot(df['snr'], df['auc_base'], 'o--', label = 'baseline train+val (full SNR)')
    plt.plot(df['snr'], df['auc_high'], 'o-',  label = f'high-SNR train (≥{snr_thr:.1f})')
    plt.xlabel('SNR (test)'); plt.ylabel('AUC'); plt.title(f'{feature_tag.upper()} / {model_name}')
    plt.legend(loc = 'best'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{feature_tag}_{model_name}_auc_vs_snr_{snr_thr:.1f}.png'), dpi = 300)
    plt.close()
    # ΔAUC bar
    plt.figure(figsize = (6,4))
    plt.bar(df['snr'], df['delta_auc'])
    plt.axhline(0, color = 'k', lw = 1)
    plt.xlabel('SNR (test)'); plt.ylabel('$\\Delta$AUC (high$-$base)')
    plt.title(f'{feature_tag.upper()} / {model_name} $-$ $\\Delta$AUC')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{feature_tag}_{model_name}_delta_auc_{snr_thr:.1f}.png'), dpi = 300)
    plt.close()