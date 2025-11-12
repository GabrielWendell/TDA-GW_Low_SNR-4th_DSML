import os, json, csv, time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from ..embed.takens import takens_embed, standardize
from ..tda.pd import vr_persistence_diagrams
from ..robust.ablation_embed import _pi_from_PD, _pl_from_PD, _bc_from_PD

# Optional aesthetics (won't fail if missing)
try:
    import smplotlib  # noqa: F401
except Exception:
    pass


def _load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def _summary_stats(values: List[float]) -> Dict[str, float]:
    import math
    arr = np.asarray(values, float)
    n = arr.size
    if n == 0:
        return dict(mean = np.nan, std = np.nan, ci_lo = np.nan, ci_hi = np.nan, n = 0)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof = 1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 1 else 0.0
    ci_half = 1.96 * se
    return dict(mean = mean, std = std, ci_lo = mean - ci_half, ci_hi = mean + ci_half, n = int(n))


def run_profile(project_root: str = '.', n_samples: int = 30, max_points: int = 2000, seed: int = 2025):
    """Profile per-signal runtime of embedding, PD, and PI/PL/BC vectorization.

    We:
      1) load synthetic metadata + embedding params,
      2) sample up to n_samples signals,
      3) for each signal, time: Takens embedding, VR persistence, PI/PL/BC.
    """
    rng = np.random.default_rng(seed)

    # --- Paths ---
    data_root = os.path.join(project_root, 'data')
    emb_root  = os.path.join(data_root, 'embeddings')
    syn_root  = os.path.join(data_root, 'synthetic')

    # Vectorizer configs (same as used in Step 2.2)
    cfg_pi = _load_json(os.path.join(data_root, 'pi', 'config.json'))
    cfg_pl = _load_json(os.path.join(data_root, 'pl', 'config.json'))
    cfg_bc = _load_json(os.path.join(data_root, 'bc', 'config.json'))

    # Embedding parameters: one jsonl line per id with m_star, tau_star
    params_path = os.path.join(emb_root, 'params.jsonl')
    params = {}
    with open(params_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            # Assume keys: {'id', 'm_star', 'tau_star', ...}
            params[rec['id']] = rec

    # Synthetic metadata with path, snr_target, etc.
    meta = {}
    with open(os.path.join(syn_root, 'metadata.csv'), 'r') as f:
        rd = csv.DictReader(f)
        for r in rd:
            meta[r['id']] = r

    # Sample ids
    all_ids = sorted(set(meta.keys()) & set(params.keys()))
    if len(all_ids) == 0:
        raise RuntimeError('No overlap between metadata and embedding params ids.')
    if n_samples > len(all_ids):
        n_samples = len(all_ids)
    sample_ids = list(rng.choice(np.array(all_ids), size = n_samples, replace = False))

    # Containers for per-sample runtimes
    t_emb, t_pd, t_pi, t_pl, t_bc = [], [], [], [], []

    # Profile loop
    for uid in sample_ids:
        m_star = int(params[uid]['m_star'])
        tau_star = int(params[uid]['tau_star'])
        # Load raw series
        path = meta[uid]['path']
        if not os.path.isabs(path):
            path = os.path.join(project_root, path)
        x = np.load(path)
        x_std = standardize(x)

        # 1) Embedding
        X, dt_emb = _time_call(takens_embed, x_std, m_star, tau_star)
        t_emb.append(dt_emb)

        # 2) Persistence diagrams (VR up to H1)
        pd_res, dt_pd = _time_call(
            vr_persistence_diagrams,
            X,
            maxdim = 1,
            max_points = max_points,
            seed = seed,
        )
        # vr_persistence_diagrams returns (PD_dict, info)
        PD, _info = pd_res
        t_pd.append(dt_pd)

        # 3) Vectorizations (PI / PL / BC)
        _, dt_pi = _time_call(_pi_from_PD, PD, cfg_pi)
        _, dt_pl = _time_call(_pl_from_PD, PD, cfg_pl)
        _, dt_bc = _time_call(_bc_from_PD, PD, cfg_bc)
        t_pi.append(dt_pi)
        t_pl.append(dt_pl)
        t_bc.append(dt_bc)

    # Summarize
    stats = {
        'embedding': _summary_stats(t_emb),
        'vr_pd':     _summary_stats(t_pd),
        'pi':        _summary_stats(t_pi),
        'pl':        _summary_stats(t_pl),
        'bc':        _summary_stats(t_bc),
    }

    out_dir = os.path.join(project_root, 'results', 'robust', 'profile')
    os.makedirs(out_dir, exist_ok=True)

    # Write CSV summary
    import pandas as pd
    rows = []
    for name, s in stats.items():
        row = dict(component = name, **s)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'runtime_summary.csv'), index = False)

    # Bar plot with error bars
    comps = [r['component'] for r in rows]
    means = [r['mean'] for r in rows]
    ci_lo = [r['mean'] - r['ci_lo'] for r in rows]
    ci_hi = [r['ci_hi'] - r['mean'] for r in rows]
    yerr = np.vstack([ci_lo, ci_hi])

    x = np.arange(len(comps))
    plt.figure(figsize = (6,4))
    plt.bar(x, means)
    plt.errorbar(x, means, yerr = yerr, fmt = 'none', ecolor = 'black', elinewidth = 1, capsize = 3)
    plt.xticks(x, comps, rotation=30, ha='right')
    plt.ylabel('Runtime per signal (seconds)')
    plt.title('Per-component runtime (mean ± 95% CI)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'runtime_components.png'), dpi = 300)
    plt.close()

    print('\n[PROFILE] wrote runtime_summary.csv and runtime_components.png to', out_dir)