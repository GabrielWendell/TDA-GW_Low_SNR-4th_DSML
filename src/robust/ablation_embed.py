import os, json, csv, joblib, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from ..embed.takens import takens_embed, standardize
from ..tda.pd import vr_persistence_diagrams
from ..ml.io import load_feature_set

# Optional aesthetics
try:
    import smplotlib  # noqa: F401
except Exception:
    pass

# ---------------- Vectorization Helpers (reuse existing configs) ----------------
# We vectorize perturbed PDs into the SAME feature space as baseline by reusing
# config.json (grid/bounds) and columns.json. Minimal single-PD vectorizers are
# implemented here consistent with Step 2.2 design.

def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# --- Persistence Image (PI) ---

def _pi_from_PD(PD: Dict[str, np.ndarray], cfg: Dict) -> np.ndarray:
    """
    Build a concatenated PI vector for H0 and H1 using cfg.

    Supported schemas:
    A) Per-homology edges:
       {"H0": {"x_edges": [...], "y_edges": [...]}, "H1": {...}, "sigma": 0.05}
    B) Shared top-level edges:
       {"x_edges": [...], "y_edges": [...], "sigma": 0.05}
    C) Compact bounds/grid form:
       {"grid_size": int|[nx,ny], "sigma_frac": float,
        "bounds": one of
           {"x":[xmin,xmax], "y":[ymin,ymax]} or
           {"birth":[xmin,xmax], "pers":[ymin,ymax]} or
           {"b":[xmin,xmax], "p":[ymin,ymax]} or
           {"bmin":xmin, "bmax":xmax, "pmin":ymin, "pmax":ymax}}

    Notes:
    - Coordinates are (birth, persistence).
    - If cfg['sigma'] is not provided, sigma = sigma_frac * max(dx,dy).
    - Return vector concatenates H0 then H1 in row-major order.
    """
    def _parse_grid_size(gs):
        if isinstance(gs, (list, tuple, np.ndarray)):
            if len(gs) != 2:
                raise ValueError(f"grid_size list must have length 2, got {gs}")
            return int(gs[0]), int(gs[1])
        n = int(gs)
        return n, n

    def _edges_from_bounds(bnd, grid_size):
        nx, ny = _parse_grid_size(grid_size)
        # Several acceptable spellings/structures for bounds
        if isinstance(bnd, dict):
            if all(k in bnd for k in ('bmin','bmax','pmin','pmax')):
                xmin, xmax = float(bnd['bmin']), float(bnd['bmax'])
                ymin, ymax = float(bnd['pmin']), float(bnd['pmax'])
            else:
                xpair = bnd.get('x') or bnd.get('birth') or bnd.get('b')
                ypair = bnd.get('y') or bnd.get('pers')  or bnd.get('p')
                if xpair is None or ypair is None:
                    raise KeyError(f"bounds missing x/y (have keys: {list(bnd.keys())})")
                xmin, xmax = float(xpair[0]), float(xpair[1])
                ymin, ymax = float(ypair[0]), float(ypair[1])
        else:
            # Assume [[xmin,xmax],[ymin,ymax]]
            xmin, xmax = float(bnd[0][0]), float(bnd[0][1])
            ymin, ymax = float(bnd[1][0]), float(bnd[1][1])
        if not (np.isfinite([xmin,xmax,ymin,ymax]).all()):
            raise ValueError("Non-finite bounds in PI config")
        if not (xmax > xmin and ymax > ymin):
            raise ValueError("Invalid bounds: require xmax > xmin and ymax > ymin")
        x_edges = np.linspace(xmin, xmax, nx+1)
        y_edges = np.linspace(ymin, ymax, ny+1)
        dx = (xmax - xmin)/nx
        dy = (ymax - ymin)/ny
        return x_edges, y_edges, dx, dy

    def _get_edges(cfg_local: Dict, H: str):
        # Case A: per-homology edges
        if H in cfg_local and isinstance(cfg_local[H], dict):
            c = cfg_local[H]
            xe = c.get('x_edges') or c.get('xbins') or c.get('xgrid')
            ye = c.get('y_edges') or c.get('ybins') or c.get('ygrid')
            if xe is not None and ye is not None:
                return np.array(xe, float), np.array(ye, float), None, None
        # Case B: shared top-level edges
        xe = cfg_local.get('x_edges') or cfg_local.get('xbins') or cfg_local.get('xgrid')
        ye = cfg_local.get('y_edges') or cfg_local.get('ybins') or cfg_local.get('ygrid')
        if xe is not None and ye is not None:
            return np.array(xe, float), np.array(ye, float), None, None
        # Case C: bounds + grid_size
        if 'bounds' in cfg_local and 'grid_size' in cfg_local:
            xe, ye, dx, dy = _edges_from_bounds(cfg_local['bounds'], cfg_local['grid_size'])
            return xe, ye, dx, dy
        # Nothing matched
        raise KeyError(
            f"PI config missing edges for {H}. Expected per-H edges, top-level edges, or bounds+grid_size. Found keys: {list(cfg_local.keys())}"
        )

    def _one(H: str):
        xe, ye, dx, dy = _get_edges(cfg, H)
        Xc = 0.5*(xe[:-1] + xe[1:])
        Yc = 0.5*(ye[:-1] + ye[1:])
        Z = np.zeros((Yc.size, Xc.size), float)
        # Sigma
        if 'sigma' in cfg:
            sigma = max(float(cfg['sigma']), 1e-12)
        else:
            sigma_frac = float(cfg.get('sigma_frac', 0.75))
            if dx is None or dy is None:
                dx_est = np.mean(np.diff(xe)); dy_est = np.mean(np.diff(ye))
                sigma = max(sigma_frac * max(dx_est, dy_est), 1e-12)
            else:
                sigma = max(sigma_frac * max(dx, dy), 1e-12)
        if H not in PD or PD[H].size == 0:
            return Z
        D = PD[H]
        b = D[:,0]; d = D[:,1]
        p = np.maximum(d-b, 0.0)
        for (xi, yi, wi) in zip(b, p, p):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            dxm = (Xc - xi)[None, :]
            dym = (Yc - yi)[:, None]
            Z += np.exp(-(dxm**2 + dym**2)/(2.0*sigma**2)) * wi
        return Z

    H0 = _one('H0'); H1 = _one('H1')
    return np.concatenate([H0.ravel(), H1.ravel()], axis = 0)



# --- Persistence Landscape (PL) ---

def _pl_from_PD(PD: Dict[str, np.ndarray], cfg: Dict) -> np.ndarray:
    """
    Build a concatenated PL vector (H0 then H1), row-major.

    Supported schemas:
    A) {"t_grid": [...], "k_max": int}
    B) {"K": int| "k_max": int, "T_samples": int| "n_t": int,
        "tmin": float, "tmax": float|"inf"|null}
       - If tmax is missing or not finite, we set tmax = max death over PD.
    """
    import numpy as np
    # --- Read k_max/K
    k_max = int(cfg.get("k_max", cfg.get("K", 5)))

    # --- Build/Obtain t_grid
    if "t_grid" in cfg:
        t = np.array(cfg["t_grid"], float)
        if t.ndim != 1 or t.size == 0:
            raise ValueError("t_grid must be a 1D non-empty array")
    else:
        # Aliases for number of samples
        T = cfg.get("T_samples", cfg.get("n_t", cfg.get("nt", None)))
        if T is None:
            raise KeyError("PL config missing 't_grid' and 'T_samples'/'n_t'.")
        T = int(T)
        if T <= 1:
            raise ValueError(f"T_samples/n_t must be >1, got {T}")
        tmin = float(cfg.get("tmin", 0.0))
        tmax_val = cfg.get("tmax", None)

        # Parse possibly infinite tmax
        def _is_finite(x):
            try:
                xv = float(x)
                return np.isfinite(xv), float(xv)
            except Exception:
                return False, None

        finite, tmax_num = _is_finite(tmax_val)
        if (tmax_val in (None, "inf", "Inf", "INF")) or (not finite):
            # Derive from PD: max death among H0/H1
            deaths = []
            for H in ("H0", "H1"):
                if H in PD and PD[H].size:
                    deaths.append(np.nanmax(PD[H][:, 1]))
            if len(deaths) == 0:
                # Fallback: ensure non-degenerate interval
                tmax_use = tmin + 1.0
            else:
                tmax_use = float(np.nanmax(deaths))
                if not np.isfinite(tmax_use) or tmax_use <= tmin:
                    tmax_use = tmin + 1.0
        else:
            tmax_use = float(tmax_num)
            if not np.isfinite(tmax_use) or tmax_use <= tmin:
                tmax_use = tmin + 1.0

        t = np.linspace(tmin, tmax_use, T)

    # --- cCore landscape computation
    def lambda_vals(D):
        if D.size == 0:
            return np.zeros((k_max, t.size), float)
        b = D[:, 0]; d = D[:, 1]
        L = np.zeros((D.shape[0], t.size), float)
        for i, (bi, di) in enumerate(zip(b, d)):
            if not (np.isfinite(bi) and np.isfinite(di) and di > bi):
                continue
            mid = 0.5 * (bi + di)
            left  = (t >= bi) & (t <= mid)
            right = (t >= mid) & (t <= di)
            L[i, left]  = t[left]  - bi
            L[i, right] = di       - t[right]
        # k-th order statistics along axis=0 (k=1 → max)
        out = np.zeros((k_max, t.size), float)
        if L.shape[0] == 0:
            return out
        L_sorted = np.sort(L, axis=0)[::-1, :]
        m = min(k_max, L_sorted.shape[0])
        out[:m, :] = L_sorted[:m, :]
        return out

    H0 = lambda_vals(PD.get("H0", np.zeros((0, 2))))
    H1 = lambda_vals(PD.get("H1", np.zeros((0, 2))))
    return np.concatenate([H0.ravel(), H1.ravel()], axis = 0)


# --- Betti Curves (BC) ---

def _bc_from_PD(PD: Dict[str, np.ndarray], cfg: Dict) -> np.ndarray:
    """
    Returns a concatenated Betti-curve vector (H0 then H1) evaluated on an
    epsilon grid.

    Accepts:
      A) explicit grid:  cfg['epsilon_grid'] or cfg['e_grid']
      B) compact form:   { n_eps|n|N|samples|T_samples|n_t|E_samples,
                           emin|e_min|min|tmin,
                           emax|e_max|max|tmax (can be 'inf'/'auto'/None),
                           scale='linear'|'log' (optional),
                           include_H0, include_H1 (optional) }
      C) columns fallbacks (cfg['_columns'] from data/bc/columns.json):
         • numeric names with e=..., eps=... (extract values), or
         • index-style names like 'H0_BC_e123' → infer n and build grid using
           [emin, emax] (or PD-derived emax).

    Betti(e) = # { intervals (b,d) with b <= e < d }.
    """
    def _finite_float(x):
        try:
            xv = float(x)
            return (np.isfinite(xv), xv)
        except Exception:
            return (False, None)

    def _derive_emax_if_needed(emin_val, emax_val):
        finite, emax_num = _finite_float(emax_val)
        if (emax_val in (None, 'inf', 'Inf', 'INF', 'auto', 'AUTO')) or (not finite):
            deaths = []
            for H in ('H0','H1'):
                if H in PD and PD[H].size:
                    deaths.append(np.nanmax(PD[H][:,1]))
            if len(deaths) == 0:
                emax_use = float(emin_val) + 1.0
            else:
                emax_use = float(np.nanmax(deaths))
                if (not np.isfinite(emax_use)) or emax_use <= float(emin_val):
                    emax_use = float(emin_val) + 1.0
        else:
            emax_use = float(emax_num)
            if (not np.isfinite(emax_use)) or emax_use <= float(emin_val):
                emax_use = float(emin_val) + 1.0
        return emax_use

    # ---- 1) Explicit grid
    if 'epsilon_grid' in cfg or 'e_grid' in cfg:
        eps = np.array(cfg.get('epsilon_grid', cfg.get('e_grid')), float)
        if eps.ndim != 1 or eps.size == 0:
            raise ValueError("epsilon_grid must be a 1D non-empty array")
    else:
        # ---- 2) Compact aliases for count and bounds
        n = (cfg.get('n_eps', cfg.get('n', cfg.get('N', cfg.get('samples', cfg.get('T_samples', cfg.get('n_t', cfg.get('E_samples'))))))))
        emin = cfg.get('emin', cfg.get('e_min', cfg.get('min', cfg.get('tmin', 0.0))))
        emin = float(emin)
        emax_val = cfg.get('emax', cfg.get('e_max', cfg.get('max', cfg.get('tmax', None))))

        if n is not None:
            n = int(n)
            if n <= 1:
                raise ValueError(f"Number of epsilon samples must be >1, got {n}")
            emax_use = _derive_emax_if_needed(emin, emax_val)
            scale = str(cfg.get('scale', 'linear')).lower()
            if scale in ('log','log10') and emin > 0.0 and emax_use > emin:
                eps = np.logspace(np.log10(emin), np.log10(emax_use), n)
            else:
                eps = np.linspace(emin, emax_use, n)
        else:
            # ---- 3) Columns.json fallbacks
            cols = cfg.get('_columns', None)
            if not cols:
                raise KeyError("BC config missing 'epsilon_grid' and n_eps/... and no _columns provided for fallback.")

            # 3a) Numeric e=... in names
            rx_val = re.compile(r"(?:^|[^\w])(?:e|eps)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
            eps_vals = []
            for c in cols:
                m = rx_val.search(str(c))
                if m:
                    try:
                        eps_vals.append(float(m.group(1)))
                    except Exception:
                        pass
            if eps_vals:
                eps = np.array(sorted(set(eps_vals)), float)
                if eps.size < 2:
                    e0 = float(eps[0]) if eps.size == 1 else 0.0
                    eps = np.linspace(max(e0-1.0, 0.0), e0+1.0, 32)
            else:
                # 3b) index-style '..._e123' → infer n and build grid from [emin, emax]
                rx_idx = re.compile(r"_e(\d+)$")
                idxs = []
                for c in cols:
                    m = rx_idx.search(str(c))
                    if m:
                        try:
                            idxs.append(int(m.group(1)))
                        except Exception:
                            pass
                if not idxs:
                    raise KeyError("Could not infer epsilon grid from columns. Ensure names encode e=... or _e<idx>, or supply n_eps/emin/emax in bc/config.json.")
                n = int(max(idxs) + 1)
                if n <= 1:
                    n = 32
                emax_use = _derive_emax_if_needed(emin, emax_val)
                eps = np.linspace(emin, emax_use, n)

    # ---- Betti curve per homology
    def betti_curve(D: np.ndarray) -> np.ndarray:
        if D.size == 0:
            return np.zeros(eps.size, float)
        b = D[:,0]; d = D[:,1]
        out = np.zeros(eps.size, int)
        for i, e in enumerate(eps):
            out[i] = int(np.sum((b <= e) & (d > e)))  # Alive if b <= e < d
        return out.astype(float)

    v = []
    if cfg.get('include_H0', True):
        v.append(betti_curve(PD.get('H0', np.zeros((0,2)))))
    if cfg.get('include_H1', True):
        v.append(betti_curve(PD.get('H1', np.zeros((0,2)))))
    return np.concatenate(v, axis = 0) if v else np.zeros(0, float)


# ---------------- Core Ablation Logic ----------------

def _metrics(y, p, thresh = 0.5):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
    y = np.asarray(y, int); p = np.asarray(p, float)
    yhat = (p >= thresh).astype(int)
    uniq = np.unique(y)
    auc = float('nan') if uniq.size<2 else float(roc_auc_score(y,p))
    ap  = float('nan') if uniq.size<2 else float(average_precision_score(y,p))
    f1  = float('nan') if uniq.size<2 else float(f1_score(y,yhat))
    br  = float(brier_score_loss(y,p))
    return dict(auc = auc, ap = ap, f1 = f1, brier = br)


def _probs(pipe, X):
    if hasattr(pipe, 'predict_proba'): return pipe.predict_proba(X)[:,1]
    s = pipe.decision_function(X); smin, smax = float(np.min(s)), float(np.max(s))
    return (s - smin) / (smax - smin + 1e-12)


def _round_int(x, lo):
    return int(max(lo, int(np.rint(x))))


def _perturb_params(m_star: int, tau_star: int, scales_m: List[float], scales_tau: List[float]):
    combos = []
    for sm in scales_m:
        for st in scales_tau:
            m_p = _round_int(sm*m_star, 2)
            t_p = _round_int(st*tau_star, 1)
            combos.append((sm, st, m_p, t_p))
    # Remove exact (1.0,1.0) duplicate if desired later
    return combos


def _vectorize_PD_set(PD: Dict[str,np.ndarray], cfgs: Dict[str,Dict]) -> Dict[str,np.ndarray]:
    return {
        'pi': _pi_from_PD(PD, cfgs['pi']),
        'pl': _pl_from_PD(PD, cfgs['pl']),
        'bc': _bc_from_PD(PD, cfgs['bc']),
    }


def run_ablation(project_root: str = '.', scales_m = (0.9,1.0,1.1), scales_tau = (0.9,1.0,1.1),
                 max_points: int = 2000, seed: int = 2025):
    # Paths & configs for vectorizers
    vec_dirs = {
        'pi': os.path.join(project_root, 'data', 'pi'),
        'pl': os.path.join(project_root, 'data', 'pl'),
        'bc': os.path.join(project_root, 'data', 'bc'),
    }
    cfgs = {
        'pi': _load_json(os.path.join(vec_dirs['pi'], 'config.json')),
        'pl': _load_json(os.path.join(vec_dirs['pl'], 'config.json')),
        'bc': _load_json(os.path.join(vec_dirs['bc'], 'config.json')),
    }

    cols_path = os.path.join(vec_dirs['bc'], 'columns.json')
    if os.path.exists(cols_path):
        try:
            with open(cols_path, 'r') as f:
                cfgs['bc']['_columns'] = json.load(f)
        except Exception:
            pass

    # Load mapping id -> (m*, tau*) and raw series path
    params_path = os.path.join(project_root, 'data', 'embeddings', 'params.jsonl')
    params = {}
    with open(params_path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            params[rec['id']] = rec
    meta = {}
    with open(os.path.join(project_root, 'data', 'synthetic', 'metadata.csv'), 'r') as f:
        rd = csv.DictReader(f)
        for r in rd:
            meta[r['id']] = r

    # Load TEST split (so sensitivity is measured on the same held-out set)
    # We also reuse labels and SNR for optional stratification later.
    fs_pi = load_feature_set(vec_dirs['pi'])  # any feature dir has aligned index.csv
    test_ids = [m['id'] for m in fs_pi.test.meta]
    y_test  = fs_pi.test.y

    # Load fitted models (train+val) from Step 3.2
    models = {}
    for feat in ('pi','pl','bc','baseline'):
        models[feat] = {}
        model_dir = os.path.join(project_root, 'models', feat)
        if not os.path.isdir(model_dir):
            continue
        for mdl in ('logreg','svm_rbf','rf'):
            pth = os.path.join(model_dir, f'{mdl}.joblib')
            if os.path.exists(pth):
                models[feat][mdl] = joblib.load(pth)

    # Baseline test features (for delta computation)
    base_feats = {
        'pi': fs_pi.test.X,
        'pl': load_feature_set(vec_dirs['pl']).test.X,
        'bc': load_feature_set(vec_dirs['bc']).test.X,
        'baseline': load_feature_set(os.path.join(project_root,'data','baseline')).test.X,
    }

    # Compute baseline metrics per (feat, model)
    baseline_metrics = {}
    for feat, mdl_dict in models.items():
        for mdl, pipe in mdl_dict.items():
            p = _probs(pipe, base_feats[feat])
            baseline_metrics[(feat,mdl)] = _metrics(y_test, p)

    # Prepare outputs
    out_dir = os.path.join(project_root, 'results', 'robust', 'ablation_embed')
    plot_dir = os.path.join(out_dir, 'plots')
    summ_dir = os.path.join(out_dir, 'summary')
    os.makedirs(plot_dir, exist_ok = True); os.makedirs(summ_dir, exist_ok = True)

    rng = np.random.default_rng(seed)

    # Build perturbation grid (including 1.0 for reference rows)
    grid = _perturb_params(10, 10, list(scales_m), list(scales_tau))  # Dummy to enumerate sizes only
    # We'll iterate per sample using real (m*,tau*)

    # Accumulate per (feat,model) metric tables
    tables = { (f,m): [] for f in models for m in models[f] }

    # Loop over test samples and perturbations
    for uid in test_ids:
        if uid not in params or uid not in meta:
            continue
        rec = params[uid]
        m_star = int(rec['m_star']); tau_star = int(rec['tau_star'])
        # Scales per this sample
        combos = _perturb_params(m_star, tau_star, list(scales_m), list(scales_tau))
        # Load raw series and standardize
        x = np.load(meta[uid]['path'])
        x_std = standardize(x)
        # For each (sm,st) → embed → PD → vectorize
        pd_cache = {}
        for (sm, st, m_p, t_p) in combos:
            # Embed
            X = takens_embed(x_std, m_p, t_p)
            # PD up to H1 using the same cap on points
            PD, _ = vr_persistence_diagrams(X, maxdim = 1, max_points = max_points, seed = seed)
            pd_cache[(sm,st)] = PD
        # Vectorize once per PD and evaluate all models
        for (sm, st) in {(sm,st) for (sm,st,_,_) in combos}:
            PD = pd_cache[(sm,st)]
            feats = _vectorize_PD_set(PD, cfgs)
            # evaluate for each (feat, model)
            for feat, mdl_dict in models.items():
                if feat not in feats and feat != 'baseline':
                    continue
                Xte = feats.get(feat, None)
                if feat == 'baseline':
                    # Baseline features do not change with (m, tau); reuse
                    Xte = base_feats['baseline'][[test_ids.index(uid)]]  # Single-row slice
                for mdl, pipe in mdl_dict.items():
                    p = _probs(pipe, Xte if Xte.ndim == 2 else Xte.reshape(1,-1))
                    # Accumulate per-sample metrics for later mean aggregation
                    tables[(feat,mdl)].append({
                        'id': uid, 'sm': float(sm), 'st': float(st),
                        **_metrics(np.array([y_test[test_ids.index(uid)]]), np.array([p if np.isscalar(p) else p[0]]))
                    })

    # Aggregate by (sm, st)
    summaries = {}
    for key, rows in tables.items():
        if not rows: continue
        df = pd.DataFrame(rows)
        grp = df.groupby(['sm','st']).agg({
            'auc':'mean','ap':'mean','brier':'mean','f1':'mean'
        }).reset_index()
        feat, mdl = key
        # Compute deltas vs baseline metrics computed over full test set
        base = baseline_metrics[(feat,mdl)]
        grp['delta_auc']   = grp['auc']   - base['auc']
        grp['delta_ap']    = grp['ap']    - base['ap']
        grp['delta_brier'] = grp['brier'] - base['brier']
        grp['delta_f1']    = grp['f1']    - base['f1']
        # Write CSVs
        base_name = f'{feat}_{mdl}'
        os.makedirs(out_dir, exist_ok = True)
        grp.to_csv(os.path.join(out_dir, f'{base_name}_ablation_deltas.csv'), index = False)
        # Pretty heatmap for ΔAUC
        P = grp.pivot(index = 'sm', columns = 'st', values = 'delta_auc')
        plt.figure(figsize = (4.5,3.6))
        im = plt.imshow(P.values, origin = 'lower', aspect = 'auto')
        plt.xticks(np.arange(P.shape[1]), P.columns)
        plt.yticks(np.arange(P.shape[0]), P.index)
        plt.xlabel('$s_\\tau$ ($\\tau$ scale)'); plt.ylabel('$s_m$ ($m$ scale)')
        cbar = plt.colorbar(im); cbar.set_label('$\Delta$AUC (Perturbed $-$ Baseline)')
        plt.title(f'{feat.upper()} / {mdl}')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{base_name}_delta_auc_heat.png'), dpi = 300)
        plt.close()
        # Summary stats
        df_nt = grp[(grp['sm'] != 1.0) | (grp['st'] != 1.0)]
        summaries[base_name] = {
            'max_drop_auc': float(-(df_nt['delta_auc'].min() if len(df_nt) else 0.0)),
            'mean_abs_drop_auc': float(df_nt['delta_auc'].abs().mean() if len(df_nt) else 0.0),
            'max_drop_ap': float(-(df_nt['delta_ap'].min() if len(df_nt) else 0.0)),
            'mean_abs_drop_ap': float(df_nt['delta_ap'].abs().mean() if len(df_nt) else 0.0),
            'max_rise_brier': float(df_nt['delta_brier'].max() if len(df_nt) else 0.0),
            'mean_abs_delta_brier': float(df_nt['delta_brier'].abs().mean() if len(df_nt) else 0.0),
            'max_drop_f1': float(-(df_nt['delta_f1'].min() if len(df_nt) else 0.0)),
            'mean_abs_drop_f1': float(df_nt['delta_f1'].abs().mean() if len(df_nt) else 0.0),
        }
    with open(os.path.join(summ_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(summaries, f, indent = 2)
    print('[ABLATION] Wrote results to', out_dir)