import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

try:
    import smplotlib  # noqa: F401
except Exception:
    pass


def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def _parse_grid_from_config(cfg):
    """Return (bx_edges, p_edges) from PI config.

    Supports the compact form used in this project, e.g.:
    {"grid_size": E_samples or [Nx, Ny],
     "bounds": {"bmin": ..., "bmax": ..., "pmin": ..., "pmax": ...}, ...}
    """
    # Grid_size can be scalar or 2-list
    gs = cfg.get('grid_size', cfg.get('E_samples'))
    if isinstance(gs, (list, tuple, np.ndarray)):
        if len(gs) != 2:
            raise ValueError(f"Grid_size list must be length 2, got {gs}")
        Nx, Ny = int(gs[0]), int(gs[1])
    else:
        Nx = Ny = int(gs)

    bnd = cfg.get('bounds', {})
    if {'bmin','bmax','pmin','pmax'} <= set(bnd.keys()):
        bmin, bmax = float(bnd['bmin']), float(bnd['bmax'])
        pmin, pmax = float(bnd['pmin']), float(bnd['pmax'])
    else:
        # Fallback to generic naming
        bx = bnd.get('x') or bnd.get('birth') or bnd.get('b')
        py = bnd.get('y') or bnd.get('pers') or bnd.get('p')
        if bx is None or py is None:
            raise KeyError(f"Bounds missing birth/pers keys; have {list(bnd.keys())}")
        bmin, bmax = float(bx[0]), float(bx[1])
        pmin, pmax = float(py[0]), float(py[1])

    if not (bmax > bmin and pmax > pmin):
        raise ValueError("Invalid PI bounds: require bmax > bmin, pmax > pmin.")

    bx_edges = np.linspace(bmin, bmax, Nx + 1)
    p_edges = np.linspace(pmin, pmax, Ny + 1)
    return bx_edges, p_edges


def _extract_logreg_weights(pipe):
    """Extract (w, b) from a sklearn pipeline containing LogisticRegression."""
    # If it's a Pipeline, last step is usually the classifier.
    clf = None
    if hasattr(pipe, 'named_steps'):
        for name, step in pipe.named_steps.items():
            from sklearn.linear_model import LogisticRegression
            if isinstance(step, LogisticRegression):
                clf = step
                break
        if clf is None:
            # Fallback: assume last step is clf
            clf = list(pipe.named_steps.values())[-1]
    else:
        clf = pipe

    if not hasattr(clf, 'coef_'):
        raise TypeError("Classifier does not expose coef_; expected LogisticRegression or similar.")
    w = np.asarray(clf.coef_).reshape(-1)
    b = float(getattr(clf, 'intercept_', [0.0])[0])
    return w, b


def build_pi_weight_maps(project_root = '.'):
    """Build H0/H1 weight maps for the PI logistic-regression model.

    Outputs:
      - results/interpret/pi_weight_maps/pi_weight_summary.csv
      - results/interpret/pi_weight_maps/pi_weights_H0.png
      - results/interpret/pi_weight_maps/pi_weights_H1.png
    """
    data_pi = os.path.join(project_root, 'data', 'pi')
    model_dir = os.path.join(project_root, 'models', 'pi')
    out_dir   = os.path.join(project_root, 'results', 'interpret', 'pi_weight_maps')
    os.makedirs(out_dir, exist_ok=True)

    cfg = _load_json(os.path.join(data_pi, 'config.json'))
    cols = _load_json(os.path.join(data_pi, 'columns.json'))

    # Infer grid
    bx_edges, p_edges = _parse_grid_from_config(cfg)
    Nx = len(bx_edges) - 1
    Ny = len(p_edges) - 1

    # Load model
    model_path = os.path.join(model_dir, 'logreg.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PI logistic-regression model not found at {model_path}")
    pipe = joblib.load(model_path)

    w, b = _extract_logreg_weights(pipe)

    # Check dimensionality consistency
    D = w.size
    expected_D = 2 * Nx * Ny
    if D != expected_D:
        raise ValueError(
            f"Mismatch between PI grid (2*{Nx}*{Ny} = {expected_D}) and weight dim {D}. "
            f"Check that config.json and the trained model use the same PI resolution."
        )

    # Split into H0/H1 and reshape
    w_H0 = w[:Nx*Ny].reshape(Ny, Nx)  # (Ny, Nx)
    w_H1 = w[Nx*Ny:].reshape(Ny, Nx)

    # Build birth/persistence centers
    bx_centers = 0.5 * (bx_edges[:-1] + bx_edges[1:])
    p_centers  = 0.5 * (p_edges[:-1] + p_edges[1:])

    # Export tabular summary (each row = one pixel)
    rows = []
    for H, W in [('H0', w_H0), ('H1', w_H1)]:
        for j in range(Ny):
            for i in range(Nx):
                rows.append({
                    'homology': H,
                    'birth_min': bx_edges[i],
                    'birth_max': bx_edges[i+1],
                    'pers_min':  p_edges[j],
                    'pers_max':  p_edges[j+1],
                    'birth_center': bx_centers[i],
                    'pers_center':  p_centers[j],
                    'weight': float(W[j, i]),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'pi_weight_summary.csv'), index = False)

    # Plot heatmaps
    extent = [bx_edges[0], bx_edges[-1], p_edges[0], p_edges[-1]]

    def _plot_heat(W, H_label, fname):
        plt.figure(figsize = (5,4))
        im = plt.imshow(W, origin = 'lower', aspect = 'auto', extent = extent)
        plt.colorbar(im, label = 'Weight on log-odds')
        plt.xlabel('Birth')
        plt.ylabel('Persistence')
        plt.title(f'PI Weight Map ({H_label})')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi  =300)
        plt.close()

    _plot_heat(w_H0, 'H0', 'pi_weights_H0.png')
    _plot_heat(w_H1, 'H1', 'pi_weights_H1.png')

    print('\n[INTERPRET] Wrote PI weight maps and CSV to', out_dir)