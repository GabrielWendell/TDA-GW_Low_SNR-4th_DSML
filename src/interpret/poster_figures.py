import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

# Optional: apply your lightweight style module if available
try:
    import smplotlib  # noqa: F401
except Exception:
    pass

from .pi_weight_maps import build_pi_weight_maps


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok = True)


def _load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


# ---------------------------------------------------------
# F1 — Pipeline schematic (minimal matplotlib schematic)
# ---------------------------------------------------------

def fig_pipeline_schematic(out_path: str):
    """Create a simple block-diagram schematic of the pipeline.

    Blocks: Raw time series -> Takens embedding -> VR PD -> (PI, PL, BC) -> ML classifier ->
    Robustness (SNR sweep, ablation) -> Interpretation (weight maps).
    """
    plt.figure(figsize = (10, 2.5))
    ax = plt.gca()
    ax.axis('off')

    # Helper to draw boxes
    def box(x, y, w, h, text):
        rect = plt.Rectangle((x, y), w, h, fill = False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha = 'center', va = 'center', wrap = True)

    # Coordinates
    y = 0.25
    w, h = 1.5, 0.8
    xs = [0.2, 2.1, 4.0, 6.0, 8.0]
    labels = [
        'Raw\nGW-like time series',
        'Takens\nembedding',
        'VR persistence\ndiagrams',
        'Topological\nfeatures\n(PI, PL, BC)',
        'ML +\nrobustness\nanalysis',
    ]

    for x, lab in zip(xs, labels):
        box(x, y, w, h, lab)

    # Arrows
    for x in xs[:-1]:
        ax.annotate('', xy = (x + w, y + h/2), xytext = (x + w + 0.2, y + h/2),
                    arrowprops = dict(arrowstyle = '->'))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi = 300)
    plt.close()


# ---------------------------------------------------------
# F2 — Example PDs by SNR
# ---------------------------------------------------------

def fig_pd_examples(project_root: str, out_path: str, n_examples_per_snr: int = 1):
    """Assemble example PDs (H0/H1) for a few SNRs into a single panel.

    Assumes that Step 2.1 stored PD visualizations or at least PD npz files
    under results/tda/ (adjust paths as per your implementation).
    """
    # This function is a template; exact loading depends on how PDs are stored.
    # We will assume PD arrays in npz with keys 'H0', 'H1' and an accompanying
    # Metadata CSV listing snr and file path.
    tda_root = os.path.join(project_root, 'results', 'tda')
    meta_path = os.path.join(tda_root, 'pd_index.csv')
    if not os.path.exists(meta_path):
        raise FileNotFoundError('Expected PD index at results/tda/pd_index.csv.')

    meta = pd.read_csv(meta_path)
    snr_values = sorted(meta['snr'].unique())
    # Pick e.g. 3 representative SNRs (low, mid, high)
    if len(snr_values) >= 3:
        snr_sel = [snr_values[0], snr_values[len(snr_values)//2], snr_values[-1]]
    else:
        snr_sel = snr_values

    n_rows = len(snr_sel)
    n_cols = 2  # H0, H1
    fig, axes = plt.subplots(n_rows, n_cols, figsize = (6, 2.5*n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    for i, snr in enumerate(snr_sel):
        row = meta[meta['snr'] == snr].iloc[0]
        npz_path = os.path.join(project_root, row['pd_path'])
        arr = np.load(npz_path, allow_pickle = True)
        for j, H in enumerate(['H0', 'H1']):
            ax = axes[i, j]
            ax.scatter(arr[H][:, 0], arr[H][:, 1], s=8)
            ax.plot([arr[H][:, 0].min(), arr[H][:, 1].max()],
                    [arr[H][:, 0].min(), arr[H][:, 1].max()], 'k--', linewidth = 0.5)
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.set_title(f'SNR = {snr}, {H}')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------
# F3 — PI / PL / BC example panel
# ---------------------------------------------------------

def fig_pi_pl_bc_examples(project_root: str, out_path: str):
    """Create a 3-panel figure with PI, PL, and BC for a single representative signal.

    Assumes Step 2.2 stored example visualizations or feature arrays with configs.
    """
    data_root = os.path.join(project_root, 'data')

    # Load PI config to reconstruct grid for a single example feature vector
    pi_cfg = _load_json(os.path.join(data_root, 'pi', 'config.json'))
    pi_feats = np.load(os.path.join(data_root, 'pi', 'features_test.npy'))
    x_pi = pi_feats[0]  # 1st example

    # Infer PI grid dim
    # (assume 2 * Nx * Ny = D)
    D = x_pi.size
    gs = pi_cfg.get('grid_size', pi_cfg.get('E_samples'))
    if isinstance(gs, (list, tuple, np.ndarray)):
        Nx, Ny = int(gs[0]), int(gs[1])
    else:
        Nx = Ny = int(gs)
    assert 2*Nx*Ny == D, 'PI feature dim mismatch; check config vs features.'
    I0 = x_pi[:Nx*Ny].reshape(Ny, Nx)
    I1 = x_pi[Nx*Ny:].reshape(Ny, Nx)

    # PL
    pl_cfg = _load_json(os.path.join(data_root, 'pl', 'config.json'))
    pl_feats = np.load(os.path.join(data_root, 'pl', 'features_test.npy'))
    x_pl = pl_feats[0]
    # Assume shape: 2 * K * T_samples
    K = int(pl_cfg.get('K', pl_cfg.get('k_max', 5)))
    T = int(pl_cfg.get('T_samples', pl_cfg.get('n_t', 256)))
    assert 2*K*T == x_pl.size
    L0 = x_pl[:K*T].reshape(K, T)
    L1 = x_pl[K*T:].reshape(K, T)

    # BC
    bc_cfg = _load_json(os.path.join(data_root, 'bc', 'config.json'))
    bc_feats = np.load(os.path.join(data_root, 'bc', 'features_test.npy'))
    x_bc = bc_feats[0]
    # Infer epsilon grid size from half-length (H0+H1)
    T_eps = x_bc.size // 2
    B0 = x_bc[:T_eps]
    B1 = x_bc[T_eps:]

    fig, axes = plt.subplots(1, 3, figsize = (10, 3))

    # PI: show H0/H1 as two subplots stacked in first column
    ax0 = axes[0]
    im0 = ax0.imshow(I0, origin = 'lower', aspect = 'auto')
    plt.colorbar(im0, ax=ax0, fraction = 0.046, pad = 0.04)
    ax0.set_title('PI (H0/H1 example)')

    # PL: plot first few landscapes for H1 to show structure
    ax1 = axes[1]
    t_grid = np.linspace(0.0, 1.0, T)
    for k in range(min(K, 3)):
        ax1.plot(t_grid, L1[k], label = f'k = {k+1}')
    ax1.set_title('Persistence landscapes (H1)')
    ax1.set_xlabel('$t$ (normalized)')
    ax1.set_ylabel('$\lambda_k(t)$')
    ax1.legend()

    # BC: plot H0 and H1
    ax2 = axes[2]
    eps = np.linspace(0.0, 1.0, T_eps)
    ax2.plot(eps, B0, label = 'H0')
    ax2.plot(eps, B1, label = 'H1')
    ax2.set_title('Betti curves')
    ax2.set_xlabel('$\\varepsilon$ (normalized)')
    ax2.set_ylabel('$\\beta(\\varepsilon)$')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi = 300)
    plt.close()


# ---------------------------------------------------------
# F4 — ROC/PR panel for best models
# ---------------------------------------------------------

def fig_roc_pr_panel(project_root: str, out_path: str):
    """Recompute ROC and PR curves for the best-performing model per feature type.

    Uses trained models under models/<feat>/ and test features under data/<feat>.
    """
    feats = ['pi', 'pl', 'bc', 'baseline']
    colors = ['C0', 'C1', 'C2', 'C3']

    plt.figure(figsize = (10,4))

    # Subplot 1: ROC
    ax1 = plt.subplot(1,2,1)
    # Subplot 2: PR
    ax2 = plt.subplot(1,2,2)

    for feat, col in zip(feats, colors):
        # Load features and labels
        X_test = np.load(os.path.join(project_root, 'data', feat, 'features_test.npy'))
        y_test = np.load(os.path.join(project_root, 'data', feat, 'labels_test.npy'))

        # Load best model (here we assume logistic regression stored as logreg.joblib)
        model_path = os.path.join(project_root, 'models', feat, 'logreg.joblib')
        if not os.path.exists(model_path):
            continue
        pipe = joblib.load(model_path)
        scores = pipe.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, scores)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        auc_val = auc(fpr, tpr)
        ap_val = average_precision_score(y_test, scores)

        ax1.plot(fpr, tpr, label = f'{feat.upper()} (AUC = {auc_val:.2f})', color = col)
        ax2.plot(rec, prec, label = f'{feat.upper()} (AP = {ap_val:.2f})', color = col)

    ax1.plot([0,1], [0,1], 'k--', linewidth=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC curves')
    ax1.legend()

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall curves')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi = 300)
    plt.close()


# --------------------------------------------------------------------------------
# F5/F6/F7/F8 — reuse existing robustness, profile, and interpretability figures
# --------------------------------------------------------------------------------

def collect_existing_figures(project_root: str, poster_root: str):
    """Copy/rename figures generated in previous phases into poster folder.

    F5: feature comparison (from Step 4.3)
        - results/robust/feature_comparison/radar_mean_scores.png
        - results/robust/feature_comparison/bar_auc_ap_f1_brier.png
    F6: robustness (SNR sweep + ablation)
        - results/robust/snr_sweep/metrics_by_snr.png (if available)
        - results/robust/ablation_embed/plots/*delta_auc_heat.png
    F7: computational profile (from Step 4.4)
        - results/robust/profile/runtime_components.png
    F8: PI interpretability maps (from Step 5.1)
        - results/interpret/pi_weight_maps/pi_weights_H0.png
        - results/interpret/pi_weight_maps/pi_weights_H1.png
    """
    import shutil

    _ensure_dir(poster_root)

    def safe_copy(src, dst_name):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(poster_root, dst_name))

    # F5
    safe_copy(os.path.join(project_root, 'results', 'robust', 'feature_comparison', 'radar_mean_scores.png'),
              'F5_feature_radar.png')
    safe_copy(os.path.join(project_root, 'results', 'robust', 'feature_comparison', 'bar_auc_ap_f1_brier.png'),
              'F5_feature_bars.png')

    # F6 (robustness)
    safe_copy(os.path.join(project_root, 'results', 'robust', 'snr_sweep', 'metrics_by_snr.png'),
              'F6_snr_sweep.png')
    # Collect all ablation heatmaps into a subfolder
    ablation_dir = os.path.join(project_root, 'results', 'robust', 'ablation_embed', 'plots')
    if os.path.isdir(ablation_dir):
        for f in os.listdir(ablation_dir):
            if f.endswith('.png'):
                safe_copy(os.path.join(ablation_dir, f), f'F6_ablation_{f}')

    # F7 (profile)
    safe_copy(os.path.join(project_root, 'results', 'robust', 'profile', 'runtime_components.png'),
              'F7_profile_runtime.png')

    # F8 (PI weight maps)
    safe_copy(os.path.join(project_root, 'results', 'interpret', 'pi_weight_maps', 'pi_weights_H0.png'),
              'F8_pi_weights_H0.png')
    safe_copy(os.path.join(project_root, 'results', 'interpret', 'pi_weight_maps', 'pi_weights_H1.png'),
              'F8_pi_weights_H1.png')


# ---------------------------------------------------------
# Captions file
# ---------------------------------------------------------

def write_captions_md(poster_root: str):
    """Create a captions.md file with concise descriptions for each figure."""
    lines = []
    lines.append("# Poster Figure Captions\n")
    lines.append("**F1 — Pipeline schematic.** From raw GW-like time series to Takens embedding, VR persistence diagrams, topological feature maps (PI, PL, BC), and machine-learning classifiers, followed by robustness and interpretability analysis.\n")
    lines.append("**F2 — Example persistence diagrams by SNR.** H0 and H1 persistence diagrams for low, medium, and high SNR signals, illustrating the degradation of topological signal structure as noise increases.\n")
    lines.append("**F3 — Topological feature representations.** Example persistence image (PI), persistence landscapes (PL), and Betti curves (BC) for a single GW-like signal, showing how PD information is embedded into fixed-length feature vectors.\n")
    lines.append("**F4 — ROC and PR curves.** Receiver Operating Characteristic (ROC) and Precision–Recall (PR) curves for the best logistic-regression models using PI, PL, BC, and baseline features. AUC and AP values are indicated in the legend.\n")
    lines.append("**F5 — Feature comparison summary.** Radar and bar plots summarizing AUC, AP, F1, and Brier score across PI, PL, BC, and baseline features, averaged over classifiers.\n")
    lines.append("**F6 — Robustness analysis.** Metric degradation as a function of SNR and sensitivity of classifiers to ±10% perturbations in embedding parameters (m, τ), summarized via heatmaps of ΔAUC.\n")
    lines.append("**F7 — Computational profile.** Per-component runtime (Takens embedding, VR PD, PI/PL/BC vectorization) with 95% confidence intervals, highlighting the computational bottlenecks of the TDA pipeline.\n")
    lines.append("**F8 — PI weight maps.** Logistic-regression weight maps over the persistence-image birth–persistence grid for H0 and H1. Red regions indicate PD regions that increase the probability of a GW signal; blue regions decrease it.\n")

    with open(os.path.join(poster_root, 'captions.md'), 'w', encoding = 'utf-8') as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------

def run_poster_figures(project_root: str = '.'):
    poster_root = os.path.join(project_root, 'results', 'poster')
    _ensure_dir(poster_root)

    # F1: pipeline schematic
    fig_pipeline_schematic(os.path.join(poster_root, 'F1_pipeline_schematic.png'))

    # F2: PD examples
    try:
        fig_pd_examples(project_root, os.path.join(poster_root, 'F2_pd_examples.png'))
    except FileNotFoundError:
        print('\n[POSTER] Skipping PD examples (index or files not found).')

    # F3: PI/PL/BC example panel
    try:
        fig_pi_pl_bc_examples(project_root, os.path.join(poster_root, 'F3_pi_pl_bc_examples.png'))
    except Exception as e:
        print('\n[POSTER] Skipping PI/PL/BC example panel:', e)

    # F4: ROC/PR panel
    try:
        fig_roc_pr_panel(project_root, os.path.join(poster_root, 'F4_roc_pr_panel.png'))
    except Exception as e:
        print('\n[POSTER] Skipping ROC/PR panel:', e)

    # F8: ensure PI weight maps exist (Step 5.1)
    try:
        build_pi_weight_maps(project_root)
    except Exception as e:
        print('\n[POSTER] PI weight maps build failed or already done:', e)

    # F5–F8: collect existing figures
    collect_existing_figures(project_root, poster_root)

    # Captions
    write_captions_md(poster_root)

    print('\n[POSTER] Poster-ready figures and captions collected under', poster_root)