import os, json, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import smplotlib
from math import pi

def load_metric_files(project_root='.'):
    base = os.path.join(project_root, 'results', 'ml', 'metrics')
    # Fallback: gather all *_val_metrics.json under models/
    paths = []
    for root, _, files in os.walk(project_root):
        for f in files:
            if f.endswith('_val_metrics.json') or f.endswith('_test_metrics.json'):
                paths.append(os.path.join(root, f))
    rows = []
    for p in paths:
        try:
            with open(p,'r') as f: data = json.load(f)
            feat, model = p.split(os.sep)[-2], os.path.basename(p).split('_')[0]
            rows.append(dict(feature = feat, model = model, **data))
        except Exception:
            pass
    return pd.DataFrame(rows)

def aggregate_scores(df):
    metrics = ['auc','ap','f1','brier']
    agg = (df.groupby(['feature','model'])[metrics]
             .mean().reset_index())
    # Compute normalized mean per feature
    norm_scores = []
    for m in metrics:
        for model, sub in agg.groupby('model'):
            vmax = sub[m].max() if m != 'brier' else sub[m].min()
            for _, r in sub.iterrows():
                s = (r[m]/vmax) if m != 'brier' else (vmax/r[m])
                norm_scores.append(dict(feature = r['feature'], model = model, metric = m, norm = s))
    norm = pd.DataFrame(norm_scores)
    meanS = norm.groupby('feature')['norm'].mean().reset_index().rename(columns = {'norm':'S_norm'})
    return agg, meanS

def radar_plot(meanS, outpath):
    categories = list(meanS['feature'])
    values = meanS['S_norm'].values.tolist()
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint = False).tolist()
    values += values[:1]; angles += angles[:1]
    plt.figure(figsize = (5,5))
    ax = plt.subplot(111, polar = True)
    ax.plot(angles, values, 'o-', linewidth = 2)
    ax.fill(angles, values, alpha = 0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0,1.05)
    plt.title('Normalized Mean Performance Score')
    plt.tight_layout()
    plt.savefig(outpath, dpi = 300)
    plt.close()

def bar_plot(agg, outpath):
    metrics = ['auc','ap','f1','brier']
    fig, axs = plt.subplots(1,4, figsize = (12,4))
    for i,m in enumerate(metrics):
        order = agg.groupby('feature')[m].mean().sort_values(ascending = (m == 'brier')).index
        axs[i].bar(order, agg.groupby('feature')[m].mean().loc[order])
        axs[i].set_title(m.upper())
        axs[i].grid(True, axis = 'y', ls = ':')
    plt.tight_layout()
    plt.savefig(outpath, dpi = 300)
    plt.close()

def run_feature_comparison(project_root = '.'):
    df = load_metric_files(project_root)
    agg, meanS = aggregate_scores(df)
    out_dir = os.path.join(project_root,'results','robust','feature_comparison')
    os.makedirs(out_dir, exist_ok = True)
    agg.to_csv(os.path.join(out_dir,'summary_metrics.csv'), index=False)
    radar_plot(meanS, os.path.join(out_dir,'radar_mean_scores.png'))
    bar_plot(agg, os.path.join(out_dir,'bar_auc_ap_f1_brier.png'))
    print(f'[feature_comparison] wrote results to {out_dir}')
