# src/ml/viz.py
import os, json, joblib, numpy as np
import matplotlib.pyplot as plt
import smplotlib
from typing import Dict, List, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from .io import load_feature_set

# --- Utilities ---

def _probs(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:,1]
    s = pipe.decision_function(X)
    smin, smax = float(np.min(s)), float(np.max(s))
    return (s - smin) / (smax - smin + 1e-12)


def _stratified_bootstrap_idx(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y, int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    i_pos = rng.choice(pos, size = pos.size, replace = True) if pos.size else np.array([], dtype = int)
    i_neg = rng.choice(neg, size = neg.size, replace = True) if neg.size else np.array([], dtype = int)
    return np.concatenate([i_pos, i_neg])


def _interp_curve(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    # Assumes x is monotone increasing; if needed, enforce monotonicity
    x = np.asarray(x, float); y = np.asarray(y, float)
    # Deduplicate x
    uniq, idx = np.unique(x, return_index = True)
    return np.interp(x_grid, uniq, y[idx], left = y[idx][0], right = y[idx][-1])


# --- Main Plotting Functions ---

def roc_pr_with_ci(pipe, X, y, B: int = 500, seed: int = 2025,
                   out_prefix: str = None, show_ci: bool = True) -> Dict:
    """Compute ROC & PR on (X,y) and bootstrap 95% CIs; save PNGs if out_prefix given."""
    y = np.asarray(y, int)
    p = _probs(pipe, X)

    # Base curves
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    auc_val = float(auc(fpr, tpr))
    ap_val  = float(average_precision_score(y, p))

    res = {"auc": auc_val, "ap": ap_val}

    if show_ci:
        rng = np.random.default_rng(seed)
        # Common grids
        fpr_grid = np.linspace(0, 1, 200)
        rec_grid = np.linspace(0, 1, 200)
        roc_samples = []
        pr_samples  = []
        auc_s, ap_s = [], []
        for b in range(B):
            idx = _stratified_bootstrap_idx(y, rng)
            yb, Xb = y[idx], X[idx]
            pb = _probs(pipe, Xb)
            fpr_b, tpr_b, _ = roc_curve(yb, pb)
            prec_b, rec_b, _ = precision_recall_curve(yb, pb)
            auc_s.append(auc(fpr_b, tpr_b))
            ap_s.append(average_precision_score(yb, pb))
            roc_samples.append(_interp_curve(fpr_b, tpr_b, fpr_grid))
            pr_samples.append(_interp_curve(rec_b, prec_b, rec_grid))
        roc_arr = np.vstack(roc_samples)
        pr_arr  = np.vstack(pr_samples)
        # Percentile intervals
        lo = 2.5; hi = 97.5
        roc_lo, roc_hi = np.percentile(roc_arr, [lo, hi], axis = 0)
        pr_lo,  pr_hi  = np.percentile(pr_arr,  [lo, hi], axis = 0)
        auc_ci = list(np.percentile(auc_s, [lo, hi]))
        ap_ci  = list(np.percentile(ap_s,  [lo, hi]))
        res.update({"auc_ci": auc_ci, "ap_ci": ap_ci})
    else:
        fpr_grid, rec_grid = fpr, rec
        roc_lo = roc_hi = pr_lo = pr_hi = None

    # Plotting
    if out_prefix is not None:
        os.makedirs(os.path.dirname(out_prefix), exist_ok = True)
        # ROC
        plt.figure(figsize = (5.0, 4.0))
        plt.plot(fpr, tpr, label = f"ROC AUC = {auc_val:.3f}")
        plt.plot([0,1],[0,1], "k--", lw=1)
        if show_ci:
            plt.fill_between(fpr_grid, roc_lo, roc_hi, alpha = 0.3, label = "95% CI")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(out_prefix + "_roc.png", dpi = 300)
        plt.close()
        # PR
        plt.figure(figsize = (5.0, 4.0))
        plt.plot(rec, prec, label = f"AP = {ap_val:.3f}")
        if show_ci:
            plt.fill_between(rec_grid, pr_lo, pr_hi, alpha = 0.3, label = "95% CI")
        # Baseline precision (prevalence)
        pi = float(np.mean(y))
        plt.hlines(pi, 0, 1, linestyles = "dashed", colors = "k", linewidth = 1, label = f"Prevalence = {pi:.2f}")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.legend(loc = "lower left"); plt.tight_layout()
        plt.savefig(out_prefix + "_pr.png", dpi = 300)
        plt.close()

    return res


def make_all_plots(project_root: str = ".", 
                   features = ("pi","pl","bc","baseline"), models = ("logreg","svm_rbf","rf"),
                    B: int = 500, seed: int = 2025) -> None:
    out_root = os.path.join(project_root, "results", "plots")
    os.makedirs(out_root, exist_ok = True)

    leaderboard = []

    for feat in features:
        fs = load_feature_set(os.path.join(project_root, "data", feat if feat != "baseline" else "baseline"))
        Xte, yte = fs.test.X, fs.test.y
        for mdl in models:
            model_path = os.path.join(project_root, "models", feat, f"{mdl}.joblib")
            if not os.path.exists(model_path):
                print(f"[WARN] Missing model: {model_path}")
                continue
            pipe = joblib.load(model_path)
            prefix = os.path.join(out_root, f"{feat}_{mdl}")
            res = roc_pr_with_ci(pipe, Xte, yte, B = B, seed = seed, out_prefix = prefix, show_ci = True)
            res.update({"feature": feat, "model": mdl})
            leaderboard.append(res)

    # Save leaderboard JSON and a summary bar chart for AUC/AP
    with open(os.path.join(out_root, "leaderboard_table.json"), "w") as f:
        json.dump(leaderboard, f, indent = 2)

    # Build bar plots of mean AUC/AP with CI whiskers
    # Aggregate as (feature, model) -> metrics
    def _collect(metric):
        items = [(f"{r['feature']}:{r['model']}", r[metric], r.get(metric+"_ci", [np.nan, np.nan])) for r in leaderboard]
        labels = [a for a,_,_ in items]
        means  = [b for _,b,_ in items]
        lows   = [b - c[0] if np.isfinite(c[0]) else np.nan for _,b,c in items]
        highs  = [c[1] - b if np.isfinite(c[1]) else np.nan for _,b,c in items]
        return labels, means, lows, highs

    for metric, title, fname in [("auc","ROC AUC","summary_auc_bars.png"), ("ap","Average Precision","summary_ap_bars.png")]:
        labels, means, lows, highs = _collect(metric)
        x = np.arange(len(labels))
        plt.figure(figsize = (max(7, 0.35*len(labels)), 4.2))
        plt.bar(x, means)
        # Error bars from CI
        yerr = np.vstack([lows, highs])
        plt.errorbar(x, means, yerr = yerr, fmt = 'none', ecolor = 'black', elinewidth = 1, capsize = 3)
        plt.xticks(x, labels, rotation = 45, ha = 'right')
        plt.ylabel(title)
        plt.title(f"{title} $-$ Test (95% CI by bootstrap)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, fname), dpi = 300)
        plt.close()