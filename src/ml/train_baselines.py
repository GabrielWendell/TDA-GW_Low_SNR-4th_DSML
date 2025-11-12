# src/ml/train_baselines.py
import os, json, joblib, numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from ..ml.io import load_feature_set
from ..ml.pipelines import pipeline_logreg, pipeline_svm_rbf, pipeline_rf
from ..ml.metrics import compute_metrics

CFG_DEFAULT = {
    "random_state": 2025,
    "cv_folds": 5,
    "models": {
        "logreg": {
            "grid": {"C": [0.01, 0.1, 1.0, 10.0]},
            "scaler": "standard", "pca": None
        },
        "svm_rbf": {
            "grid": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 1e-3, 1e-2, 1e-1]},
            "scaler": "standard", "pca": None
        },
        "rf": {
            "grid": {"n_estimators": [300, 600], "max_depth": [None, 12, 24], "max_features": ["sqrt", 0.3, 0.5]},
            "scaler": "standard", "pca": None
        }
    },
    "feature_dirs": {
        "pi": "data/pi",
        "pl": "data/pl",
        "bc": "data/bc",
        "baseline": "data/baseline"
    }
}


def _cv_auc(pipe_maker, grid: Dict, X: np.ndarray, y: np.ndarray, random_state: int, cv_folds: int) -> Tuple[Dict, float]:
    """Return (best_params, best_auc) found by manual grid over StratifiedKFold."""
    kf = StratifiedKFold(n_splits = cv_folds, shuffle = True, random_state = random_state)
    best_auc, best_params = -np.inf, None
    # Build cartesian product of grid
    from itertools import product
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in product(*values):
        params = dict(zip(keys, combo))
        aucs = []
        for tr, va in kf.split(X, y):
            pipe = pipe_maker(**params)
            pipe.fit(X[tr], y[tr])
            # Use predict_proba if available, else decision_function scaled to [0,1]
            if hasattr(pipe, "predict_proba"):
                p = pipe.predict_proba(X[va])[:, 1]
            else:
                s = pipe.decision_function(X[va])
                # Min-Max to pseudo-prob (for AUROC only this scaling is monotone and OK)
                smin, smax = np.min(s), np.max(s)
                p = (s - smin) / (smax - smin + 1e-12)
            aucs.append(roc_auc_score(y[va], p))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc, best_params = mean_auc, params
    return best_params, best_auc


def _refit_and_save(tag: str, pipe, Xtr, ytr, Xva, yva, out_dir: str):
    os.makedirs(out_dir, exist_ok = True)
    # Fit on train+val
    Xtv = np.vstack([Xtr, Xva])
    ytv = np.concatenate([ytr, yva])
    pipe.fit(Xtv, ytv)
    # Quick val metrics (using original val as a sanity check)
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(Xva)[:, 1]
    else:
        s = pipe.decision_function(Xva)
        smin, smax = np.min(s), np.max(s)
        p = (s - smin) / (smax - smin + 1e-12)
    metrics = compute_metrics(yva, p)
    # Save
    joblib.dump(pipe, os.path.join(out_dir, f"{tag}.joblib"))
    with open(os.path.join(out_dir, f"{tag}_val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent = 2)
    return metrics


def run_train(project_root: str = ".", cfg: Dict = None):
    if cfg is None:
        cfg = CFG_DEFAULT
    rs = cfg["random_state"]
    folds = cfg["cv_folds"]

    # Feature families
    for feat_tag, feat_dir in cfg["feature_dirs"].items():
        path = os.path.join(project_root, feat_dir)
        fs = load_feature_set(path)
        Xtr, ytr = fs.train.X, fs.train.y
        Xva, yva = fs.val.X, fs.val.y

        print(f"\n=== [{feat_tag.upper()}] n_train = {Xtr.shape[0]} | n_val = {Xva.shape[0]} | d = {Xtr.shape[1]} ===")

        # --- Logistic Regression ---
        print("[LOGREG] grid-search by AUROC…")
        lr_best, lr_auc = _cv_auc(lambda **kw: pipeline_logreg(scaler = "standard", pca = None, **kw), cfg["models"]["logreg"]["grid"], Xtr, ytr, rs, folds)
        print(f"[LOGREG] best = {lr_best} | cv_auc = {lr_auc:.3f}")
        lr_pipe = pipeline_logreg(scaler = "standard", pca = None, **lr_best)
        lr_metrics = _refit_and_save("logreg", lr_pipe, Xtr, ytr, Xva, yva, os.path.join(project_root, "models", feat_tag))
        print(f"[LOGREG] val metrics: {lr_metrics}")

        # --- SVM RBF ---
        print("[SVM-RBF] grid-search by AUROC…")
        svm_best, svm_auc = _cv_auc(lambda **kw: pipeline_svm_rbf(scaler = "standard", pca = None, **kw), cfg["models"]["svm_rbf"]["grid"], Xtr, ytr, rs, folds)
        print(f"[SVM-RBF] best = {svm_best} | cv_auc = {svm_auc:.3f}")
        svm_pipe = pipeline_svm_rbf(scaler = "standard", pca=None, **svm_best)
        svm_metrics = _refit_and_save("svm_rbf", svm_pipe, Xtr, ytr, Xva, yva, os.path.join(project_root, "models", feat_tag))
        print(f"[SVM-RBF] val metrics: {svm_metrics}")

        # --- Random Forest ---
        print("[RF] grid-search by AUROC…")
        def make_rf(**kw):
            return pipeline_rf(**kw)
        rf_best, rf_auc = _cv_auc(make_rf, cfg["models"]["rf"]["grid"], Xtr, ytr, rs, folds)
        print(f"[RF] best={rf_best} | cv_auc = {rf_auc:.3f}")
        rf_pipe = pipeline_rf(**rf_best)
        rf_metrics = _refit_and_save("rf", rf_pipe, Xtr, ytr, Xva, yva, os.path.join(project_root, "models", feat_tag))
        print(f"[RF] val metrics: {rf_metrics}")

        # Save a small leaderboard per feature set
        leaderboard = {
            "logreg": {"cv_auc": lr_auc, "val": lr_metrics},
            "svm_rbf": {"cv_auc": svm_auc, "val": svm_metrics},
            "rf": {"cv_auc": rf_auc, "val": rf_metrics}
        }
        out_dir = os.path.join(project_root, "reports", "baselines")
        os.makedirs(out_dir, exist_ok = True)
        with open(os.path.join(out_dir, f"leaderboard_{feat_tag}.json"), "w") as f:
            json.dump(leaderboard, f, indent=2)

if __name__ == "__main__":
    run_train(project_root = ".")