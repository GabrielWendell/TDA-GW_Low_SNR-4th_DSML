# src/ml/eval.py
import os, json, joblib, numpy as np
from typing import Dict, List, Tuple
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from .io import load_feature_set   # Step 3.1
# Models were saved under models/<feature>/{logreg,svm_rbf,rf}.joblib

MetricDict = Dict[str, float]

def _probs(pipe, X):
    """Return positive-class probabilities from a pipeline; fall back to a calibrated rank for AUC/AP."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    # SVM without probas: use decision_function and min-max scale to (0,1).
    s = pipe.decision_function(X)
    smin, smax = float(np.min(s)), float(np.max(s))
    return (s - smin) / (smax - smin + 1e-12)

def _metrics(y_true: np.ndarray, p: np.ndarray, thresh: float = 0.5) -> MetricDict:
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    y_hat = (p >= thresh).astype(int)
    # Guard degenerate splits (all one class)
    auc = float("nan") if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, p))
    ap  = float("nan") if len(np.unique(y_true)) < 2 else float(average_precision_score(y_true, p))
    return {
        "auc": auc,
        "ap": ap,
        "brier": float(brier_score_loss(y_true, p)),
        "f1": float(f1_score(y_true, y_hat)) if len(np.unique(y_true)) > 1 else float("nan"),
    }

def cv_evaluate_unfitted(pipe_fitted, X: np.ndarray, y: np.ndarray, cv_folds: int = 5, random_state: int = 2025) -> MetricDict:
    """
    Cross-validated metrics on TRAIN using an unfitted clone of the loaded pipeline (same hyperparameters).
    """
    kf = StratifiedKFold(n_splits = cv_folds, shuffle = True, random_state = random_state)
    aucs, aps, briers, f1s = [], [], [], []
    base = clone(pipe_fitted)  # Unfitted clone with same hyperparams
    for tr, va in kf.split(X, y):
        pipe = clone(base)
        pipe.fit(X[tr], y[tr])
        p = _probs(pipe, X[va])
        m = _metrics(y[va], p)
        if np.isfinite(m["auc"]):   aucs.append(m["auc"])
        if np.isfinite(m["ap"]):    aps.append(m["ap"])
        briers.append(m["brier"])
        if np.isfinite(m["f1"]):    f1s.append(m["f1"])
    def mean_or_nan(v): return float(np.mean(v)) if len(v) else float("nan")
    return {"auc": mean_or_nan(aucs), "ap": mean_or_nan(aps),
            "brier": mean_or_nan(briers), "f1": mean_or_nan(f1s)}

def test_evaluate_fitted(pipe_fitted, X: np.ndarray, y: np.ndarray, snr: np.ndarray = None, thresh: float = 0.5) -> Dict:
    """
    Evaluate saved (train+val-fitted) pipeline on TEST: overall + per-SNR table.
    """
    p = _probs(pipe_fitted, X)
    overall = _metrics(y, p, thresh = thresh)
    per_snr = []
    if snr is not None:
        snr_unique = np.unique(snr)
        for s in sorted(snr_unique):
            mask = (snr == s)
            mm = _metrics(y[mask], p[mask], thresh = thresh)
            mm.update({"snr": float(s), "n": int(mask.sum())})
            per_snr.append(mm)
    return {"overall": overall, "per_snr": per_snr}

def run_eval_for_feature(project_root: str,
                         feature_tag: str,
                         models: List[str] = ("logreg", "svm_rbf", "rf"),
                         cv_folds: int = 5,
                         random_state: int = 2025,
                         thresh: float = 0.5) -> None:
    """
    Load feature set + trained models and write CV (train) and TEST metrics.
    """
    feat_dir = os.path.join(project_root, "data", feature_tag if feature_tag != "baseline" else "baseline")
    fs = load_feature_set(feat_dir)
    out_dir = os.path.join(project_root, "reports", "eval", feature_tag)
    os.makedirs(out_dir, exist_ok = True)

    # TEST metadata (for per-SNR)
    snr_test = np.array([float(m["snr_target"]) for m in fs.test.meta], dtype = float)

    for model_name in models:
        model_path = os.path.join(project_root, "models", feature_tag, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model: {model_path} — skipping.")
            continue
        pipe_fitted = joblib.load(model_path)

        # 1) CV on TRAIN (using unfitted clone with same hyperparams)
        cvm = cv_evaluate_unfitted(pipe_fitted, fs.train.X, fs.train.y, cv_folds=cv_folds, random_state=random_state)
        with open(os.path.join(out_dir, f"cv_{model_name}.json"), "w") as f:
            json.dump(cvm, f, indent=2)

        # 2) HELD-OUT TEST using the saved (train+val–fitted) pipeline
        tm = test_evaluate_fitted(pipe_fitted, fs.test.X, fs.test.y, snr=snr_test, thresh=thresh)
        with open(os.path.join(out_dir, f"test_{model_name}_overall.json"), "w") as f:
            json.dump(tm["overall"], f, indent=2)
        # per-SNR table
        with open(os.path.join(out_dir, f"test_{model_name}_per_snr.json"), "w") as f:
            json.dump(tm["per_snr"], f, indent=2)

        print(f"\n[{feature_tag}/{model_name}] CV(train) = {cvm} | TEST(overall) = {tm['overall']}")
