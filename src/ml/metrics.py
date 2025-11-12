# src/ml/metrics.py
import numpy as np
from typing import Dict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_hat = (y_prob >= thresh).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_hat)) if len(np.unique(y_true)) > 1 else float("nan"),
    }