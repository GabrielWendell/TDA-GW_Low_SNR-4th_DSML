# src/ml/pipelines.py
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .preprocess import build_preprocessor


DEFAULT_CW = "balanced"  # Robust with SNR‑stratified class imbalance


def pipeline_logreg(scaler: str = "standard", pca: Optional[float] = None,
                    C: float = 1.0, penalty: str = "l2", max_iter: int = 2000, random_state: int = 42) -> Pipeline:
    pre = build_preprocessor(scaler, pca)
    clf = LogisticRegression(C=C, penalty = penalty, solver = "lbfgs", max_iter = max_iter,
                             class_weight = DEFAULT_CW, random_state = random_state, n_jobs = None)
    return Pipeline([("pre", pre), ("clf", clf)])


def pipeline_svm_rbf(scaler: str = "standard", pca: Optional[float] = None,
                     C: float = 1.0, gamma: str = "scale", random_state: int = 42) -> Pipeline:
    pre = build_preprocessor(scaler, pca)
    clf = SVC(C = C, kernel = "rbf", gamma = gamma, probability = True, 
              class_weight = DEFAULT_CW, random_state = random_state)
    return Pipeline([("pre", pre), ("clf", clf)])


def pipeline_rf(n_estimators: int = 500, max_depth: Optional[int] = None,
                max_features: str = "sqrt", random_state: int = 42) -> Pipeline:
    # RF is scale‑invariant; keep a no‑op preprocessor for API consistency if desired
    pre = build_preprocessor(scaler = "standard", pca_var = None)
    clf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth,
                                 max_features = max_features, class_weight = DEFAULT_CW,
                                 random_state = random_state, n_jobs = -1)
    return Pipeline([("pre", pre), ("clf", clf)])