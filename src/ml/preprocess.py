# src/ml/preprocess.py
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA


def make_scaler(name: str = "standard"):
    if name == "standard":
        return ("scaler", StandardScaler(with_mean = True, with_std = True))
    if name == "yeo_johnson":
        return ("pt", PowerTransformer(method = "yeo-johnson", standardize = True))
    raise ValueError(f"Unknown scaler: {name}")


def make_pca(pct_variance: Optional[float] = None):
    if pct_variance is None:
        return None
    if not (0.0 < pct_variance <= 1.0):
        raise ValueError("pct_variance must be in (0,1]")
    return ("pca", PCA(n_components = pct_variance, svd_solver = "full", whiten = False, random_state = 0))


def build_preprocessor(scaler: str = "standard", pca_var: Optional[float] = None) -> Pipeline:
    steps = [make_scaler(scaler)]
    p = make_pca(pca_var)
    if p is not None:
        steps.append(p)
    return Pipeline(steps)