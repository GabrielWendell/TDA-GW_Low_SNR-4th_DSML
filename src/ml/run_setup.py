# src/ml/run_setup.py
import os, json
from sklearn.metrics import roc_auc_score
from .io import load_feature_set, dataset_summary
from .pipelines import pipeline_logreg, pipeline_svm_rbf, pipeline_rf


def _fit_eval_quick(pipe, Xtr, ytr, Xva, yva):
    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xva)[:,1]
    return float(roc_auc_score(yva, p))


def main(project_root: str = "."):
    cfg = json.load(open(os.path.join(project_root, "configs", "ml_setup.json")))
    dirs = cfg["feature_dirs"]

    def do_one(tag: str, path: str):
        fs = load_feature_set(os.path.join(project_root, path))
        summ = dataset_summary(fs)
        print(f"[{tag}] {summ}")
        # Quick smoke on small models (no tuning yet)
        auc_lr  = _fit_eval_quick(pipeline_logreg(cfg["scaler"], cfg["pca_variance"]), fs.train.X, fs.train.y, fs.val.X, fs.val.y)
        auc_svm = _fit_eval_quick(pipeline_svm_rbf(cfg["scaler"], cfg["pca_variance"]), fs.train.X, fs.train.y, fs.val.X, fs.val.y)
        auc_rf  = _fit_eval_quick(pipeline_rf(), fs.train.X, fs.train.y, fs.val.X, fs.val.y)
        print(f"[{tag}] AUC (val) — LR: {auc_lr:.3f} | SVM: {auc_svm:.3f} | RF: {auc_rf:.3f}")

    for tag, path in dirs.items():
        do_one(tag, path)

if __name__ == "__main__":
    main(project_root = ".")