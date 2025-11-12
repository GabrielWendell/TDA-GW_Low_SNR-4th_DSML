# src/ml/run_eval.py
import os
from .eval import run_eval_for_feature

def main(project_root: str = "."):
    os.makedirs(os.path.join(project_root, "reports", "eval"), exist_ok = True)
    for feature_tag in ("pi", "pl", "bc", "baseline"):
        run_eval_for_feature(project_root, feature_tag,
                             models = ("logreg", "svm_rbf", "rf"),
                             cv_folds=5, random_state = 2025, thresh = 0.5)

if __name__ == "__main__":
    main(project_root = ".")
