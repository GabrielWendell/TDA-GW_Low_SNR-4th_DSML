# src/ml/run_visualize.py
from .viz import make_all_plots

if __name__ == "__main__":
    make_all_plots(project_root = ".", 
                   features = ("pi","pl","bc","baseline"), models = ("logreg","svm_rbf","rf"), 
                   B = 500, seed = 2025)