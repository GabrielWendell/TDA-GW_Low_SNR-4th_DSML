from .snr_sweep import train_high_snr_clone, compare_baseline_vs_high, plot_curves

def main(project_root: str = "."):
    features = ("pi","pl","bc","baseline")
    models   = ("logreg","svm_rbf","rf")
    snr_levels = [2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    for thr in (8.0, 6.0):  # Two regimes: very-high SNR and mid-high SNR
        for feat in features:
            for mdl in models:
                train_high_snr_clone(project_root, feat, mdl, snr_min = thr)
                compare_baseline_vs_high(project_root, feat, mdl, snr_thr = thr, snr_levels = snr_levels)
                plot_curves(project_root, feat, mdl, snr_thr = thr)

if __name__ == "__main__":
    main(project_root = ".")