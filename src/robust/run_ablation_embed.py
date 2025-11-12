from .ablation_embed import run_ablation

if __name__ == "__main__":
    # Default ±10% on both m and τ.
    run_ablation(project_root = ".", scales_m = (0.9,1.0,1.1), scales_tau = (0.9,1.0,1.1), 
                 max_points = 2000, seed = 2025)