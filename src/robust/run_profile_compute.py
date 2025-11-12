from .profile_compute import run_profile

if __name__ == "__main__":
    # Default: 30 signals, max_points = 2000
    run_profile(project_root = ".", n_samples = 30, max_points = 2000, seed = 2025)