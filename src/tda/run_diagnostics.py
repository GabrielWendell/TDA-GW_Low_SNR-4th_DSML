# src/tda/run_diagnostics.py
from .diagnostics import run_diagnostics

if __name__ == "__main__":
    run_diagnostics(
        project_root = ".",
        repeats = 3,          # Small-noise repeats
        alpha = 1e-3,         # Jitter scale factor
        max_points = 2000,    # Cap points for PD recomputation
        seed = 2025,
        use_backend = None    # 'ripser'|'gudhi' or None to auto-pick
    )
