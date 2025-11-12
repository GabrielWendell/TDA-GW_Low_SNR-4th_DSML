# src/tda/run_vectorize.py
import os
from .vectorize import vectorize_PI, vectorize_PL, vectorize_BC

def run_vectorize(project_root: str = "."):
    pd_index_path = os.path.join(project_root, "data", "pd", "index.csv")
    # 1) PI
    out_pi = os.path.join(project_root, "data", "pi")
    vectorize_PI(pd_index_path, out_pi, grid_size = (50,50), sigma_frac = 0.05, weight = "p")
    print(f"\n> PI features -> {out_pi}")
    # 2) PL
    out_pl = os.path.join(project_root, "data", "pl")
    vectorize_PL(pd_index_path, out_pl, K = 5, T_samples = 256)
    print(f"> PL features -> {out_pl}")
    # 3) BC
    out_bc = os.path.join(project_root, "data", "bc")
    vectorize_BC(pd_index_path, out_bc, E_samples = 200)
    print(f"> BC features -> {out_bc}")

if __name__ == "__main__":
    run_vectorize(project_root = ".")
