import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath("06_Code/scripts"))
from analyze_nanda_unified import evaluate_frozen_protocol, attach_pass_checks

G = np.array([8021, 8253, 4094, 7831, 6637], dtype=float)
rng = np.random.default_rng(0)
N = 2000

for cv in (0.02, 0.05, 0.10, 0.20, 0.30, 0.40):
    hits = 0
    for _ in range(N):
        P = 0.5 * G * (1 + rng.normal(0, cv, size=5))
        v = attach_pass_checks(evaluate_frozen_protocol(G, list(P), rule="fraction", c=0.8))
        hits += v["passed"]
    print(f"cv={cv:0.2f}  pass rate = {hits/N:0.3f}")
