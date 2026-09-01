"""
Scaffold script — creates the src/ folder structure for the grokking benchmark rebuild.
Run once from the project root: python scaffold.py
"""

import os

FILES = [
    "src/data/modular_arithmetic.py",
    "src/models/transformer.py",
    "src/train.py",
]

for path in FILES:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(path):
        open(path, "w").close()
        print(f"created: {path}")
    else:
        print(f"skipped (already exists): {path}")
