"""Verify the sfp conda environment is correctly set up.

Usage:
    conda activate sfp
    python verify_env.py
"""
import sys
import importlib
import subprocess
import os

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
errors = []


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"{status}  {label}"
    if detail:
        msg += f"  ({detail})"
    if not condition:
        errors.append(label)
    return msg


results = []

# 1. Python version
v = sys.version_info
results.append(check(
    "Python 3.8.x",
    v.major == 3 and v.minor == 8,
    f"found {v.major}.{v.minor}.{v.micro}"
))

# 2. Critical pinned packages
import numpy as np
results.append(check("numpy == 1.23.x", np.__version__.startswith("1.23"), np.__version__))

import torch
results.append(check("torch == 2.2.x", torch.__version__.startswith("2.2"), torch.__version__))

# 3. Range-pinned packages
import snakemake
results.append(check("snakemake 5.x", snakemake.__version__.startswith("5."), snakemake.__version__))

import seaborn as sns
sn_major, sn_minor = (int(x) for x in sns.__version__.split(".")[:2])
results.append(check("seaborn >= 0.11, < 0.14", sn_major == 0 and 11 <= sn_minor < 14, sns.__version__))

try:
    from seaborn.algorithms import bootstrap
    results.append(check("seaborn.algorithms.bootstrap importable", True))
except ImportError as e:
    results.append(check("seaborn.algorithms.bootstrap importable", False, str(e)))

import pandas as pd
pd_major = int(pd.__version__.split(".")[0])
results.append(check("pandas < 2.0", pd_major < 2, pd.__version__))

import nibabel
nib_major = int(nibabel.__version__.split(".")[0])
results.append(check("nibabel < 4.0", nib_major < 4, nibabel.__version__))

# 4. Other required packages
for pkg_name, import_name in [
    ("matplotlib", "matplotlib"),
    ("scipy", "scipy"),
    ("h5py", "h5py"),
    ("pyyaml", "yaml"),
    ("scikit-learn", "sklearn"),
    ("pillow", "PIL"),
    ("requests", "requests"),
    ("GSN", "gsn"),
]:
    try:
        importlib.import_module(import_name)
        results.append(check(f"import {pkg_name}", True))
    except ImportError as e:
        results.append(check(f"import {pkg_name}", False, str(e)))

# 5. Local package
try:
    import sfp_nsdsyn
    results.append(check("sfp_nsdsyn importable", True))
except ImportError:
    results.append(check("sfp_nsdsyn importable", False, "run: pip install -e ."))

# 6. Snakemake can parse Snakefile
project_dir = os.path.dirname(os.path.abspath(__file__))
result = subprocess.run(
    [sys.executable, "-m", "snakemake", "-j1", "-n", "test_run", "--quiet"],
    capture_output=True, text=True, cwd=project_dir
)
snakemake_ok = result.returncode == 0
detail = "ok" if snakemake_ok else result.stderr.strip()[:120]
results.append(check("Snakemake parses Snakefile (dry-run test_run)", snakemake_ok, detail))

# Print results
for r in results:
    sys.stdout.write(r + "\n")

sys.stdout.write(f"\n{'=' * 50}\n")
if errors:
    sys.stdout.write(f"FAILED: {len(errors)} check(s)\n")
    for e in errors:
        sys.stdout.write(f"  - {e}\n")
    sys.exit(1)
else:
    sys.stdout.write("All checks passed.\n")
