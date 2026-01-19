"""
Check if all required dependencies are installed
"""
import sys

missing_packages = []

try:
    import pandas
except ImportError:
    missing_packages.append("pandas")

try:
    import requests
except ImportError:
    missing_packages.append("requests")

try:
    import datasets
except ImportError:
    missing_packages.append("datasets")

try:
    import transformers
except ImportError:
    missing_packages.append("transformers")

try:
    import torch
except ImportError:
    missing_packages.append("torch")

try:
    import optuna
except ImportError:
    missing_packages.append("optuna")

try:
    import matplotlib
except ImportError:
    missing_packages.append("matplotlib")

try:
    import seaborn
except ImportError:
    missing_packages.append("seaborn")

try:
    import sklearn
except ImportError:
    missing_packages.append("scikit-learn")

if missing_packages:
    print("Missing required packages:")
    for pkg in missing_packages:
        print(f"  - {pkg}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(missing_packages)}")
    print("\nOr install all dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
else:
    print("✓ All required packages are installed!")
    sys.exit(0)
