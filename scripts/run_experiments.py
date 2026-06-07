"""Ebola experiment pipeline: ablation study + figure generation.

Usage:
  python scripts/run_experiments.py --dry-run      # verify without GPU
  python scripts/run_experiments.py --ablation     # run full experiment (GPU needed)
  python scripts/run_experiments.py --figures      # generate paper figures from results
"""

import os, sys, json, re, argparse
import numpy as np
import pandas as pd

os.chdir("D:/Ebola")
sys.path.insert(0, "D:/Ebola")

TREND_THRESHOLD = 0.15
CONTEXT_WINDOW = 4
CLASSES = ["increase", "decrease", "stable"]
CLASS_LABELS = ["Increase", "Decrease", "Stable"]

