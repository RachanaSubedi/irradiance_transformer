# ════════════════════════════════════════════════════════════
# scripts/02_build_pretrain_dataset.py
# Standalone utility: build and save the pretraining dataset
# (X, y, meta) WITHOUT training a model. Useful for inspecting
# the dataset before committing GPU time to 03_pretrain.py.
#
# FIXED (was broken): previously imported a nonexistent `CFG`
# object and a nonexistent `build_pretrain_dataset` function.
# That function never existed in irradiance.data — the real
# per-task sequence building always lived inline in
# 03_pretrain.py's Part A. This script now reproduces that
# logic standalone, saving to the same v3 artifact paths that
# 03_pretrain.py will load if the .npy files already exist.
#
# Note: 03_pretrain.py does NOT currently check for existing
# .npy files before rebuilding — if you run this script first,
# you would need to add a load-if-exists guard to 03_pretrain.py
# Part A to actually skip rebuilding. As-is, this script is for
# inspection; 03_pretrain.py always rebuilds from raw CSVs.
# ════════════════════════════════════════════════════════════

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd

from irradiance import config as cfg
from irradiance.data import build_master, build_pretrain_sequences

print("Building master dataframe...")
master = build_master(cfg)

print("\nBuilding pretraining sequences for all 3 tasks...")
all_X, all_y, all_meta = [], [], []
for target, anchor1, anchor2 in cfg.PRETRAIN_TASKS:
    Xt, yt, mt = build_pretrain_sequences(
        master, target, anchor1, anchor2,
        seq_len           = cfg.MODEL["seq_len"],
        center            = cfg.MODEL["center"],
        anchor_bad_thresh = 0.9,
        c13_gap_thresh    = 0.3,
    )
    all_X.extend(Xt); all_y.extend(yt); all_meta.extend(mt)

X       = np.stack(all_X)
y       = np.array(all_y)
meta_df = pd.DataFrame(all_meta)

rng = np.random.RandomState(cfg.TRAIN["seed"])
idx = rng.permutation(len(X))
X, y    = X[idx], y[idx]
meta_df = meta_df.iloc[idx].reset_index(drop=True)

print(f"\nDataset: X={X.shape}  y mean={y.mean():.3f}  y max={y.max():.3f}")
n_tail = (y > cfg.FINETUNE["tail_threshold"]).sum()
print(f"Tail samples (CSI > {cfg.FINETUNE['tail_threshold']}): "
      f"{n_tail:,} ({n_tail/len(y)*100:.1f}%)")
print(f"\nSamples per task:\n{meta_df['task'].value_counts().to_string()}")

np.save(cfg.ARTIFACTS["x_pretrain_v3"], X)
np.save(cfg.ARTIFACTS["y_pretrain_v3"], y)
meta_df.to_csv(cfg.ARTIFACTS["meta_pretrain_v3"], index=False)

print(f"\nSaved:")
print(f"  {cfg.ARTIFACTS['x_pretrain_v3']}")
print(f"  {cfg.ARTIFACTS['y_pretrain_v3']}")
print(f"  {cfg.ARTIFACTS['meta_pretrain_v3']}")
print("\nNext: run scripts/03_pretrain.py to train the encoder.")