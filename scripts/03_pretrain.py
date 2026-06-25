# ════════════════════════════════════════════════════════════
# scripts/03_pretrain.py
# Build dataset + train Transformer encoder.
# Run ONCE on GPU. Outputs saved to datasets folder.
# Delete pretrain_best_model_v3.pt to retrain from scratch.
#
# v3 changes from v2 (audit for GHI ceiling fix):
#   - Trains on data built from snapshot-resampled stations
#     (data.py fix — no more .mean() peak compression)
#   - Uses the same tail-weighted MSE as fine-tuning, so the
#     encoder itself learns to represent high-CSI states well
#     before fine-tuning even starts.
# ════════════════════════════════════════════════════════════

import os, sys
sys.path.append("/content")  # no-op on PyCharm, needed for Colab

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from irradiance import config as cfg
from irradiance.data import build_master, CSIDataset, build_pretrain_sequences
from irradiance.model import TransformerImputer

TAIL_THRESHOLD = cfg.FINETUNE["tail_threshold"]
TAIL_WEIGHT    = cfg.FINETUNE["tail_weight"]

# ── Guard — skip if already trained ──────────────────────────
if os.path.exists(cfg.ARTIFACTS["pretrain_model_v3"]):
    print(f"✅ pretrain_best_model_v3.pt already exists. Skipping pretraining.")
    print(f"   Delete it to retrain: {cfg.ARTIFACTS['pretrain_model_v3']}")
    raise SystemExit(0)

print("=" * 60)
print("PRETRAINING V3 — 14 features, snapshot-resampled, tail-weighted loss")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# PART A — Build dataset
# ════════════════════════════════════════════════════════════

print("\n[A] Building pretraining dataset (snapshot-resampled stations)...")
master = build_master(cfg)

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
n_tail = (y > TAIL_THRESHOLD).sum()
print(f"Tail samples (CSI > {TAIL_THRESHOLD}): {n_tail:,} ({n_tail/len(y)*100:.1f}%)")
print(f"Samples per task:\n{meta_df['task'].value_counts().to_string()}")

print(f"\nFeature layout at center timestep:")
for i, name in enumerate(cfg.FEATURE_NAMES):
    print(f"  [{i:>2}] {name:<24}  "
          f"mean={X[:, cfg.MODEL['center'], i].mean():+.3f}  "
          f"std={X[:, cfg.MODEL['center'], i].std():.3f}")

np.save(cfg.ARTIFACTS["x_pretrain_v3"], X)
np.save(cfg.ARTIFACTS["y_pretrain_v3"], y)
meta_df.to_csv(cfg.ARTIFACTS["meta_pretrain_v3"], index=False)
print(f"\nSaved dataset ✅")

# ════════════════════════════════════════════════════════════
# PART B — Train model
# ════════════════════════════════════════════════════════════

print("\n[B] Training Transformer encoder...")

# Time-based 70/15/15 split (no data leakage)
meta_df["datetime_center"] = pd.to_datetime(meta_df["datetime_center"])
sort_idx  = meta_df["datetime_center"].argsort().values
X_s, y_s  = X[sort_idx], y[sort_idx]
meta_s    = meta_df.iloc[sort_idx].reset_index(drop=True)

n       = len(X_s)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)

X_train   = X_s[:n_train];               y_train   = y_s[:n_train]
X_val     = X_s[n_train:n_train+n_val];  y_val     = y_s[n_train:n_train+n_val]
X_test    = X_s[n_train+n_val:];         y_test    = y_s[n_train+n_val:]
meta_test = meta_s.iloc[n_train+n_val:].reset_index(drop=True)

print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

BS = cfg.TRAIN["batch_size"]
train_loader = DataLoader(CSIDataset(X_train, y_train), batch_size=BS,   shuffle=True)
val_loader   = DataLoader(CSIDataset(X_val,   y_val),   batch_size=BS*2, shuffle=False)
test_loader  = DataLoader(CSIDataset(X_test,  y_test),  batch_size=BS*2, shuffle=False)

model = TransformerImputer(
    input_dim  = cfg.MODEL["n_features"],
    d_model    = cfg.MODEL["d_model"],
    nhead      = cfg.MODEL["n_heads"],
    num_layers = cfg.MODEL["n_layers"],
    d_ff       = cfg.MODEL["d_ff"],
    dropout    = cfg.MODEL["dropout"],
    center     = cfg.MODEL["center"],
).to(cfg.DEVICE)

print(f"Parameters: {model.count_params():,} | "
      f"Input projection: Linear({cfg.MODEL['n_features']} → {cfg.MODEL['d_model']})")

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN["lr"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.7, patience=5, min_lr=1e-6)


def tail_weighted_mse(pred, target, threshold=TAIL_THRESHOLD, tail_weight=TAIL_WEIGHT):
    """Same tail-weighted MSE used in fine-tuning — see 04_finetune.py
    for full rationale. Applied here too so the encoder itself learns
    good high-CSI representations during pretraining, not just the head."""
    se = (pred - target) ** 2
    weight = torch.ones_like(target)
    weight = torch.where(target > threshold,
                         torch.full_like(target, tail_weight),
                         weight)
    return (se * weight).mean()


loss_fn = tail_weighted_mse

best_val_loss = float("inf")
best_state    = None
best_epoch    = 0
patience_cnt  = 0
train_losses, val_losses = [], []

print(f"\n{'Ep':>4} {'Train':>10} {'Val':>10} {'LR':>10}  *")
print("-" * 48)

for epoch in range(1, cfg.TRAIN["epochs"] + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
            val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
    val_loss /= len(val_loader.dataset)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss
        best_epoch    = epoch
        patience_cnt  = 0
        best_state    = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_cnt += 1

    cur_lr = optimizer.param_groups[0]["lr"]
    print(f"{epoch:>4} {train_loss:>10.5f} {val_loss:>10.5f} "
          f"{cur_lr:>10.2e}  {'✓' if is_best else ''}")

    if patience_cnt >= cfg.TRAIN["patience"]:
        print(f"\nEarly stopping at epoch {epoch} (best: epoch {best_epoch})")
        break

# ── Test evaluation ───────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_preds.append(model(xb.to(cfg.DEVICE)).cpu().numpy())
        all_trues.append(yb.numpy())

preds_pt = np.concatenate(all_preds)
trues_pt = np.concatenate(all_trues)
rmse_pt  = np.sqrt(((preds_pt - trues_pt) ** 2).mean())
r2_pt    = 1 - ((preds_pt - trues_pt) ** 2).sum() / ((trues_pt - trues_pt.mean()) ** 2).sum()

print(f"\n{'='*50}")
print(f"TEST RESULTS:  RMSE={rmse_pt:.4f}  R²={r2_pt:.4f}")
for task in ["hide_s1", "hide_s2", "hide_s3"]:
    m = meta_test["task"] == task
    if m.sum() == 0:
        continue
    tp  = preds_pt[m.values]; tt = trues_pt[m.values]
    r   = np.sqrt(((tp - tt) ** 2).mean())
    r2t = 1 - ((tp - tt) ** 2).sum() / ((tt - tt.mean()) ** 2).sum()
    print(f"  {task}  RMSE={r:.4f}  R²={r2t:.4f}  N={m.sum()}")

# ── Tail check on test set ─────────────────────────────────────
tail_mask_pt = trues_pt > TAIL_THRESHOLD
if tail_mask_pt.sum() > 0:
    tail_rmse_pt = np.sqrt(((preds_pt[tail_mask_pt] - trues_pt[tail_mask_pt]) ** 2).mean())
    tail_bias_pt = (preds_pt[tail_mask_pt] - trues_pt[tail_mask_pt]).mean()
    print(f"\nPretraining tail check (CSI > {TAIL_THRESHOLD}, n={tail_mask_pt.sum()}):")
    print(f"  Tail RMSE: {tail_rmse_pt:.4f}  |  Tail bias: {tail_bias_pt:+.4f}")

# ── Save checkpoint ───────────────────────────────────────────
torch.save({
    "model_state": best_state,
    "config": {
        "input_dim":  cfg.MODEL["n_features"],
        "d_model":    cfg.MODEL["d_model"],
        "nhead":      cfg.MODEL["n_heads"],
        "num_layers": cfg.MODEL["n_layers"],
        "d_ff":       cfg.MODEL["d_ff"],
        "dropout":    cfg.MODEL["dropout"],
        "seq_len":    cfg.MODEL["seq_len"],
        "center":     cfg.MODEL["center"],
        "features":   cfg.FEATURE_NAMES,
    },
    "best_epoch":     best_epoch,
    "val_loss":       best_val_loss,
    "test_rmse":      float(rmse_pt),
    "test_r2":        float(r2_pt),
    "loss_fn":        "tail_weighted_mse",
    "tail_threshold": TAIL_THRESHOLD,
    "tail_weight":    TAIL_WEIGHT,
    "version":        "v3_14features",
}, cfg.ARTIFACTS["pretrain_model_v3"])

print(f"\nSaved: {cfg.ARTIFACTS['pretrain_model_v3']} ✅")
print("\nNext: run scripts/04_finetune.py")