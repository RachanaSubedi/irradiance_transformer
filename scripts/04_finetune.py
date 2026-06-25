# ════════════════════════════════════════════════════════════
# scripts/finetune_p2_v2.py
# Fine-tune pretrained v2 model on partial station P2.
# Requires pretrain_best_model_v2.pt to already exist on Drive.
# Outputs full-year GHI CSV + fine-tuned model checkpoint.
# ════════════════════════════════════════════════════════════

import os, sys
sys.path.append("/content")

import numpy as np
import pandas as pd
import pvlib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")   # headless-safe; remove if running in notebook
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from irradiance import config as cfg
from irradiance.data import (
    build_master,
    CSIDataset,
    process_station_utc,
    process_nsrdb_utc,
    build_finetune_sequences,
    fix_missing_anchor,
)
from irradiance.model import TransformerImputer

# ════════════════════════════════════════════════════════════
# STATION CONFIG — edit this block for new stations
# ════════════════════════════════════════════════════════════

STATION_NAME  = "p2"
ANCHOR1       = cfg.P2["anchor1"]         # "s3"
ANCHOR2       = cfg.P2["anchor2"]         # "s1"
NEW_LAT       = cfg.STATIONS["p2"]["lat"]
NEW_LON       = cfg.STATIONS["p2"]["lon"]
NEW_ALT       = cfg.STATIONS["p2"]["alt"]

LOCAL_GHI_PATH = cfg.RAW["ghi_p2"]
NSRDB_NEW_PATH = cfg.RAW["nsrdb_p2"]
C13_NEW_PATH   = cfg.RAW["c13_p2"]
C13_NEW_COL    = cfg.C13_COLS["p2"]

OVERLAP_START  = cfg.P2["overlap_start"]
OVERLAP_END    = cfg.P2["overlap_end"]
VAL_START      = cfg.P2["val_start"]
IMP_START      = cfg.P2["imp_start"]
IMP_END        = cfg.P2["imp_end"]

OUT_GHI_CSV    = cfg.ARTIFACTS["ghi_csv_v2"]
OUT_MODEL_PT   = cfg.ARTIFACTS["ft_model_v2"]
OUT_PLOT_FT    = os.path.join(os.path.dirname(OUT_GHI_CSV), "finetune_results_v2.png")

FEAT_NAMES_V2 = [
    "anchor1_CSI", "anchor1_mask", "anchor2_CSI", "anchor2_mask",
    "NSRDB_CSI_target", "C13_norm_target",
    "NSRDB_CSI_anchor1", "C13_norm_anchor1",
    "NSRDB_CSI_anchor2", "C13_norm_anchor2",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]

# ════════════════════════════════════════════════════════════
print("=" * 60)
print(f"FINE-TUNING V2 — Station {STATION_NAME.upper()}")
print(f"Anchors: {ANCHOR1.upper()} (nearest) | {ANCHOR2.upper()} (second)")
print("=" * 60)

# ── Step 1: Load partial station GHI ─────────────────────────
print("\n[1] Loading partial station GHI...")
st_new = process_station_utc(pd.read_csv(LOCAL_GHI_PATH), NEW_LAT, NEW_LON, NEW_ALT)
print(f"  Rows: {len(st_new)} | daytime CSI: {st_new['CSI'].notna().sum()}")
print(f"  Range: {st_new['datetime'].min()} → {st_new['datetime'].max()}")

# ── Step 2: Load target NSRDB and C13 ────────────────────────
print("\n[2] Loading target station NSRDB and C13...")
ns_raw = pd.read_csv(NSRDB_NEW_PATH, skiprows=2)
for col in ["Year", "Month", "Day", "Hour", "Minute", "GHI", "Clearsky GHI"]:
    ns_raw[col] = pd.to_numeric(ns_raw[col], errors="coerce")
ns_raw["datetime"] = (pd.to_datetime({
    "year":   ns_raw["Year"].astype(int),
    "month":  ns_raw["Month"].astype(int),
    "day":    ns_raw["Day"].astype(int),
    "hour":   ns_raw["Hour"].astype(int),
    "minute": ns_raw["Minute"].astype(int),
}) + pd.Timedelta(hours=8)).dt.tz_localize("UTC")
ns_raw["NSRDB_CSI_new"] = (ns_raw["GHI"] / ns_raw["Clearsky GHI"]).clip(0, 1.5)
ns_raw.loc[ns_raw["Clearsky GHI"] < 10, "NSRDB_CSI_new"] = np.nan
ns_new = ns_raw[["datetime", "NSRDB_CSI_new"]].sort_values("datetime").reset_index(drop=True)
print(f"  NSRDB: {len(ns_new)} rows | valid: {ns_new['NSRDB_CSI_new'].notna().sum()}")

c13_raw = pd.read_csv(C13_NEW_PATH)
c13_raw["datetime"] = pd.to_datetime(c13_raw["datetime_local"], utc=True)
c13_raw[C13_NEW_COL] = pd.to_numeric(c13_raw[C13_NEW_COL], errors="coerce")
c13_mean = c13_raw[C13_NEW_COL].mean()
c13_std  = c13_raw[C13_NEW_COL].std()
c13_raw["c13_new_norm"] = (c13_raw[C13_NEW_COL] - c13_mean) / c13_std
c13_new = c13_raw[["datetime", "c13_new_norm"]].sort_values("datetime").reset_index(drop=True)
print(f"  C13: {len(c13_new)} rows | mean={c13_mean:.1f}K  std={c13_std:.1f}K")

# ── Step 3 & 4: Build master and ft_master ────────────────────
print("\n[3] Building global master...")
master = build_master(cfg)

print("\n[4] Building ft_master...")
ft_master = master.copy()
for df_merge in [ns_new, c13_new]:
    ft_master = pd.merge_asof(
        ft_master.sort_values("datetime"),
        df_merge.sort_values("datetime"),
        on="datetime", direction="nearest",
        tolerance=pd.Timedelta("16min"))

ft_master = pd.merge_asof(
    ft_master.sort_values("datetime"),
    st_new[["datetime", "CSI"]].rename(columns={"CSI": "CSI_new"}).sort_values("datetime"),
    on="datetime", direction="nearest", tolerance=pd.Timedelta("16min"))

ft_master["NSRDB_CSI_new"] = ft_master["NSRDB_CSI_new"].fillna(-1.0)
ft_master["c13_new_norm"]   = ft_master["c13_new_norm"].ffill(limit=2).fillna(0.0)
ft_master["CSI_new"]        = ft_master["CSI_new"].fillna(-1.0)
ft_master["CSI_new_mask"]   = (ft_master["CSI_new"] >= 0).astype(np.float32)
ft_master = ft_master.reset_index(drop=True)
print(f"  ft_master shape: {ft_master.shape}")

# ── Step 5: Overlap sequences + train/val split ───────────────
print("\n[5] Building overlap sequences...")
ov_df = ft_master[
    (ft_master["datetime"] >= pd.Timestamp(OVERLAP_START, tz="UTC")) &
    (ft_master["datetime"] <  pd.Timestamp(OVERLAP_END,   tz="UTC"))
].reset_index(drop=True)
print(f"  Overlap: {ov_df['datetime'].min()} → {ov_df['datetime'].max()}")

X_ft, y_ft, dt_ft = build_finetune_sequences(
    ov_df, ANCHOR1, ANCHOR2,
    seq_len=cfg.MODEL["seq_len"],
    center=cfg.MODEL["center"],
    has_target=True)

dt_ft_pd   = pd.DatetimeIndex(dt_ft).tz_localize("UTC")
is_val     = dt_ft_pd >= pd.Timestamp(VAL_START, tz="UTC")
X_ft_train = X_ft[~is_val];  y_ft_train = y_ft[~is_val]
X_ft_val   = X_ft[is_val];   y_ft_val   = y_ft[is_val]
print(f"  Train: {len(X_ft_train)} | Val: {len(X_ft_val)}")

baseline_rmse = np.sqrt(((X_ft_val[:, cfg.MODEL["center"], 0] - y_ft_val) ** 2).mean())
print(f"\n  Baseline RMSE ({ANCHOR1.upper()} direct): {baseline_rmse:.4f} CSI")

# ── Step 6: Synthetic proxy sequences ────────────────────────
print("\n[6] Building synthetic proxy sequences (S2 as proxy target)...")
synth_df = master[
    (master["datetime"] >= pd.Timestamp(IMP_START, tz="UTC")) &
    (master["datetime"] <  pd.Timestamp(IMP_END,   tz="UTC"))
].copy().reset_index(drop=True)

synth_df = pd.merge_asof(
    synth_df.sort_values("datetime"),
    ft_master[["datetime", "NSRDB_CSI_new", "c13_new_norm"]].sort_values("datetime"),
    on="datetime", direction="nearest", tolerance=pd.Timedelta("16min"))
synth_df["CSI_new"]      = synth_df["CSI_s2"]
synth_df["CSI_new_mask"] = (synth_df["CSI_new"] >= 0).astype(np.float32)
synth_df = synth_df.reset_index(drop=True)

X_synth, y_synth, _ = build_finetune_sequences(
    synth_df, ANCHOR1, ANCHOR2,
    seq_len=cfg.MODEL["seq_len"],
    center=cfg.MODEL["center"],
    anchor_bad_thresh=0.9,
    has_target=True)
print(f"  Synthetic: {len(X_synth):,} | y mean={y_synth.mean():.3f}")

# ── Step 7: Load pretrained model + partial unfreeze ──────────
print("\n[7] Loading pretrained v2 model...")
model, ckpt = TransformerImputer.from_checkpoint(
    cfg.ARTIFACTS["pretrain_model_v2"], cfg.DEVICE)

for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "head" in name or "encoder.layers.1" in name:
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} (head + last encoder layer)")

# ── Step 8: Combine + fine-tune ───────────────────────────────
print("\n[8] Fine-tuning...")
X_real_rep = np.repeat(X_ft_train, cfg.FINETUNE["real_repeat"], axis=0)
y_real_rep = np.repeat(y_ft_train, cfg.FINETUNE["real_repeat"], axis=0)
X_combined = np.concatenate([X_synth, X_real_rep])
y_combined  = np.concatenate([y_synth, y_real_rep])

rng = np.random.RandomState(cfg.TRAIN["seed"])
idx = rng.permutation(len(X_combined))
X_combined = X_combined[idx]
y_combined  = y_combined[idx]

print(f"  Synthetic: {len(X_synth):,} | Real×{cfg.FINETUNE['real_repeat']}: {len(X_real_rep):,}")
print(f"  Total training: {len(X_combined):,} | Validation: {len(X_ft_val):,}")

BS = cfg.FINETUNE["batch_size"]
ft_train_loader = DataLoader(CSIDataset(X_combined, y_combined), batch_size=BS, shuffle=True)
ft_val_loader   = DataLoader(CSIDataset(X_ft_val, y_ft_val), batch_size=BS * 2, shuffle=False)

ft_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.FINETUNE["lr"], weight_decay=1e-4)
ft_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    ft_optimizer, mode="min", factor=0.7, patience=5, min_lr=1e-7)
ft_loss_fn = nn.HuberLoss(delta=0.1)

best_val, best_state_ft, best_epoch_ft, patience_c = float("inf"), None, 0, 0

# ← MUST be initialized here, before the loop
train_losses, val_losses = [], []

print(f"\n{'Ep':>4} {'Train':>10} {'Val':>10}  *")
print("-" * 32)

for epoch in range(1, 100):
    model.train()
    tr = 0.0
    for xb, yb in ft_train_loader:
        xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
        ft_optimizer.zero_grad()
        loss = ft_loss_fn(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ft_optimizer.step()
        tr += loss.item() * xb.size(0)
    tr /= len(ft_train_loader.dataset)

    model.eval()
    vl = 0.0
    with torch.no_grad():
        for xb, yb in ft_val_loader:
            xb, yb = xb.to(cfg.DEVICE), yb.to(cfg.DEVICE)
            vl += ft_loss_fn(model(xb), yb).item() * xb.size(0)
    vl /= len(ft_val_loader.dataset)

    ft_scheduler.step(vl)

    # ← Append losses every epoch
    train_losses.append(tr)
    val_losses.append(vl)

    is_best = vl < best_val
    if is_best:
        best_val, best_epoch_ft, patience_c = vl, epoch, 0
        best_state_ft = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_c += 1
        if patience_c >= cfg.FINETUNE["patience"]:
            print(f"\nEarly stopping at epoch {epoch} (best: {best_epoch_ft})")
            break
    print(f"{epoch:>4} {tr:>10.5f} {vl:>10.5f}  {'✓' if is_best else ''}")

# ── Step 9: Evaluate ──────────────────────────────────────────
# ← Must happen BEFORE plots (preds/trues/rmse/r2/bias defined here)
model.load_state_dict(best_state_ft)
model.eval()

all_preds, all_trues = [], []
with torch.no_grad():
    for xb, yb in ft_val_loader:
        all_preds.append(model(xb.to(cfg.DEVICE)).cpu().numpy())
        all_trues.append(yb.numpy())

preds = np.concatenate(all_preds)
trues = np.concatenate(all_trues)
rmse  = np.sqrt(((preds - trues) ** 2).mean())
mae   = np.abs(preds - trues).mean()
r2    = 1 - ((preds - trues) ** 2).sum() / ((trues - trues.mean()) ** 2).sum()
bias  = (preds - trues).mean()

# ← Baselines defined here, used in plots below
b_a1  = X_ft_val[:, cfg.MODEL["center"], 0]
b_a2  = X_ft_val[:, cfg.MODEL["center"], 2]
b_avg = 0.5 * (b_a1 + b_a2)

print(f"\nBaselines:  {ANCHOR1.upper()}={np.sqrt(((b_a1 - y_ft_val)**2).mean()):.4f}  "
      f"{ANCHOR2.upper()}={np.sqrt(((b_a2 - y_ft_val)**2).mean()):.4f}  "
      f"avg={np.sqrt(((b_avg - y_ft_val)**2).mean()):.4f}")
print(f"\n{'='*50}")
print(f"RESULTS:  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  Bias={bias:+.4f}")
print(f"Improvement vs {ANCHOR1.upper()} direct: "
      f"{(baseline_rmse - rmse) / baseline_rmse * 100:+.1f}%")

# ── Two-panel fine-tuning diagnostic plot (white background) ──
# Requires: train_losses, val_losses, best_epoch_ft,
#           preds, trues, rmse, r2  (all defined in Step 9)

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    f"Fine-Tuning Diagnostics — Station P2  |  V2 14-Feature Transformer",
    fontsize=13, fontweight="bold"
)

# ── Panel 1: Training curves ──────────────────────────────────
epochs_range = range(1, len(train_losses) + 1)
ax1.plot(epochs_range, train_losses, color="#1D4ED8", lw=1.8, label="Train loss")
ax1.plot(epochs_range, val_losses,   color="#DC2626", lw=1.8, label="Val loss")
ax1.axvline(best_epoch_ft, color="#D97706", lw=1.5, linestyle="--",
            label=f"Best epoch {best_epoch_ft}")
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel("Huber Loss", fontsize=11)
ax1.set_title("Training Curves", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ── Panel 2: Predicted vs True scatter ───────────────────────
ax2.scatter(trues, preds, alpha=0.45, s=14,
            color="#0891B2", edgecolors="none", label="Val samples")
lim = [0, max(float(trues.max()), float(preds.max())) * 1.05]
ax2.plot(lim, lim, color="#DC2626", lw=1.5, linestyle="--", label="Perfect (1:1)")
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel("True CSI", fontsize=11)
ax2.set_ylabel("Predicted CSI", fontsize=11)
ax2.set_title(
    f"Predicted vs True  (Val set)\nRMSE = {rmse:.4f}   R² = {r2:.4f}",
    fontsize=12, fontweight="bold"
)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()

# Save
import os
plot_path = os.path.normpath(
    os.path.join(os.path.dirname(OUT_GHI_CSV), "training_curves_v2.png")
)
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {plot_path} ✅")

# ════════════════════════════════════════════════════════════
# FINE-TUNING DIAGNOSTIC PLOTS
# ← Runs AFTER Step 9 so preds/trues/rmse/r2/bias/b_a1/b_a2 all exist
# ════════════════════════════════════════════════════════════

print("\nGenerating fine-tuning diagnostic plots...")

# Permutation feature importance (needed for Plot 4)
rng_perm   = np.random.RandomState(0)
perm_deltas = {}
model.eval()
for fi, fn in enumerate(FEAT_NAMES_V2):
    X_p  = X_ft_val.copy()
    pidx = rng_perm.permutation(len(X_p))
    X_p[:, :, fi] = X_p[pidx, :, fi]
    with torch.no_grad():
        p_p = model(torch.tensor(X_p, dtype=torch.float32).to(cfg.DEVICE)).cpu().numpy()
    perm_deltas[fn] = float(np.sqrt(((p_p - y_ft_val) ** 2).mean()) - rmse)

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0F172A")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)

# ── Plot 1: Training curves ────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor("#1E293B")
epochs_range = range(1, len(train_losses) + 1)
ax1.plot(epochs_range, train_losses, color="#38BDF8", lw=1.8, label="Train")
ax1.plot(epochs_range, val_losses,   color="#F87171", lw=1.8, label="Val")
ax1.axvline(best_epoch_ft, color="#FBBF24", lw=1.5, linestyle="--",
            label=f"Best ep {best_epoch_ft}")
ax1.set_xlabel("Epoch", color="#94A3B8")
ax1.set_ylabel("Huber Loss", color="#94A3B8")
ax1.set_title("Training Curves", color="#F8FAFC", fontsize=11, fontweight="bold")
ax1.legend(fontsize=8, facecolor="#0F172A", labelcolor="white")
ax1.tick_params(colors="#94A3B8")
for sp in ax1.spines.values(): sp.set_edgecolor("#334155")

# ── Plot 2: Predicted vs True scatter ─────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor("#1E293B")
ax2.scatter(trues, preds, alpha=0.4, s=12, color="#38BDF8", edgecolors="none")
lim = [0, max(trues.max(), preds.max()) * 1.06]
ax2.plot(lim, lim, "r--", lw=1.5, label="Perfect (1:1)")
ax2.set_xlabel("True CSI", color="#94A3B8")
ax2.set_ylabel("Predicted CSI", color="#94A3B8")
ax2.set_title(f"Predicted vs True  (Val)\nRMSE={rmse:.4f}   R²={r2:.4f}",
              color="#F8FAFC", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, facecolor="#0F172A", labelcolor="white")
ax2.tick_params(colors="#94A3B8")
for sp in ax2.spines.values(): sp.set_edgecolor("#334155")

# ── Plot 3: Residual distribution ─────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.set_facecolor("#1E293B")
residuals = preds - trues
ax3.hist(residuals, bins=25, color="#A78BFA", alpha=0.85, edgecolor="#0F172A")
ax3.axvline(0,                color="#F87171", lw=1.5, linestyle="--", label="Zero")
ax3.axvline(residuals.mean(), color="#FBBF24", lw=1.5, linestyle="--",
            label=f"Mean={residuals.mean():+.4f}")
ax3.set_xlabel("Residual  (Pred − True CSI)", color="#94A3B8")
ax3.set_ylabel("Count", color="#94A3B8")
ax3.set_title(f"Residual Distribution\nBias={bias:+.4f}   Std={residuals.std():.4f}",
              color="#F8FAFC", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8, facecolor="#0F172A", labelcolor="white")
ax3.tick_params(colors="#94A3B8")
for sp in ax3.spines.values(): sp.set_edgecolor("#334155")

# ── Plot 4: Permutation feature importance ─────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.set_facecolor("#1E293B")
sorted_items  = sorted(perm_deltas.items(), key=lambda x: x[1], reverse=True)
names_sorted  = [x[0] for x in sorted_items]
deltas_sorted = [x[1] for x in sorted_items]
bar_colors4   = ["#F87171" if d > 0.01 else "#60A5FA" if d > 0.003 else "#334155"
                 for d in deltas_sorted]
ax4.barh(names_sorted, deltas_sorted, color=bar_colors4, alpha=0.9, edgecolor="#0F172A")
ax4.axvline(0,    color="white",   lw=0.8)
ax4.axvline(0.01, color="#FBBF24", lw=1.2, linestyle="--", alpha=0.7,
            label="Importance threshold (0.01)")
ax4.set_xlabel("ΔRMSE (permutation importance)", color="#94A3B8")
ax4.set_title("Feature Importance   |   Red = important   Blue = minor   Gray = negligible",
              color="#F8FAFC", fontsize=10, fontweight="bold")
ax4.legend(fontsize=8, facecolor="#0F172A", labelcolor="white")
ax4.tick_params(colors="#94A3B8", axis="y", labelsize=9)
ax4.tick_params(colors="#94A3B8", axis="x")
for sp in ax4.spines.values(): sp.set_edgecolor("#334155")

# ── Plot 5: Model vs Baselines bar chart ──────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.set_facecolor("#1E293B")
model_names = [f"{ANCHOR1.upper()}\ndirect", f"{ANCHOR2.upper()}\ndirect",
               "Anchor\naverage", "Transformer\nV2"]
model_rmses = [
    float(np.sqrt(((b_a1  - y_ft_val) ** 2).mean())),
    float(np.sqrt(((b_a2  - y_ft_val) ** 2).mean())),
    float(np.sqrt(((b_avg - y_ft_val) ** 2).mean())),
    float(rmse),
]
bar_cols5 = ["#475569", "#475569", "#60A5FA", "#34D399"]
bars5     = ax5.bar(model_names, model_rmses, color=bar_cols5,
                    alpha=0.9, edgecolor="#0F172A", width=0.6)
for bar, val in zip(bars5, model_rmses):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.001,
             f"{val:.4f}", ha="center", va="bottom",
             color="white", fontsize=9, fontweight="bold")
ax5.set_ylabel("RMSE (CSI)", color="#94A3B8")
ax5.set_title("Model vs Baselines", color="#F8FAFC", fontsize=10, fontweight="bold")
ax5.tick_params(colors="#94A3B8")
for sp in ax5.spines.values(): sp.set_edgecolor("#334155")

fig.suptitle(
    f"Fine-Tuning Results — Station P2 (46.78°N, 119.23°W)   |   V2 14-Feature Transformer",
    color="white", fontsize=13, fontweight="bold", y=0.98)

plt.savefig(OUT_PLOT_FT, dpi=130, bbox_inches="tight", facecolor="#0F172A")
plt.close()
print(f"  Saved: {OUT_PLOT_FT} ✅")

# ── Step 10: Impute missing period ────────────────────────────
print(f"\n[10] Imputing {IMP_START} → {IMP_END}...")
imp_df = ft_master[
    (ft_master["datetime"] >= pd.Timestamp(IMP_START, tz="UTC")) &
    (ft_master["datetime"] <  pd.Timestamp(IMP_END,   tz="UTC"))
].reset_index(drop=True)

X_imp, dt_imp = build_finetune_sequences(
    imp_df, ANCHOR1, ANCHOR2,
    seq_len=cfg.MODEL["seq_len"],
    center=cfg.MODEL["center"],
    has_target=False)

X_imp_fixed, n_fixed = fix_missing_anchor(X_imp, center=cfg.MODEL["center"])

model.eval()
imp_preds_list = []
with torch.no_grad():
    for xb in DataLoader(torch.tensor(X_imp_fixed, dtype=torch.float32), batch_size=512):
        imp_preds_list.append(model(xb.to(cfg.DEVICE)).cpu().numpy())

imp_csi = np.concatenate(imp_preds_list).clip(0, 2.0)
imp_dt  = pd.DatetimeIndex(dt_imp).tz_localize("UTC")

site_new = pvlib.location.Location(NEW_LAT, NEW_LON, tz="UTC", altitude=NEW_ALT)
cs_new   = site_new.get_clearsky(imp_dt, model="ineichen")
sp_new   = site_new.get_solarposition(imp_dt)

imp_result = pd.DataFrame({
    "datetime":    imp_dt.tz_localize(None),
    "CSI_imputed": imp_csi,
    "GHI_clear":   cs_new["ghi"].values,
    "solar_elev":  sp_new["apparent_elevation"].values,
    "source":      "imputed_synth_ft_v2",
})
imp_result["GHI_imputed"] = imp_result["CSI_imputed"] * imp_result["GHI_clear"]
imp_result.loc[imp_result["solar_elev"] <= 1, ["GHI_imputed", "CSI_imputed"]] = 0.0

# ── Step 11: Combine + gap fill ───────────────────────────────
print("\n[11] Building full year...")
real_part = st_new.copy()
real_part["datetime"]    = real_part["datetime"].dt.tz_localize(None)
real_part["CSI_imputed"] = real_part["CSI"].fillna(0.0)
real_part["GHI_imputed"] = real_part["GHI"].fillna(0.0)
real_part["source"]      = "measured"
real_part = real_part[["datetime", "CSI_imputed", "GHI_imputed", "GHI_clear", "solar_elev", "source"]]

full_year = pd.concat([imp_result, real_part], ignore_index=True
                      ).sort_values("datetime").reset_index(drop=True)

full_spine = pd.DataFrame({"datetime": pd.date_range(
    full_year["datetime"].min(), full_year["datetime"].max(), freq="30min")})
fy = pd.merge(full_spine, full_year, on="datetime", how="left")
missing_mask = fy["source"].isna()
n_missing    = missing_mask.sum()
print(f"  Missing timestamps to gap-fill: {n_missing}")

if n_missing > 0:
    missing_dt = pd.DatetimeIndex(fy.loc[missing_mask, "datetime"]).tz_localize("UTC")
    cs_gap = site_new.get_clearsky(missing_dt, model="ineichen")
    sp_gap = site_new.get_solarposition(missing_dt)
    gap_df = pd.DataFrame({"datetime": missing_dt})
    gap_df = pd.merge_asof(
        gap_df.sort_values("datetime"),
        ft_master[["datetime", "CSI_s2", "GHI_clear_s2"]].sort_values("datetime"),
        on="datetime", direction="nearest", tolerance=pd.Timedelta("16min"))
    gap_csi = gap_df["CSI_s2"].clip(lower=0).fillna(0.0).values
    gap_ghi = gap_csi * cs_gap["ghi"].values
    night   = sp_gap["apparent_elevation"].values <= 1
    gap_ghi[night] = 0.0
    gap_csi[night] = 0.0
    fy.loc[missing_mask, "GHI_imputed"]  = gap_ghi
    fy.loc[missing_mask, "CSI_imputed"]  = gap_csi
    fy.loc[missing_mask, "GHI_clear"]    = cs_gap["ghi"].values
    fy.loc[missing_mask, "solar_elev"]   = sp_gap["apparent_elevation"].values
    fy.loc[missing_mask, "source"]       = "gap_fill_s2_copy"
    n_day = (sp_gap["apparent_elevation"].values > 5).sum()
    print(f"  Daytime gap-filled: {n_day} | Nighttime: {n_missing - n_day}")

# ── Step 12: Convert UTC → PST and save ──────────────────────
print("\n[12] Converting timestamps UTC → PST and saving...")

fy_save = fy.copy()

# datetime column is UTC stored as naive (master grid starts 2024-01-01 08:00 UTC
# which is 2024-01-01 00:00 PST). Subtract 8h to get PST naive.
fy_save["datetime"] = pd.to_datetime(fy_save["datetime"]) - pd.Timedelta(hours=8)

# Verification — solar noon on June 10 should be near 12:00 PST
june10 = fy_save[
    (pd.to_datetime(fy_save["datetime"]).dt.month == 6) &
    (pd.to_datetime(fy_save["datetime"]).dt.day   == 10) &
    (fy_save["solar_elev"] > 5)
]
if len(june10) > 0:
    peak_row  = june10.loc[june10["GHI_imputed"].idxmax()]
    peak_hour = (pd.to_datetime(peak_row["datetime"]).hour +
                 pd.to_datetime(peak_row["datetime"]).minute / 60)
    status = "✅ correct" if 11.0 < peak_hour < 14.0 else "❌ still wrong"
    print(f"  June 10 peak GHI at {peak_hour:.1f} PST  |  {status}")

# Quick check on April 30 — noon should be ~12:00 PST
apr30 = fy_save[
    (pd.to_datetime(fy_save["datetime"]).dt.month == 4) &
    (pd.to_datetime(fy_save["datetime"]).dt.day   == 30) &
    (fy_save["GHI_imputed"] > 5)
]
if len(apr30) > 0:
    first = pd.to_datetime(apr30["datetime"].iloc[0]).strftime("%H:%M")
    last  = pd.to_datetime(apr30["datetime"].iloc[-1]).strftime("%H:%M")
    peak_apr = apr30.loc[apr30["GHI_imputed"].idxmax(), "datetime"]
    print(f"  Apr 30 daylight: {first} – {last} PST  "
          f"| peak {pd.to_datetime(peak_apr).strftime('%H:%M')} PST"
          f"  (expected daylight ~05:30–20:30)")

fy_save.to_csv(OUT_GHI_CSV, index=False)