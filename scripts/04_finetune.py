# ════════════════════════════════════════════════════════════
# scripts/04_finetune.py
# Fine-tune pretrained v3 model on partial station P2.
# Requires pretrain_best_model_v3.pt to already exist.
# Outputs full-year GHI CSV + fine-tuned model checkpoint.
#
# v3 changes from v2 (this audit, targeting the GHI ceiling issue):
#   1. Snapshot resampling upstream (data.py) instead of mean —
#      restores true peak GHI signal lost to averaging.
#   2. Tail-weighted MSE loss instead of plain HuberLoss — Huber's
#      robustness-to-outliers behavior was actively suppressing
#      high-CSI (clear-sky peak) predictions, since those samples
#      are a small, high-variance minority of the training set.
#   3. Consolidated to ONE diagnostic plot block (was duplicated).
#   4. Fixed timestamp verification to check solar_elev peak
#      (purely geometric) instead of GHI peak (cloud-dependent,
#      caused false-alarm "still wrong" messages on cloudy days).
#   5. All paths go through cfg._p() / os.path.normpath — fixes
#      the recurring PermissionError from mixed path separators.
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from irradiance import config as cfg
from irradiance.data import (
    build_master,
    CSIDataset,
    process_station_utc,
    build_finetune_sequences,
    fix_missing_anchor,
)
from irradiance.model import TransformerImputer

# ════════════════════════════════════════════════════════════
# STATION CONFIG
# ════════════════════════════════════════════════════════════

STATION_NAME  = "p2"
ANCHOR1       = cfg.P2["anchor1"]
ANCHOR2       = cfg.P2["anchor2"]
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

# v3 artifacts — pretrain model also retrained as v3 (see 03_pretrain.py)
PRETRAIN_MODEL = cfg.ARTIFACTS["pretrain_model_v3"]
OUT_GHI_CSV    = cfg.ARTIFACTS["ghi_csv_v3"]
OUT_MODEL_PT   = cfg.ARTIFACTS["ft_model_v3"]
OUT_PLOT_FT    = cfg.ARTIFACTS["plot_v3"]

FEAT_NAMES = cfg.FEATURE_NAMES

TAIL_THRESHOLD = cfg.FINETUNE["tail_threshold"]
TAIL_WEIGHT    = cfg.FINETUNE["tail_weight"]

# ════════════════════════════════════════════════════════════
print("=" * 60)
print(f"FINE-TUNING V3 — Station {STATION_NAME.upper()}")
print(f"Anchors: {ANCHOR1.upper()} (nearest) | {ANCHOR2.upper()} (second)")
print(f"Loss: tail-weighted MSE (threshold={TAIL_THRESHOLD}, weight={TAIL_WEIGHT})")
print("=" * 60)

# ── Step 1: Load partial station GHI ─────────────────────────
print("\n[1] Loading partial station GHI (snapshot-resampled)...")
st_new = process_station_utc(pd.read_csv(LOCAL_GHI_PATH), NEW_LAT, NEW_LON, NEW_ALT)
print(f"  Rows: {len(st_new)} | daytime CSI: {st_new['CSI'].notna().sum()}")
print(f"  Range: {st_new['datetime'].min()} → {st_new['datetime'].max()}")
print(f"  Max GHI in real P2 overlap window: {st_new['GHI'].max():.1f} W/m²")

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
print("\n[3] Building global master (snapshot-resampled S1/S2/S3)...")
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
print(f"  Train y max CSI: {y_ft_train.max():.3f} | Val y max CSI: {y_ft_val.max():.3f}")

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
print(f"  Synthetic: {len(X_synth):,} | y mean={y_synth.mean():.3f} | y max={y_synth.max():.3f}")

# ── Step 7: Load pretrained model + partial unfreeze ──────────
print("\n[7] Loading pretrained v3 model...")
model, ckpt = TransformerImputer.from_checkpoint(PRETRAIN_MODEL, cfg.DEVICE)

for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "head" in name or "encoder.layers.1" in name:
        param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable:,} / {total:,} (head + last encoder layer)")

# ── Step 8: Combine + fine-tune with TAIL-WEIGHTED loss ────────
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
n_tail = (y_combined > TAIL_THRESHOLD).sum()
print(f"  Tail samples (CSI > {TAIL_THRESHOLD}): {n_tail:,} "
      f"({n_tail/len(y_combined)*100:.1f}% of training set)")

BS = cfg.FINETUNE["batch_size"]
ft_train_loader = DataLoader(CSIDataset(X_combined, y_combined), batch_size=BS, shuffle=True)
ft_val_loader   = DataLoader(CSIDataset(X_ft_val, y_ft_val), batch_size=BS * 2, shuffle=False)

ft_optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=cfg.FINETUNE["lr"], weight_decay=1e-4)
ft_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    ft_optimizer, mode="min", factor=0.7, patience=5, min_lr=1e-7)


def tail_weighted_mse(pred, target, threshold=TAIL_THRESHOLD, tail_weight=TAIL_WEIGHT):
    """
    MSE loss with extra weight on high-CSI (clear-sky peak) samples.

    Why MSE instead of HuberLoss:
    Huber behaves like MAE for large errors, which deliberately
    DAMPENS the gradient contribution of samples the model gets
    badly wrong — exactly the behavior that was suppressing
    clear-sky peak predictions, since those are rare, high-variance
    samples that initially produce larger errors during training.
    Plain MSE keeps the quadratic penalty so the optimizer is not
    let off the hook for under-predicting the tail.

    Why add tail weighting on top of MSE:
    Even with MSE, if <0.85 CSI samples are ~93% of the training
    distribution, the average gradient is dominated by them. Tail
    weighting explicitly tells the optimizer "getting the clear-sky
    samples right matters more than their frequency alone implies."
    """
    se = (pred - target) ** 2
    weight = torch.ones_like(target)
    weight = torch.where(target > threshold,
                         torch.full_like(target, tail_weight),
                         weight)
    return (se * weight).mean()


ft_loss_fn = tail_weighted_mse

best_val, best_state_ft, best_epoch_ft, patience_c = float("inf"), None, 0, 0
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

# ── Tail-specific evaluation — DOES the fix work? ─────────────
tail_mask = trues > TAIL_THRESHOLD
if tail_mask.sum() > 0:
    tail_rmse = np.sqrt(((preds[tail_mask] - trues[tail_mask]) ** 2).mean())
    tail_bias = (preds[tail_mask] - trues[tail_mask]).mean()
    print(f"\nTail performance (CSI > {TAIL_THRESHOLD}, n={tail_mask.sum()}):")
    print(f"  Tail RMSE: {tail_rmse:.4f}  |  Tail bias: {tail_bias:+.4f}")
    print(f"  Mean true CSI in tail: {trues[tail_mask].mean():.4f}  |  "
          f"Mean pred CSI in tail: {preds[tail_mask].mean():.4f}")
    if tail_bias < -0.02:
        print(f"  ⚠️  Still under-predicting the tail by {-tail_bias:.4f} CSI on average")
    else:
        print(f"  ✅ Tail bias is small — compression issue appears resolved")
else:
    print(f"\n  No validation samples above CSI {TAIL_THRESHOLD} — "
          f"cannot directly verify tail fix on val set.")

# ════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS — single consolidated 6-panel figure
# (previously duplicated across two separate plot blocks)
# ════════════════════════════════════════════════════════════

print("\nGenerating diagnostic plots...")

rng_perm    = np.random.RandomState(0)
perm_deltas = {}
model.eval()
for fi, fn in enumerate(FEAT_NAMES):
    X_p  = X_ft_val.copy()
    pidx = rng_perm.permutation(len(X_p))
    X_p[:, :, fi] = X_p[pidx, :, fi]
    with torch.no_grad():
        p_p = model(torch.tensor(X_p, dtype=torch.float32).to(cfg.DEVICE)).cpu().numpy()
    perm_deltas[fn] = float(np.sqrt(((p_p - y_ft_val) ** 2).mean()) - rmse)

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# Plot 1: Training curves
ax1 = fig.add_subplot(gs[0, 0])
epochs_range = range(1, len(train_losses) + 1)
ax1.plot(epochs_range, train_losses, color="#1D4ED8", lw=1.8, label="Train")
ax1.plot(epochs_range, val_losses,   color="#DC2626", lw=1.8, label="Val")
ax1.axvline(best_epoch_ft, color="#D97706", lw=1.5, linestyle="--",
            label=f"Best ep {best_epoch_ft}")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Tail-weighted MSE")
ax1.set_title("Training Curves", fontweight="bold")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

# Plot 2: Predicted vs True, colored by tail membership
ax2 = fig.add_subplot(gs[0, 1])
non_tail = ~tail_mask
ax2.scatter(trues[non_tail], preds[non_tail], alpha=0.4, s=14,
            color="#0891B2", edgecolors="none", label="CSI ≤ threshold")
ax2.scatter(trues[tail_mask], preds[tail_mask], alpha=0.7, s=20,
            color="#DC2626", edgecolors="none", label="CSI > threshold (tail)")
lim = [0, max(float(trues.max()), float(preds.max())) * 1.05]
ax2.plot(lim, lim, color="black", lw=1.2, linestyle="--", label="Perfect (1:1)")
ax2.set_xlim(lim); ax2.set_ylim(lim)
ax2.set_xlabel("True CSI"); ax2.set_ylabel("Predicted CSI")
ax2.set_title(f"Predicted vs True\nRMSE={rmse:.4f}  R²={r2:.4f}", fontweight="bold")
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

# Plot 3: Residual distribution
ax3 = fig.add_subplot(gs[0, 2])
residuals = preds - trues
ax3.hist(residuals, bins=25, color="#A78BFA", alpha=0.85, edgecolor="white")
ax3.axvline(0, color="#DC2626", lw=1.5, linestyle="--", label="Zero")
ax3.axvline(residuals.mean(), color="#D97706", lw=1.5, linestyle="--",
            label=f"Mean={residuals.mean():+.4f}")
ax3.set_xlabel("Residual (Pred − True)"); ax3.set_ylabel("Count")
ax3.set_title(f"Residual Distribution\nBias={bias:+.4f}", fontweight="bold")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

# Plot 4: Feature importance
ax4 = fig.add_subplot(gs[1, 0:2])
sorted_items  = sorted(perm_deltas.items(), key=lambda x: x[1], reverse=True)
names_sorted  = [x[0] for x in sorted_items]
deltas_sorted = [x[1] for x in sorted_items]
bar_colors = ["#DC2626" if d > 0.01 else "#0891B2" if d > 0.003 else "#94A3B8"
              for d in deltas_sorted]
ax4.barh(names_sorted, deltas_sorted, color=bar_colors, alpha=0.9, edgecolor="white")
ax4.axvline(0, color="black", lw=0.8)
ax4.set_xlabel("ΔRMSE (permutation importance)")
ax4.set_title("Feature Importance", fontweight="bold")
ax4.grid(True, alpha=0.3, axis="x")
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

# Plot 5: Model vs baselines (overall + tail-specific)
ax5 = fig.add_subplot(gs[1, 2])
cats = ["Overall\nRMSE", f"Tail\nRMSE (>{TAIL_THRESHOLD})"]
vals = [rmse, tail_rmse if tail_mask.sum() > 0 else 0]
bars = ax5.bar(cats, vals, color=["#34D399", "#DC2626"], alpha=0.85, width=0.5)
for bar, val in zip(bars, vals):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
              f"{val:.4f}", ha="center", fontweight="bold")
ax5.set_ylabel("RMSE (CSI)")
ax5.set_title("Overall vs Tail Performance", fontweight="bold")
ax5.grid(True, alpha=0.3, axis="y")
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

fig.suptitle(
    f"Fine-Tuning v3 Diagnostics — Station P2 — Tail-Weighted Loss",
    fontsize=14, fontweight="bold")

plt.savefig(OUT_PLOT_FT, dpi=150, bbox_inches="tight", facecolor="white")
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
    "source":      "imputed_synth_ft_v3",
})
imp_result["GHI_imputed"] = imp_result["CSI_imputed"] * imp_result["GHI_clear"]
imp_result.loc[imp_result["solar_elev"] <= 1, ["GHI_imputed", "CSI_imputed"]] = 0.0

print(f"  Imputed max GHI: {imp_result['GHI_imputed'].max():.1f} W/m²")
print(f"  Imputed max CSI: {imp_result['CSI_imputed'].max():.3f}")

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
fy = pd.merge(full_spine, full_year, on="datetime", how="left").copy()
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

    # .copy() is required here: pandas ≥2.0 Copy-on-Write semantics can
    # return a read-only array from .values after a .clip()/.fillna()
    # chain, which raises "assignment destination is read-only" on the
    # in-place mutations below (gap_ghi[night] = 0.0 etc). This bug is
    # version-dependent — reproduces on pandas 3.0, may or may not on
    # older versions depending on CoW settings, so the explicit .copy()
    # is the version-safe fix either way.
    gap_csi = gap_df["CSI_s2"].clip(lower=0).fillna(0.0).to_numpy().copy()
    gap_ghi = (gap_csi * cs_gap["ghi"].to_numpy()).copy()
    night   = sp_gap["apparent_elevation"].to_numpy() <= 1
    gap_ghi[night] = 0.0
    gap_csi[night] = 0.0
    fy.loc[missing_mask, "GHI_imputed"]  = gap_ghi
    fy.loc[missing_mask, "CSI_imputed"]  = gap_csi
    fy.loc[missing_mask, "GHI_clear"]    = cs_gap["ghi"].to_numpy().copy()
    fy.loc[missing_mask, "solar_elev"]   = sp_gap["apparent_elevation"].to_numpy().copy()
    fy.loc[missing_mask, "source"]       = "gap_fill_s2_copy"
    n_day = (sp_gap["apparent_elevation"].to_numpy() > 5).sum()
    print(f"  Daytime gap-filled: {n_day} | Nighttime: {n_missing - n_day}")

# ── Step 12: Convert UTC → PST and save ──────────────────────
print("\n[12] Converting timestamps UTC → PST and saving...")

fy_save = fy.copy()
fy_save["datetime"] = pd.to_datetime(fy_save["datetime"]) - pd.Timedelta(hours=8)

# Verification — use solar_elev peak (purely geometric, always at true
# solar noon) NOT GHI peak (cloud-dependent, gives false alarms on
# cloudy days — this was the bug in the v2 verification).
june10 = fy_save[
    (pd.to_datetime(fy_save["datetime"]).dt.month == 6) &
    (pd.to_datetime(fy_save["datetime"]).dt.day   == 10)
]
if len(june10) > 0:
    peak_row  = june10.loc[june10["solar_elev"].idxmax()]
    peak_hour = (pd.to_datetime(peak_row["datetime"]).hour +
                 pd.to_datetime(peak_row["datetime"]).minute / 60)
    status = "✅ correct" if 11.5 < peak_hour < 12.5 else "❌ check offset"
    print(f"  Jun 10 solar noon (elev peak) at {peak_hour:.1f} PST  |  {status}")

# ── Save with normalized path (fixes recurring PermissionError) ──
out_path = os.path.normpath(OUT_GHI_CSV)
fy_save.to_csv(out_path, index=False)

torch.save({
    "model_state":    best_state_ft,
    "config":         ckpt["config"],
    "best_epoch":     best_epoch_ft,
    "val_loss":       best_val,
    "ft_rmse":        float(rmse),
    "ft_r2":          float(r2),
    "tail_rmse":      float(tail_rmse) if tail_mask.sum() > 0 else None,
    "tail_bias":      float(tail_bias) if tail_mask.sum() > 0 else None,
    "baseline_rmse":  float(baseline_rmse),
    "loss_fn":        "tail_weighted_mse",
    "tail_threshold": TAIL_THRESHOLD,
    "tail_weight":    TAIL_WEIGHT,
    "station":        {"name": STATION_NAME, "lat": NEW_LAT, "lon": NEW_LON, "alt": NEW_ALT},
    "anchors":        [ANCHOR1, ANCHOR2],
    "version":        "v3",
    "n_features":     cfg.MODEL["n_features"],
}, os.path.normpath(OUT_MODEL_PT))

# ── Summary ───────────────────────────────────────────────────
day_full = fy_save[fy_save["solar_elev"] > 5]
months_lbl = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
fy_save["month"] = pd.to_datetime(fy_save["datetime"]).dt.month
monthly = day_full.groupby(
    pd.to_datetime(day_full["datetime"]).dt.month
)["GHI_imputed"].agg(["mean", "max"])

print(f"\nFull year summary (v3, PST timestamps):")
print(f"  Total:      {len(fy_save):,}")
print(f"  Imputed:    {(fy_save['source'] == 'imputed_synth_ft_v3').sum():,}")
print(f"  Measured:   {(fy_save['source'] == 'measured').sum():,}")
print(f"  Gap-filled: {(fy_save['source'] == 'gap_fill_s2_copy').sum():,}")
print(f"  Mean daytime GHI: {day_full['GHI_imputed'].mean():.1f} W/m²")
print(f"  Max  daytime GHI: {day_full['GHI_imputed'].max():.1f} W/m²  "
      f"(v2 ceiling was ~760-780 W/m² — compare against this)")
print(f"\n  Monthly GHI:")
for m in range(1, 13):
    if m in monthly.index:
        print(f"    {months_lbl[m-1]:>4}: "
              f"mean={monthly.loc[m, 'mean']:>6.1f}  "
              f"max={monthly.loc[m, 'max']:>6.1f}")

print(f"\nSaved: {out_path} ✅")
print(f"Saved: {os.path.normpath(OUT_MODEL_PT)} ✅")
print(f"Saved: {OUT_PLOT_FT} ✅")
print("\nDone. Compare max GHI above against v2 to confirm ceiling improved.")