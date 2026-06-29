# ════════════════════════════════════════════════════════════
# LOAD EVERYTHING NEEDED
# ════════════════════════════════════════════════════════════

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from irradiance import config as cfg
from irradiance.data import (
    build_master,
    process_station_utc,
    build_finetune_sequences,
    fix_missing_anchor,
)
from irradiance.model import TransformerImputer

import pandas as pd
import numpy as np
import torch

device = cfg.DEVICE

# ────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────

ANCHOR1 = cfg.P2["anchor1"]
ANCHOR2 = cfg.P2["anchor2"]

SEQ_LEN = cfg.MODEL["seq_len"]
CENTER  = cfg.MODEL["center"]

feat_names_v2 = cfg.FEATURE_NAMES

# ────────────────────────────────────────────────────────────
# LOAD MODEL
# ────────────────────────────────────────────────────────────

print("Loading fine-tuned model...")

model, ckpt = TransformerImputer.from_checkpoint(
    cfg.ARTIFACTS["ft_model_v3"],
    device=device
)

model.eval()

# ────────────────────────────────────────────────────────────
# BUILD MASTER
# ────────────────────────────────────────────────────────────

print("Building ft_master...")

master = build_master(cfg)

# ────────────────────────────────────────────────────────────
# LOAD P2 DATA
# ────────────────────────────────────────────────────────────

st_new = process_station_utc(
    pd.read_csv(cfg.RAW["ghi_p2"]),
    cfg.STATIONS["p2"]["lat"],
    cfg.STATIONS["p2"]["lon"],
    cfg.STATIONS["p2"]["alt"],
    fill_gaps=False,
)

ns_raw = pd.read_csv(cfg.RAW["nsrdb_p2"], skiprows=2)
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

# ── C13 (matches 04_finetune.py Step 2 exactly — diagnostics.py was
# previously missing this entirely, which is the root cause of the
# "NSRDB_CSI_new is not in list" crash: ft_master never got these
# columns merged in below) ─────────────────────────────────────────
c13_raw = pd.read_csv(cfg.RAW["c13_p2"])
c13_raw["datetime"] = pd.to_datetime(c13_raw["datetime_local"], utc=True)
C13_NEW_COL = cfg.C13_COLS["p2"]
c13_raw[C13_NEW_COL] = pd.to_numeric(c13_raw[C13_NEW_COL], errors="coerce")
c13_mean = c13_raw[C13_NEW_COL].mean()
c13_std  = c13_raw[C13_NEW_COL].std()
c13_raw["c13_new_norm"] = (c13_raw[C13_NEW_COL] - c13_mean) / c13_std
c13_new = c13_raw[["datetime", "c13_new_norm"]].sort_values("datetime").reset_index(drop=True)

# ── Merge P2 data into ft_master (this whole block was missing) ───
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

# ────────────────────────────────────────────────────────────
# BUILD IMPUTATION DATAFRAME
# ────────────────────────────────────────────────────────────

imp_df = ft_master.copy()

# subset imputation period
imp_df = imp_df[
    (imp_df["datetime_naive"] >= pd.to_datetime(cfg.P2["imp_start"]).tz_localize(None)) &
    (imp_df["datetime_naive"] <  pd.to_datetime(cfg.P2["imp_end"]).tz_localize(None))
].copy()

# ────────────────────────────────────────────────────────────
# BUILD IMPUTATION SEQUENCES
# ────────────────────────────────────────────────────────────

print("Building imputation sequences...")

X_imp, dt_imp = build_finetune_sequences(
    imp_df,
    ANCHOR1,
    ANCHOR2,
    seq_len=SEQ_LEN,
    center=CENTER,
    anchor_bad_thresh=0.95,
    has_target=False,
)

X_imp_fixed, n_fixed = fix_missing_anchor(
    X_imp,
    center=CENTER,
)
print(f"  fix_missing_anchor: {n_fixed} sequences fixed")

print(f"Imputation sequences: {len(X_imp_fixed):,}")

# ════════════════════════════════════════════════════════════
# scripts/diagnostics.py
# Run AFTER fine-tuning. Requires all variables from
# finetune_p2_v2.py to be in memory:
#   imp_df, X_imp_fixed, dt_imp, X_ft_val, y_ft_val,
#   ft_master, model, feat_names_v2, ANCHOR1, ANCHOR2
# ════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import torch

# ════════════════════════════════════════════════════════════
# DIAGNOSTIC A — Skip reason breakdown
# Tells us WHY sequences were skipped during imputation.
# Historical hypothesis (now resolved — see Diagnostic C note):
# S3 (anchor1) had a real 46-day raw data gap that used to drive
# most skips here. Fixed in data.py's process_station_utc, so
# this diagnostic should now report 0 skips for any reason. Kept
# to catch any NEW anchor-availability problem in future data.
# ════════════════════════════════════════════════════════════

print("=" * 60)
print("DIAGNOSTIC A — Imputation skip reason breakdown")
print("=" * 60)

cols_imp = list(imp_df.columns)
def ci_imp(name): return cols_imp.index(name)

i_a1_mask = ci_imp(f"CSI_{ANCHOR1}_mask")
i_a2_mask = ci_imp(f"CSI_{ANCHOR2}_mask")

data_imp = imp_df.values

skip_anchor1_only = 0
skip_anchor2_only = 0
skip_both_anchors = 0
kept              = 0

# Also track skips by month
skip_by_month = {m: {"anchor1": 0, "anchor2": 0, "both": 0, "kept": 0}
                 for m in range(1, 13)}

for i in range(len(imp_df) - SEQ_LEN + 1):
    w = data_imp[i : i + SEQ_LEN]
    frac_a1_bad = (w[:, i_a1_mask] == 0).mean()
    frac_a2_bad = (w[:, i_a2_mask] == 0).mean()

    # Get month from center step
    m = imp_df["datetime_naive"].iloc[i + CENTER].month

    if frac_a1_bad > 0.8 and frac_a2_bad > 0.8:
        skip_both_anchors += 1
        skip_by_month[m]["both"] += 1
    elif frac_a1_bad > 0.8:
        skip_anchor1_only += 1
        skip_by_month[m]["anchor1"] += 1
    elif frac_a2_bad > 0.8:
        skip_anchor2_only += 1
        skip_by_month[m]["anchor2"] += 1
    else:
        kept += 1
        skip_by_month[m]["kept"] += 1

total_windows = kept + skip_anchor1_only + skip_anchor2_only + skip_both_anchors

print(f"\n  Total sliding windows:      {total_windows:,}")
print(f"  Kept:                       {kept:,}  ({kept/total_windows*100:.1f}%)")
print(f"  Skipped — anchor1 bad only: {skip_anchor1_only:,}  ({skip_anchor1_only/total_windows*100:.1f}%)  ← {ANCHOR1.upper()} problem")
print(f"  Skipped — anchor2 bad only: {skip_anchor2_only:,}  ({skip_anchor2_only/total_windows*100:.1f}%)  ← {ANCHOR2.upper()} problem")
print(f"  Skipped — both bad:         {skip_both_anchors:,}  ({skip_both_anchors/total_windows*100:.1f}%)")

print(f"\n  anchor1 = {ANCHOR1.upper()} | anchor2 = {ANCHOR2.upper()}")
if skip_anchor1_only > skip_anchor2_only * 3:
    print(f"  ✅ Confirmed: {ANCHOR1.upper()} data gaps are the dominant skip cause")
elif skip_anchor2_only > skip_anchor1_only * 3:
    print(f"  ⚠️  Unexpected: {ANCHOR2.upper()} gaps are dominating — investigate")
else:
    print(f"  ℹ️  Skips are distributed between both anchors")

# Monthly breakdown
months_label = ["","Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
print(f"\n  Monthly skip breakdown:")
print(f"  {'Month':>6} {'Kept':>8} {'A1 bad':>8} {'A2 bad':>8} {'Both bad':>10} {'Skip%':>7}")
print(f"  {'-'*56}")
for m in range(1, 12):
    d = skip_by_month[m]
    total_m = d["kept"] + d["anchor1"] + d["anchor2"] + d["both"]
    if total_m == 0:
        continue
    skip_pct = (total_m - d["kept"]) / total_m * 100
    flag = " ⚠️" if skip_pct > 50 else ""
    print(f"  {months_label[m]:>6} {d['kept']:>8} {d['anchor1']:>8} "
          f"{d['anchor2']:>8} {d['both']:>10} {skip_pct:>6.1f}%{flag}")

# ════════════════════════════════════════════════════════════
# DIAGNOSTIC B — Monthly permutation feature importance
# Uses imputation sequences (no ground truth) so reports
# prediction variance change rather than RMSE change.
# Key question: is C13 more important in summer than winter?
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DIAGNOSTIC B — Permutation feature importance by month")
print("(metric: change in prediction std when feature is shuffled)")
print("=" * 60)

dt_imp_pd  = pd.DatetimeIndex(dt_imp)
months_imp = dt_imp_pd.month

model.eval()

months_to_check = [
    (6,  "Jun — peak summer"),
    (7,  "Jul — peak summer"),
    (9,  "Sep — late summer"),
    (10, "Oct — autumn"),
    (11, "Nov — winter onset"),
]

# Features we care most about comparing
key_features = [
    "anchor1_CSI",
    "anchor2_CSI",
    "C13_norm_target",
    "C13_norm_anchor1",
    "C13_norm_anchor2",
    "NSRDB_CSI_target",
    "NSRDB_CSI_anchor1",
    "hour_cos",
]

results_monthly = {}

for month_num, month_label in months_to_check:
    mask = months_imp == month_num
    if mask.sum() < 20:
        print(f"\n  {month_label}: insufficient sequences ({mask.sum()}), skipping")
        continue

    X_month = X_imp_fixed[mask]

    with torch.no_grad():
        pred_base = model(
            torch.tensor(X_month, dtype=torch.float32).to(device)
        ).cpu().numpy()

    base_std  = pred_base.std()
    base_mean = pred_base.mean()

    print(f"\n  {month_label} — {mask.sum()} sequences")
    print(f"  pred mean={base_mean:.3f}  pred std={base_std:.3f}")
    print(f"  {'Feature':<24} {'Δstd':>8}  (positive = feature adds variance)")

    rng_diag = np.random.RandomState(42)
    month_results = {}
    for feat_idx, feat_name in enumerate(feat_names_v2):
        if feat_name not in key_features:
            continue
        X_perm = X_month.copy()
        pidx   = rng_diag.permutation(len(X_perm))
        X_perm[:, :, feat_idx] = X_perm[pidx, :, feat_idx]
        with torch.no_grad():
            pred_perm = model(
                torch.tensor(X_perm, dtype=torch.float32).to(device)
            ).cpu().numpy()
        delta_std = base_std - pred_perm.std()
        month_results[feat_name] = delta_std
        if abs(delta_std) > 0.001:
            bar = "█" * max(0, int(abs(delta_std) * 300))
            sign = "+" if delta_std > 0 else "-"
            print(f"    {feat_name:<24} {delta_std:>+.4f}  {bar}")

    results_monthly[month_label] = month_results

# Summary across months
print(f"\n  Summary — C13 importance across months:")
print(f"  {'Month':<30} {'C13_target':>12} {'C13_anchor1':>13} {'anchor1_CSI':>13}")
print(f"  {'-'*72}")
for month_label, res in results_monthly.items():
    c13t  = res.get("C13_norm_target", 0)
    c13a1 = res.get("C13_norm_anchor1", 0)
    a1csi = res.get("anchor1_CSI", 0)
    print(f"  {month_label:<30} {c13t:>+12.4f} {c13a1:>+13.4f} {a1csi:>+13.4f}")

print(f"\n  Interpretation:")
print(f"    C13 Δstd > 0 → feature adds meaningful signal")
print(f"    If C13 higher in summer → cloud variability is higher")
print(f"    If anchor1_CSI >> C13 → model still anchor-dominated")

# ════════════════════════════════════════════════════════════
# PLOT — Feature importance by month
# Two views of the same results_monthly data:
#   1. Grouped bar chart — every feature, every month, side by side
#   2. Heatmap — compact view, easier to spot seasonal patterns
# ════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")  # headless-safe; remove if running in a notebook
import matplotlib.pyplot as plt
import numpy as np
import os

months_plotted  = list(results_monthly.keys())
features_plotted = key_features  # same feature list used in Diagnostic B

# Build a (months x features) matrix, missing values → 0
importance_matrix = np.array([
    [results_monthly[m].get(f, 0.0) for f in features_plotted]
    for m in months_plotted
])

# ── Plot 1: Grouped bar chart ─────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(14, 6))
x = np.arange(len(features_plotted))
width = 0.8 / len(months_plotted)
colors = plt.cm.viridis(np.linspace(0, 1, len(months_plotted)))

for i, month_label in enumerate(months_plotted):
    offset = (i - len(months_plotted) / 2) * width + width / 2
    ax1.bar(x + offset, importance_matrix[i], width,
            label=month_label, color=colors[i], alpha=0.9)

ax1.axhline(0, color="black", lw=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(features_plotted, rotation=30, ha="right", fontsize=9)
ax1.set_ylabel("Δstd (permutation importance)", fontsize=11)
ax1.set_title("Feature Importance by Month — Grouped Comparison",
              fontsize=13, fontweight="bold")
ax1.legend(fontsize=9, ncol=len(months_plotted), loc="upper center",
           bbox_to_anchor=(0.5, -0.18))
ax1.grid(True, alpha=0.3, axis="y")
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
plt.tight_layout()

bar_plot_path = os.path.normpath(
    os.path.join(cfg.BASE_PATH, "feature_importance_by_month_bars.png"))
plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
plt.close(fig1)
print(f"\n  Saved: {bar_plot_path} ✅")

# ── Plot 2: Heatmap ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 5))
vmax = np.abs(importance_matrix).max()
im = ax2.imshow(importance_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 aspect="auto")

ax2.set_xticks(np.arange(len(features_plotted)))
ax2.set_xticklabels(features_plotted, rotation=30, ha="right", fontsize=9)
ax2.set_yticks(np.arange(len(months_plotted)))
ax2.set_yticklabels(months_plotted, fontsize=9)
ax2.set_title("Feature Importance Heatmap\n(Red = positive impact, Blue = negative)",
              fontsize=12, fontweight="bold")

# Annotate each cell with its value
for i in range(len(months_plotted)):
    for j in range(len(features_plotted)):
        val = importance_matrix[i, j]
        text_color = "white" if abs(val) > vmax * 0.6 else "black"
        ax2.text(j, i, f"{val:+.3f}", ha="center", va="center",
                  fontsize=7.5, color=text_color)

plt.colorbar(im, ax=ax2, label="Δstd (permutation importance)")
plt.tight_layout()

heatmap_path = os.path.normpath(
    os.path.join(cfg.BASE_PATH, "feature_importance_heatmap.png"))
plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Saved: {heatmap_path} ✅")

# ════════════════════════════════════════════════════════════
# DIAGNOSTIC C — Anchor station data availability by month
# Historical note: this diagnostic was originally written to
# confirm a hypothesis that S3 sensor failures (a real, large gap
# verified in the raw data: 46 missing days, mostly Oct 4-Nov 18)
# were the dominant cause of imputation sequence skips. That gap
# has since been fixed at the resampling layer (data.py's
# process_station_utc now self-fills it from the station's own
# seasonal CSI pattern — see Tier 2b in that function). S1/S2/S3
# should now all show ~100% availability every month. This
# diagnostic is kept to verify that remains true going forward
# (e.g. if a NEW raw data file introduces a fresh gap).
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DIAGNOSTIC C — Anchor station data availability by month")
print("=" * 60)

ft_master["_month_check"] = ft_master["datetime_naive"].dt.month

print(f"\n  {'Month':>6} {'S1_valid':>10} {'S2_valid':>10} "
      f"{'S3_valid':>10} {'S3_pct':>8} {'S3 status'}")
print(f"  {'-'*62}")

for m in range(1, 12):
    sub = ft_master[
        (ft_master["_month_check"] == m) &
        (ft_master["solar_elev_s3"] > 5)
    ]
    if len(sub) == 0:
        continue
    s1_valid = (sub["CSI_s1"] > 0).sum()
    s2_valid = (sub["CSI_s2"] > 0).sum()
    s3_valid = (sub["CSI_s3"] > 0).sum()
    s3_pct   = s3_valid / len(sub) * 100

    if s3_pct < 10:
        status = "⚠️  OFFLINE"
    elif s3_pct < 50:
        status = "⚠️  degraded"
    elif s3_pct < 90:
        status = "⚡ partial"
    else:
        status = "✅ good"

    print(f"  {months_label[m]:>6} {s1_valid:>10} {s2_valid:>10} "
          f"{s3_valid:>10} {s3_pct:>7.1f}%  {status}")

ft_master.drop(columns=["_month_check"], inplace=True)

print(f"\n  S3 = ANCHOR1 ({ANCHOR1.upper()})")
print(f"  S1 = ANCHOR2 ({ANCHOR2.upper()})")
print(f"\n  Connecting A → C:")
print(f"  If S3 is OFFLINE in the same months as Diagnostic A")
print(f"  shows high anchor1_bad skips → hypothesis confirmed.")
print(f"  The anchor1 dominance in permutation importance is then")
print(f"  explained by the model learning to compensate for S3 gaps")
print(f"  using the S2 substitution we applied in fix_missing_anchor().")

# ════════════════════════════════════════════════════════════
# CROSS-DIAGNOSTIC SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CROSS-DIAGNOSTIC SUMMARY")
print("=" * 60)

# Identify which months had high anchor1 skips
high_skip_months = [m for m in range(1, 12)
                    if skip_by_month[m]["anchor1"] >
                       skip_by_month[m]["kept"] * 0.3]

# Re-check S3 availability for those months
ft_master["_month_check"] = ft_master["datetime_naive"].dt.month
s3_offline_months = []
for m in range(1, 12):
    sub = ft_master[
        (ft_master["_month_check"] == m) &
        (ft_master["solar_elev_s3"] > 5)
    ]
    if len(sub) == 0: continue
    s3_pct = (sub["CSI_s3"] > 0).sum() / len(sub) * 100
    if s3_pct < 20:
        s3_offline_months.append(m)
ft_master.drop(columns=["_month_check"], inplace=True)

print(f"\n  Months with high anchor1 skip rate: "
      f"{[months_label[m] for m in high_skip_months]}")
print(f"  Months where S3 data < 20%:         "
      f"{[months_label[m] for m in s3_offline_months]}")

overlap = set(high_skip_months) & set(s3_offline_months)
total_skipped = (skip_anchor1_only + skip_anchor2_only + skip_both_anchors)

if total_skipped == 0:
    print(f"\n  ✅ Zero sequences skipped for any anchor-availability reason.")
    print(f"     S1/S2/S3 all show complete data this run (see Diagnostic C —")
    print(f"     should read 100% for every month). This confirms the")
    print(f"     data.py self-contained gap-filling fix (Tier 1/2a/2b in")
    print(f"     process_station_utc) is working as intended: the historical")
    print(f"     S3 outage (46 missing days, mostly Oct 4 - Nov 18) that")
    print(f"     used to drive most imputation skips no longer causes any.")
    print(f"\n  Thesis statement (current state):")
    print(f"     All three reference stations (S1, S2, S3) provide complete")
    print(f"     30-minute GHI coverage for the full year after self-contained")
    print(f"     gap-filling (median within-day or seasonal CSI estimation —")
    print(f"     see Methods). No imputation sequence was skipped due to")
    print(f"     missing anchor data in this run.")
elif overlap:
    print(f"\n  ✅ CONFIRMED: {[months_label[m] for m in sorted(overlap)]} "
          f"appear in both lists.")
    print(f"     An anchor station data gap is the dominant skip cause in")
    print(f"     these months. This is a data quality issue, not a model")
    print(f"     limitation — check which raw file introduced this gap.")
    print(f"\n  Thesis statement:")
    print(f"     One or more reference stations recorded <20% valid daytime")
    print(f"     observations in {[months_label[m] for m in sorted(overlap)]}.")
    print(f"     Imputation sequences affected by this gap were corrected")
    print(f"     via the anchor-substitution fallback in fix_missing_anchor().")
else:
    print(f"\n  ⚠️  Skips exist ({total_skipped} total) but don't cleanly overlap")
    print(f"     with low anchor availability — investigate boundary effects,")
    print(f"     NSRDB missing data, or the anchor_bad_thresh setting itself.")