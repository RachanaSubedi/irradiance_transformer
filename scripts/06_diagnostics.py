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
    process_nsrdb_utc,
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
    cfg.ARTIFACTS["ft_model_v2"],
    device=device
)

model.eval()

# ────────────────────────────────────────────────────────────
# BUILD MASTER
# ────────────────────────────────────────────────────────────

print("Building ft_master...")

ft_master = build_master(cfg)

# ────────────────────────────────────────────────────────────
# LOAD P2 DATA
# ────────────────────────────────────────────────────────────

st_new = process_station_utc(
    pd.read_csv(cfg.RAW["ghi_p2"]),
    cfg.STATIONS["p2"]["lat"],
    cfg.STATIONS["p2"]["lon"],
    cfg.STATIONS["p2"]["alt"],
)

ns_new = process_nsrdb_utc(
    pd.read_csv(cfg.RAW["nsrdb_p2"], skiprows=2),
    "new"
)

# ────────────────────────────────────────────────────────────
# BUILD IMPUTATION DATAFRAME
# ────────────────────────────────────────────────────────────

imp_df = ft_master.copy()

# subset imputation period
imp_df = imp_df[
    (imp_df["datetime_naive"] >= cfg.P2["imp_start"]) &
    (imp_df["datetime_naive"] <  cfg.P2["imp_end"])
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

X_imp_fixed = fix_missing_anchor(
    X_imp,
    feat_names_v2,
    anchor_name=ANCHOR1
)

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
# Hypothesis: S3 (anchor1) missing in Oct-Nov is the cause.
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
    (10, "Oct — autumn, S3 gaps"),
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
# DIAGNOSTIC C — Anchor station data availability by month
# Confirms whether S3 gaps explain Diagnostic A findings.
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
if overlap:
    print(f"\n  ✅ CONFIRMED: {[months_label[m] for m in sorted(overlap)]} "
          f"appear in both lists.")
    print(f"     S3 sensor failure directly causes the gap fills.")
    print(f"     This is a data quality issue, not a model limitation.")
    print(f"\n  Thesis statement:")
    print(f"     Station S3 recorded <20% valid daytime observations in "
          f"{[months_label[m] for m in sorted(overlap)]}.")
    print(f"     Imputation sequences with missing anchor1 were corrected")
    print(f"     by substituting anchor2 (S1) values — consistent with")
    print(f"     nearest-neighbor fallback used for other gap periods.")
else:
    print(f"\n  ⚠️  No strong overlap found — gap fills may have another cause.")
    print(f"     Investigate boundary effects and NSRDB missing data.")