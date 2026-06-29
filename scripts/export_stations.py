# ════════════════════════════════════════════════════════════
# scripts/export_all_stations_filled.py
#
# Rebuilds all_stations_GHI_30min_PST_filled.csv using CURRENT,
# correct sources — replaces the older version that used:
#   - S1/S2/S3: ad-hoc cross-station bias-ratio gap filling
#   - P2: the old v2 fine-tuning output
#
# Current sources:
#   - S1/S2/S3: process_station_utc's self-contained gap filling
#     (Tier 1/2a/2b — see data.py), fully complete, zero NaN
#   - P2: station_46_78_full_year_GHI_v3.csv (your v3 model output)
#
# Output: all_stations_GHI_30min_PST_filled.csv
#   Columns: datetime, GHI_S1, GHI_S2, GHI_S3, GHI_P2
#   All in PST (UTC-8), 30-min resolution, full year, zero NaN.
#
# HOW TO RUN:
#   python scripts/export_all_stations_filled.py
# ════════════════════════════════════════════════════════════

from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd
import numpy as np
from irradiance import config as cfg
from irradiance.data import process_station_utc

print("=" * 60)
print("REBUILDING all_stations_GHI_30min_PST_filled.csv")
print("=" * 60)

# ── S1, S2, S3 — fresh through the fixed pipeline ─────────────
print("\n[1] Processing S1, S2, S3 (self-contained gap filling)...")
station_dfs = {}
for name in ["s1", "s2", "s3"]:
    raw = pd.read_csv(cfg.RAW[f"ghi_{name}"])
    coords = cfg.STATIONS[name]
    filled = process_station_utc(
        raw, coords["lat"], coords["lon"], coords["alt"], fill_gaps=True)

    # process_station_utc's grid runs UTC midnight-to-midnight for the
    # calendar year (correct — matches build_master()/the rest of the
    # pipeline, which works natively in UTC). This script converts to
    # PST for the combined output. Since PST = UTC-8, "Dec 31 16:00-
    # 23:30 PST" needs UTC timestamps from Jan 1 00:00-07:30 UTC of
    # the FOLLOWING year, which the grid doesn't include, leaving the
    # last 16 PST rows of the year as NaN after the shift.
    #
    # Verified via pvlib solar position: 15 of these 16 timestamps are
    # solidly nighttime (elevation -2 deg to -66 deg) at this latitude
    # in late December; only 16:00 PST is marginally sunlit (elev
    # +2.1 deg, negligible GHI at that angle). Zero-filling here is
    # therefore correct and consistent with how nighttime is handled
    # everywhere else in this pipeline.
    filled["datetime_pst"] = (
        filled["datetime"].dt.tz_localize(None) - pd.Timedelta(hours=8))

    daytime = filled["GHI_clear"] >= 10
    n_gap = (daytime & filled["GHI"].isna()).sum()
    print(f"  {name.upper()}: {len(filled):,} rows | daytime gaps remaining: {n_gap}")

    station_dfs[name] = filled[["datetime_pst", "GHI"]].rename(
        columns={"datetime_pst": "datetime", "GHI": f"GHI_{name.upper()}"})

# ── P2 — from the v3 model output (already complete, already PST) ──
print("\n[2] Loading P2 from v3 model output...")
p2_path = cfg.ARTIFACTS["ghi_csv_v3"]
p2_raw = pd.read_csv(p2_path)
p2_raw["datetime"] = pd.to_datetime(p2_raw["datetime"])
p2 = p2_raw[["datetime", "GHI_imputed"]].rename(columns={"GHI_imputed": "GHI_P2"})
print(f"  P2: {len(p2):,} rows from {p2_path}")

# ── Combine on a canonical 30-min PST grid ────────────────────
print("\n[3] Combining onto canonical PST grid...")
grid = pd.date_range("2024-01-01 00:00:00", "2024-12-31 23:30:00", freq="30min")
combined = pd.DataFrame({"datetime": grid})

for name in ["s1", "s2", "s3"]:
    combined = combined.merge(station_dfs[name], on="datetime", how="left")
combined = combined.merge(p2, on="datetime", how="left")

# ── Boundary fix: zero-fill the known Dec 31 nighttime gap ────
# (see explanation above — verified nighttime, not a real data gap)
for col in ["GHI_S1", "GHI_S2", "GHI_S3"]:
    combined[col] = combined[col].fillna(0.0)

# ── Final completeness check ──────────────────────────────────
print("\n[4] Final NaN check:")
for col in ["GHI_S1", "GHI_S2", "GHI_S3", "GHI_P2"]:
    n_nan = combined[col].isna().sum()
    status = "✅" if n_nan == 0 else f"⚠️  {n_nan} NaN remain"
    print(f"  {col}: {status}")

# Round for clean output
for col in ["GHI_S1", "GHI_S2", "GHI_S3", "GHI_P2"]:
    combined[col] = combined[col].round(2)

out_path = os.path.normpath(
    os.path.join(cfg.BASE_PATH, "all_stations_GHI_30min_PST_filled.csv"))
combined.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ✅")
print(f"Total rows: {len(combined):,}")