# ════════════════════════════════════════════════════════════
# scripts/export_filled_stations.py
#
# Standalone utility: runs process_station_utc (snapshot resampling
# + self-contained gap filling) on S1, S2, S3 individually and saves
# each as its own complete CSV — no master, no merging, no model.
#
# Use this to inspect/verify the filled station data on its own,
# independent of the rest of the pipeline.
#
# Output: 3 CSVs in your datasets folder:
#   S1_filled_30min.csv
#   S2_filled_30min.csv
#   S3_filled_30min.csv
# Each has columns: datetime, GHI, GHI_clear, solar_elev, CSI
#
# HOW TO RUN:
#   python scripts/export_filled_stations.py
# ════════════════════════════════════════════════════════════

from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd
from irradiance import config as cfg
from irradiance.data import process_station_utc

STATIONS = [
    ("S1", cfg.RAW["ghi_s1"], cfg.STATIONS["s1"]),
    ("S2", cfg.RAW["ghi_s2"], cfg.STATIONS["s2"]),
    ("S3", cfg.RAW["ghi_s3"], cfg.STATIONS["s3"]),
]

print("=" * 60)
print("EXPORTING FILLED STATION DATA (S1, S2, S3)")
print("=" * 60)

for name, path, coords in STATIONS:
    print(f"\n[{name}] Loading and filling from {path} ...")
    raw = pd.read_csv(path)
    filled = process_station_utc(
        raw, coords["lat"], coords["lon"], coords["alt"],
        fill_gaps=True,   # explicit — these are complete stations
    )

    # Sanity check — confirm zero remaining daytime gaps
    daytime = filled["GHI_clear"] >= 10
    n_missing = (daytime & filled["GHI"].isna()).sum()
    print(f"  Rows: {len(filled):,} | Daytime gaps remaining: {n_missing}")

    out_path = os.path.normpath(
        os.path.join(cfg.BASE_PATH, f"{name}_filled_30min.csv")
    )
    filled.to_csv(out_path, index=False)
    print(f"  Saved: {out_path} ✅")

print("\nDone. All 3 stations exported with zero daytime gaps.")
