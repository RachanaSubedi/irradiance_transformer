# ════════════════════════════════════════════════════════════
# scripts/prepare_deepkriging_inputs.py
#
# Harmonizes all 4 stations into identical format for Deep Kriging:
#   - 3 complete stations: raw 5-min Ambient Weather → 30-min averaged
#   - 1 imputed station:   already 30-min, just reformatted
#
# Output: 4 CSVs, each with exactly two columns [datetime, GHI]
#         all on the SAME 30-min PST time grid, fully aligned.
#
# Timezone: PST (UTC-8, local standard, no DST) — matches NSRDB.
#   Raw stations carry "-08:00" offset (already PST).
#   Imputed station already converted to PST in finetuning Step 12.
# ════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd

# ── CONFIG — edit paths here ─────────────────────────────────
INPUT_DIR  = r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar\data\raw\stations"     # where raw CSVs live
OUTPUT_DIR = r"C:\Users\C838122727\Documents\CSU\research\deepkriging_solar\data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# The canonical 30-min PST grid every station is reindexed onto.
# Full year 2024 in local standard time.
GRID_START = "2024-01-01 00:00:00"
GRID_END   = "2024-12-31 23:30:00"

# Complete stations: raw Ambient Weather 5-min files
# (name, raw_filename, lat, lon)
COMPLETE_STATIONS = [
    ("S1", "46_59__-119_15_2024.csv", 46.594029, -119.152367),
    # Add S2 and S3 with their actual filenames:
    # ("S2", "46_82__-119_15_2024.csv", 46.823242, -119.163197),
    # ("S3", "46_82__-119_16_2024.csv", 46.821036, -119.150761),
]

# Imputed station: your model output
IMPUTED_STATION = ("P2", "station_46_78_full_year_GHI_v2.csv",
                   46.780547, -119.228783)

# Column names in raw Ambient Weather files
RAW_DATE_COL = "Date"                          # ISO 8601 with -08:00 offset
RAW_GHI_COL  = "Solar Radiation (W/m^2)"

# Column names in imputed file
IMP_DATE_COL = "datetime"                      # naive, already PST
IMP_GHI_COL  = "GHI_imputed"

# ── Canonical grid ───────────────────────────────────────────
GRID = pd.date_range(GRID_START, GRID_END, freq="30min")
print(f"Canonical 30-min PST grid: {len(GRID)} timesteps "
      f"({GRID[0]} → {GRID[-1]})\n")


def process_complete_station(name, fname, lat, lon):
    """
    Raw 5-min Ambient Weather CSV → 30-min averaged GHI on PST grid.

    Steps:
      1. Parse ISO timestamp (carries -08:00 = PST offset)
      2. Convert to fixed PST naive datetime (UTC - 8h)
      3. Resample 5-min → 30-min by mean
      4. Reindex onto canonical grid (fills any gaps with NaN)
    """
    path = os.path.join(INPUT_DIR, fname)
    print(f"[{name}] reading {fname} ...")
    df = pd.read_csv(path)

    # Parse the ISO timestamp. It contains -08:00, so utc=True gives
    # correct UTC, then subtract 8h to get PST naive.
    df["dt_utc"] = pd.to_datetime(df[RAW_DATE_COL], utc=True)
    df["datetime"] = (df["dt_utc"] - pd.Timedelta(hours=8)).dt.tz_localize(None)

    df["GHI"] = pd.to_numeric(df[RAW_GHI_COL], errors="coerce")
    df = df[["datetime", "GHI"]].sort_values("datetime").reset_index(drop=True)

    # Resample 5-min → 30-min mean (averages every 6 readings)
    df = df.set_index("datetime")
    df30 = df["GHI"].resample("30min").mean().reset_index()

    # Reindex onto canonical grid so all stations share identical timestamps
    df30 = df30.set_index("datetime").reindex(GRID).rename_axis("datetime").reset_index()

    n_missing = df30["GHI"].isna().sum()
    print(f"  raw rows: {len(df):,}  →  30-min rows: {len(df30):,}  "
          f"(missing: {n_missing})")

    # Optional: fill short gaps by interpolation (max 2 steps = 1 hour)
    df30["GHI"] = df30["GHI"].interpolate(limit=2)

    # Negative or tiny nighttime noise → clip to 0
    df30["GHI"] = df30["GHI"].clip(lower=0)

    return df30


def process_imputed_station(name, fname, lat, lon):
    """
    Imputed model output → same 2-column 30-min PST format.
    Already 30-min and already PST, so just rename + reindex.
    """
    path = os.path.join(INPUT_DIR, fname)
    print(f"[{name}] reading {fname} ...")
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[IMP_DATE_COL])
    df["GHI"] = pd.to_numeric(df[IMP_GHI_COL], errors="coerce")
    df = df[["datetime", "GHI"]].sort_values("datetime").reset_index(drop=True)

    # Reindex onto canonical grid
    df = df.set_index("datetime").reindex(GRID).rename_axis("datetime").reset_index()

    n_missing = df["GHI"].isna().sum()
    print(f"  rows: {len(df):,}  (missing on grid: {n_missing})")
    df["GHI"] = df["GHI"].interpolate(limit=2).clip(lower=0)
    return df


def save_station(df, name):
    """Save with two clean columns and consistent float formatting."""
    out_path = os.path.join(OUTPUT_DIR, f"{name}_GHI_30min_PST.csv")
    df_out = df.copy()
    df_out["GHI"] = df_out["GHI"].round(2)
    df_out.to_csv(out_path, index=False)
    print(f"  saved → {out_path}\n")
    return out_path


# ── Run all stations ─────────────────────────────────────────
print("=" * 60)
print("HARMONIZING STATIONS FOR DEEP KRIGING")
print("=" * 60 + "\n")

all_dfs = {}

# Complete stations
for name, fname, lat, lon in COMPLETE_STATIONS:
    df = process_complete_station(name, fname, lat, lon)
    save_station(df, name)
    all_dfs[name] = df

# Imputed station
name, fname, lat, lon = IMPUTED_STATION
df_imp = process_imputed_station(name, fname, lat, lon)
save_station(df_imp, name)
all_dfs[name] = df_imp

# ── Build a combined wide table (one column per station) ─────
print("Building combined wide table...")
combined = pd.DataFrame({"datetime": GRID})
for name, df in all_dfs.items():
    combined = combined.merge(
        df.rename(columns={"GHI": f"GHI_{name}"}),
        on="datetime", how="left")

combined_path = os.path.join(OUTPUT_DIR, "all_stations_GHI_30min_PST.csv")
combined.to_csv(combined_path, index=False)
print(f"  saved combined → {combined_path}")

# ── Sanity check: solar noon alignment ───────────────────────
print("\n" + "=" * 60)
print("ALIGNMENT CHECK — June 17 peak GHI per station")
print("=" * 60)
for name, df in all_dfs.items():
    jun = df[(df["datetime"].dt.month == 6) & (df["datetime"].dt.day == 17)]
    if jun["GHI"].notna().any():
        peak = jun.loc[jun["GHI"].idxmax(), "datetime"]
        # Daylight window
        daylight = jun[jun["GHI"] > 5]
        if len(daylight) > 0:
            first = daylight["datetime"].iloc[0].strftime("%H:%M")
            last  = daylight["datetime"].iloc[-1].strftime("%H:%M")
            print(f"  {name}: peak {peak.strftime('%H:%M')} PST  |  "
                  f"daylight {first}–{last} PST  |  "
                  f"max {jun['GHI'].max():.0f} W/m²")

print("\nAll 4 stations now share the identical 30-min PST grid.")
print("Ready for Deep Kriging.")