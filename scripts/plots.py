# ════════════════════════════════════════════════════════════
# scripts/07_plot_results.py
#
# Standalone plotting script
# Each figure is saved separately.
#
# Run:
#   python scripts/07_plot_results.py
# ════════════════════════════════════════════════════════════

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from irradiance import config as cfg

# ────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ────────────────────────────────────────────────────────────

OUT_DIR = ROOT / "outputs" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ════════════════════════════════════════════════════════════
# LOAD FULL YEAR CSV
# ════════════════════════════════════════════════════════════

print("Loading full year CSV...")

df = pd.read_csv(cfg.ARTIFACTS["ghi_csv_v2"])

df["datetime"] = pd.to_datetime(df["datetime"])

df["month"]       = df["datetime"].dt.month
df["hour_utc"]    = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
df["hour_pst"]    = (df["hour_utc"] - 8) % 24
df["date"]        = df["datetime"].dt.date
df["day_of_year"] = df["datetime"].dt.day_of_year

print(f"Loaded {len(df):,} rows")

# daytime only
day = df[df["solar_elev"] > 5].copy()

# source groups
imputed_sources = [
    s for s in df["source"].unique()
    if "imputed" in s or "gap_fill" in s
]

measured_sources = [
    s for s in df["source"].unique()
    if "measured" in s
]

imp  = day[day["source"].isin(imputed_sources)]
meas = day[day["source"].isin(measured_sources)]

# monthly stats
monthly = day.groupby("month")["GHI_imputed"].agg(["mean","max","std"])

# ════════════════════════════════════════════════════════════
# PLOT 1 — FULL YEAR TIMESERIES
# ════════════════════════════════════════════════════════════

print("Plot 1 — full year timeseries")

plt.figure(figsize=(16,5))

plt.scatter(
    imp["datetime"],
    imp["GHI_imputed"],
    s=1,
    alpha=0.4,
    label="Imputed"
)

plt.scatter(
    meas["datetime"],
    meas["GHI_imputed"],
    s=4,
    alpha=0.9,
    label="Measured"
)

plt.axvline(
    pd.Timestamp("2024-11-19"),
    linestyle="--",
    linewidth=1.5,
    label="Imputed → Measured"
)

plt.xlabel("Date")
plt.ylabel("GHI (W/m²)")
plt.title("Full Year Daytime GHI — Station P2")

plt.legend()
plt.grid(alpha=0.3)

plt.ylim(0, 950)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "01_full_year_timeseries.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ════════════════════════════════════════════════════════════
# PLOT 2 — MONTHLY MEAN ± STD
# ════════════════════════════════════════════════════════════

print("Plot 2 — monthly mean/std")

means = [monthly.loc[m,"mean"] if m in monthly.index else 0
         for m in range(1,13)]

stds  = [monthly.loc[m,"std"] if m in monthly.index else 0
         for m in range(1,13)]

colors = [
    "#38BDF8" if m < 11 else "#34D399"
    for m in range(1,13)
]

plt.figure(figsize=(10,5))

plt.bar(
    range(12),
    means,
    yerr=stds,
    capsize=4,
    color=colors,
    alpha=0.85
)

plt.xticks(range(12), MONTHS)

plt.ylabel("Mean Daytime GHI (W/m²)")
plt.title("Monthly Mean ± Std Daytime GHI")

plt.grid(axis="y", alpha=0.3)

plt.legend(handles=[
    Patch(color="#38BDF8", label="Imputed"),
    Patch(color="#34D399", label="Measured")
])

plt.tight_layout()

plt.savefig(
    OUT_DIR / "02_monthly_mean_std.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ════════════════════════════════════════════════════════════
# PLOT 3 — GHI HEATMAP
# ════════════════════════════════════════════════════════════

print("Plot 3 — heatmap")

day["hour_pst_int"] = day["hour_pst"].astype(int)

pivot = day.pivot_table(
    index="day_of_year",
    columns="hour_pst_int",
    values="GHI_imputed",
    aggfunc="mean"
)

plt.figure(figsize=(12,8))

im = plt.imshow(
    pivot.values,
    aspect="auto",
    cmap="YlOrRd",
    interpolation="nearest",
    origin="upper"
)

plt.colorbar(im, label="GHI (W/m²)")

plt.xlabel("Hour (PST)")
plt.ylabel("Day of Year")

plt.title("GHI Heatmap (Day of Year × Hour PST)")

month_doys = [1,32,61,92,122,153,183,214,245,275,306,336]

plt.yticks(
    [d-1 for d in month_doys],
    MONTHS,
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "03_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ════════════════════════════════════════════════════════════
# PLOT 4 — BEST CLEAR SUMMER DAY
# ════════════════════════════════════════════════════════════

print("Plot 4 — best summer day")

summer = day[day["month"].between(6,8)].copy()

best_day = (
    summer.groupby("date")["GHI_imputed"]
    .max()
    .idxmax()
)

sday = day[day["date"] == best_day].sort_values("hour_pst")

plt.figure(figsize=(10,5))

plt.plot(
    sday["hour_pst"],
    sday["GHI_imputed"],
    linewidth=2,
    marker="o",
    markersize=4
)

plt.axvline(
    12,
    linestyle="--",
    linewidth=1,
    alpha=0.5,
    label="12:00 PST"
)

plt.xlabel("Hour (PST)")
plt.ylabel("GHI (W/m²)")

plt.title(f"Best Clear Summer Day — {best_day}")

plt.xlim(4,21)
plt.ylim(0,950)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_DIR / "04_best_summer_day.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ════════════════════════════════════════════════════════════
# PLOT 5 — CLEAR VS CLOUDY
# ════════════════════════════════════════════════════════════

print("Plot 5 — clear vs cloudy")

meas_copy = meas.copy()

meas_copy["date"] = meas_copy["datetime"].dt.date

cloudy_day = (
    meas_copy.groupby("date")["GHI_imputed"]
    .mean()
    .idxmin()
)

scloudy = day[
    day["date"] == cloudy_day
].sort_values("hour_pst")

plt.figure(figsize=(10,5))

plt.plot(
    sday["hour_pst"],
    sday["GHI_imputed"],
    linewidth=2,
    label=f"Clear {best_day}"
)

plt.plot(
    scloudy["hour_pst"],
    scloudy["GHI_imputed"],
    linewidth=2,
    label=f"Cloudy {cloudy_day}"
)

plt.axvline(
    12,
    linestyle="--",
    linewidth=1,
    alpha=0.5
)

plt.xlabel("Hour (PST)")
plt.ylabel("GHI (W/m²)")

plt.title("Clear vs Cloudy Day")

plt.xlim(4,21)
plt.ylim(0,950)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUT_DIR / "05_clear_vs_cloudy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ════════════════════════════════════════════════════════════
# PLOT 6 — MONTHLY BOXPLOTS
# ════════════════════════════════════════════════════════════

print("Plot 6 — monthly boxplots")

monthly_data = [
    day[day["month"] == m]["GHI_imputed"].values
    for m in range(1,13)
]

plt.figure(figsize=(14,6))

plt.boxplot(
    monthly_data,
    labels=MONTHS,
    showfliers=False
)

plt.ylabel("Daytime GHI (W/m²)")

plt.title("Seasonal Distribution of Daytime GHI")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "06_monthly_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ════════════════════════════════════════════════════════════
# MONTHLY SUMMARY CSV
# ════════════════════════════════════════════════════════════

summary = (
    day.groupby("month")["GHI_imputed"]
    .agg(["mean","max","std"])
)

summary.index = MONTHS[:len(summary)]

summary.to_csv(
    OUT_DIR / "monthly_summary.csv"
)

print("\nMonthly summary:")
print(summary.round(1))

print("\nSaved plots to:")
print(OUT_DIR)

print("\nDONE.")