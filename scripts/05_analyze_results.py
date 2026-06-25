# ════════════════════════════════════════════════════════════
# scripts/07_plot_results.py
#
# Final visualization + diagnostics plotting script
#
# PURPOSE
# ───────
# Loads the already-generated full-year imputed GHI CSV
# and produces publication-quality figures:
#
#   1. Full-year time series
#   2. Monthly mean/std/max bar chart
#   3. Daily average profile by month
#   4. CSI distribution histograms
#   5. Seasonal boxplots
#
# No model training.
# No fine-tuning.
# No inference.
#
# ════════════════════════════════════════════════════════════

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────
# PATHS
# ────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = ROOT / "datasets" / "station_46_78_full_year_GHI_v2.csv"

OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────
# LOAD
# ────────────────────────────────────────────────────────────

print("Loading full-year CSV...")
df = pd.read_csv(CSV_PATH)

df["datetime"] = pd.to_datetime(df["datetime"])

print(f"Rows: {len(df):,}")

# ────────────────────────────────────────────────────────────
# BASIC FEATURES
# ────────────────────────────────────────────────────────────

df["month"] = df["datetime"].dt.month
df["hour"]  = df["datetime"].dt.hour + df["datetime"].dt.minute / 60

month_names = {
    1:"Jan",2:"Feb",3:"Mar",4:"Apr",
    5:"May",6:"Jun",7:"Jul",8:"Aug",
    9:"Sep",10:"Oct",11:"Nov",12:"Dec"
}

# daytime only
day_df = df[df["GHI"] > 0].copy()

# ────────────────────────────────────────────────────────────
# 1. FULL YEAR TIMESERIES
# ────────────────────────────────────────────────────────────

print("Plot 1: full year timeseries")

plt.figure(figsize=(16,5))

plt.plot(
    df["datetime"],
    df["GHI"],
    linewidth=0.6
)

plt.xlabel("Date")
plt.ylabel("GHI (W/m²)")
plt.title("Full-Year Imputed GHI — Station P2")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "01_full_year_timeseries.png",
    dpi=300
)

plt.close()

# ────────────────────────────────────────────────────────────
# 2. MONTHLY STATISTICS
# ────────────────────────────────────────────────────────────

print("Plot 2: monthly statistics")

monthly = day_df.groupby("month")["GHI"].agg(["mean","std","max"])

x = np.arange(len(monthly))

plt.figure(figsize=(12,5))

plt.bar(
    x,
    monthly["mean"],
    yerr=monthly["std"],
    capsize=4
)

plt.xticks(
    x,
    [month_names[m] for m in monthly.index]
)

plt.ylabel("GHI (W/m²)")
plt.title("Monthly Mean ± Std Daytime GHI")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "02_monthly_statistics.png",
    dpi=300
)

plt.close()

# ────────────────────────────────────────────────────────────
# 3. DAILY PROFILE BY MONTH
# ────────────────────────────────────────────────────────────

print("Plot 3: daily profiles")

monthly_hourly = (
    day_df
    .groupby(["month","hour"])["GHI"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(12,6))

for m in [1,4,7,10]:
    sub = monthly_hourly[monthly_hourly["month"] == m]

    plt.plot(
        sub["hour"],
        sub["GHI"],
        linewidth=2,
        label=month_names[m]
    )

plt.xlabel("Hour of Day")
plt.ylabel("Average GHI (W/m²)")
plt.title("Average Daily GHI Profiles")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "03_daily_profiles.png",
    dpi=300
)

plt.close()

# ────────────────────────────────────────────────────────────
# 4. MONTHLY BOX PLOTS
# ────────────────────────────────────────────────────────────

print("Plot 4: monthly boxplots")

monthly_data = [
    day_df[day_df["month"] == m]["GHI"].values
    for m in range(1,13)
]

plt.figure(figsize=(14,6))

plt.boxplot(
    monthly_data,
    labels=[month_names[m] for m in range(1,13)],
    showfliers=False
)

plt.ylabel("GHI (W/m²)")
plt.title("Seasonal Distribution of Daytime GHI")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "04_monthly_boxplots.png",
    dpi=300
)

plt.close()

# ────────────────────────────────────────────────────────────
# 5. CSI HISTOGRAM
# ────────────────────────────────────────────────────────────

if "CSI_imputed" in df.columns:

    print("Plot 5: CSI histogram")

    csi = df["CSI_imputed"].dropna()

    plt.figure(figsize=(8,5))

    plt.hist(
        csi,
        bins=50
    )

    plt.xlabel("CSI")
    plt.ylabel("Frequency")
    plt.title("Distribution of Imputed CSI")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / "05_csi_histogram.png",
        dpi=300
    )

    plt.close()

# ────────────────────────────────────────────────────────────
# MONTHLY SUMMARY TABLE
# ────────────────────────────────────────────────────────────

print("\nMonthly daytime GHI summary:\n")

summary = (
    day_df
    .groupby("month")["GHI"]
    .agg(["mean","max","std"])
)

summary.index = [month_names[m] for m in summary.index]

print(summary.round(1))

summary.to_csv(
    OUT_DIR / "monthly_summary.csv"
)

# ────────────────────────────────────────────────────────────
# DONE
# ────────────────────────────────────────────────────────────

print("\nSaved plots to:")
print(OUT_DIR)

print("\nDone.")