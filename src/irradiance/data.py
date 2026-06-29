# ════════════════════════════════════════════════════════════
# utils/data_utils.py
# All data loading and preprocessing functions.
# Import these — never copy-paste them into scripts.
# ════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import pvlib
import torch
from torch.utils.data import Dataset


def process_station_utc(df_raw, lat, lon, alt=120, fill_gaps=True, freq="5min"):
    """
    Process raw Ambient Weather CSV → UTC GHI + CSI dataframe at the
    given resolution. Sets CSI=NaN for nighttime (GHI_clear < 10 W/m²).

    v4 change: default freq switched from "30min" to "5min" — station
    files are natively ~5-min already, so this is no longer a
    downsampling step, just a snap onto a clean regular grid (handles
    the logger clock jitter noted below).

    HISTORICAL NOTE (v2/v3, freq="30min"): the original pipeline used
    `.resample("30min").mean()`, averaging ~6 raw readings per 30-min
    bin. This destroyed brief cloud-edge spikes and genuine clear-sky
    peaks (verified: cut yearly max GHI by 16-42% across stations).
    Fixed by snapshotting the nearest raw reading instead of
    averaging. At freq="5min" this concern mostly disappears since
    each grid step now corresponds to ~1 raw reading directly, but
    the same snapshot logic is kept for consistency and to handle
    the occasional reading that lands a minute or two off-grid due
    to logger clock drift.
    """
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["Date"], utc=True)
    df["GHI"]      = pd.to_numeric(df["Solar Radiation (W/m^2)"], errors="coerce")
    df = df[["datetime", "GHI"]].sort_values("datetime").reset_index(drop=True)

    if fill_gaps:
        file_year = df["datetime"].dt.year.mode()[0]
        grid = pd.date_range(
            f"{file_year}-01-01 00:00:00", f"{file_year}-12-31 23:55:00",
            freq=freq, tz="UTC"
        )
    else:
        grid = pd.date_range(
            df["datetime"].min().floor(freq),
            df["datetime"].max().ceil(freq),
            freq=freq, tz="UTC"
        )
    grid_df = pd.DataFrame({"datetime": grid})

    # Snapshot: nearest raw reading within 1.5 minutes of each grid
    # timestamp (tightened from 3min used at 30-min resolution, since
    # at 5-min native cadence a wider tolerance risks pulling in the
    # ADJACENT 5-min reading instead of a jittered version of the
    # intended one).
    df = pd.merge_asof(
        grid_df, df, on="datetime",
        direction="nearest", tolerance=pd.Timedelta("1.5min")
    )

    site  = pvlib.location.Location(lat, lon, tz="UTC", altitude=alt)
    times = pd.DatetimeIndex(df["datetime"])
    cs    = site.get_clearsky(times, model="ineichen")
    sp    = site.get_solarposition(times)
    df["GHI_clear"]  = cs["ghi"].values
    df["solar_elev"] = sp["apparent_elevation"].values
    df["CSI"]        = (df["GHI"] / df["GHI_clear"]).clip(0, 2)
    df.loc[df["GHI_clear"] < 10, "CSI"] = np.nan

    if not fill_gaps:
        return df

    # ── Self-contained gap filling (complete stations only) ──
    # Uses ONLY this station's own data — no external/sibling-station
    # borrowing. Verified against real outages: S3 had a 46-day
    # continuous gap (mostly Oct 4 - Nov 18); S2 had several partial-
    # day gaps in November. Both are filled here using the station's
    # own seasonal/within-day CSI pattern, not borrowed from elsewhere.
    df["date"] = df["datetime"].dt.date
    daytime = df["GHI_clear"] >= 10
    missing_daytime = df["GHI"].isna() & daytime

    if missing_daytime.any():
        # Tier 1: per-day median CSI from whatever real daytime
        # readings exist that same day.
        daily_csi = (
            df.loc[daytime & df["CSI"].notna()]
              .groupby("date")["CSI"].median()
        )
        df["month"] = df["datetime"].dt.month
        date_to_month = df.drop_duplicates("date").set_index("date")["month"]

        all_dates = sorted(df["date"].unique())
        days_with_data = set(daily_csi.index)

        def _nearest_day_csi(target_date, max_lookaround=5):
            """Tier 2a: median CSI from nearby calendar days (short
            gaps, a few days) that have real data."""
            idx = all_dates.index(target_date)
            candidates = []
            for offset in range(1, max_lookaround + 1):
                for j in (idx - offset, idx + offset):
                    if 0 <= j < len(all_dates):
                        d = all_dates[j]
                        if d in days_with_data:
                            candidates.append(daily_csi.loc[d])
                if candidates:
                    break
            return float(np.median(candidates)) if candidates else np.nan

        def _seasonal_month_csi(target_date, max_month_radius=3):
            """Tier 2b: for gaps too large for day-level lookback to
            be meaningful (verified case: S3's 46-day gap), fall back
            to a seasonal/climatological estimate — median CSI this
            SAME station shows in the nearest calendar month(s) with
            substantial data, expanding outward until >=5 real days
            are found."""
            target_month = date_to_month.get(target_date)
            if target_month is None:
                return np.nan
            for radius in range(0, max_month_radius + 1):
                months_to_try = {
                    ((target_month - 1 + d) % 12) + 1
                    for d in range(-radius, radius + 1)
                }
                month_days = [d for d in days_with_data
                              if date_to_month.get(d) in months_to_try]
                if len(month_days) >= 5:
                    return float(np.median([daily_csi.loc[d] for d in month_days]))
            if len(days_with_data) > 0:
                return float(np.median(list(daily_csi.values)))
            return 0.6  # absolute last resort, station has zero real data

        tier2_cache = {}
        for d in df.loc[missing_daytime, "date"].unique():
            if d not in days_with_data:
                est = _nearest_day_csi(d)
                if np.isnan(est):
                    est = _seasonal_month_csi(d)
                tier2_cache[d] = est

        def _fill_row(row):
            d = row["date"]
            if d in days_with_data:
                est_csi = daily_csi.loc[d]            # Tier 1
            else:
                est_csi = tier2_cache.get(d, np.nan)   # Tier 2a/2b
            if np.isnan(est_csi):
                est_csi = 0.6
            return est_csi * row["GHI_clear"]

        filled_vals = df.loc[missing_daytime].apply(_fill_row, axis=1)
        df.loc[missing_daytime, "GHI"] = filled_vals
        df.loc[missing_daytime, "CSI"] = (
            df.loc[missing_daytime, "GHI"] / df.loc[missing_daytime, "GHI_clear"]
        ).clip(0, 2)
        df = df.drop(columns=["month"])

    # Nighttime: GHI is genuinely zero, not missing.
    nighttime = ~daytime
    df.loc[nighttime & df["GHI"].isna(), "GHI"] = 0.0

    df = df.drop(columns=["date"])
    return df


def process_nsrdb_utc(df_raw, station_name, source_tz="LST"):
    """
    Process NSRDB CSV (skiprows=2) → UTC NSRDB_CSI dataframe.

    source_tz : str, "LST" or "UTC"
        "LST" (default) — old-format files where Year/Month/Day/Hour/
        Minute are in local standard time (UTC-8 for this region).
        Converts to UTC by adding 8h, as the original pipeline did.

        "UTC" — new-format 5-min files downloaded directly in UTC
        (confirmed via solar-noon check: peak Clearsky GHI lands at
        ~19:55-20:00, matching true UTC solar noon at this longitude,
        not the ~12:00 LST the old files showed). No shift applied.

        Getting this wrong silently shifts every NSRDB feature by
        8 hours — verify with a solar-noon check on any new file
        before trusting source_tz="UTC" blindly.
    """
    df = df_raw.copy()
    for col in ["Year","Month","Day","Hour","Minute","GHI","Clearsky GHI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    raw_dt = pd.to_datetime({
        "year":   df["Year"].astype(int),
        "month":  df["Month"].astype(int),
        "day":    df["Day"].astype(int),
        "hour":   df["Hour"].astype(int),
        "minute": df["Minute"].astype(int),
    })
    if source_tz == "LST":
        df["datetime"] = (raw_dt + pd.Timedelta(hours=8)).dt.tz_localize("UTC")
    elif source_tz == "UTC":
        df["datetime"] = raw_dt.dt.tz_localize("UTC")
    else:
        raise ValueError(f"source_tz must be 'LST' or 'UTC', got {source_tz!r}")

    key = f"NSRDB_CSI_{station_name}"
    df[key] = (df["GHI"] / df["Clearsky GHI"]).clip(0, 10)
    df.loc[df["Clearsky GHI"] < 10, key] = np.nan
    return df[["datetime", key]].sort_values("datetime").reset_index(drop=True)


def process_c13_c02_utc(df_raw, station_name, c13_divisor=10.0):
    """
    Process a single-pixel GOES-18 C13+C02 GEE export CSV (NEW format,
    one file per pixel, ~5-min cadence) → UTC dataframe with calibrated
    C13 (Kelvin) and raw-DN C02.

    Replaces the old process_c13_utc, which expected one bundled file
    with pre-named columns (s1_c13, s2_c13, s3_c13) for all three
    complete stations together. The new GEE extraction produces a
    separate file per pixel instead — this function processes ONE
    such file and tags its columns with the given station_name.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw CSV with columns: datetime_utc, bt_c13_raw, refl_c02_raw,
        pixel_id (pixel_id is informational only, not used here).
    station_name : str
        Station/anchor key this pixel represents (e.g. "s1", "s2",
        "s3", "new"). Used to name output columns.
    c13_divisor : float, default 10.0
        bt_c13_raw is uncalibrated DN, not Kelvin. Dividing by 10
        recovers values matching the old pipeline's already-
        calibrated C13 (verified: mean ~291K both before and after,
        same as historical data for this site/season). C02 has no
        documented divisor and is used as raw DN.

    Returns
    -------
    pd.DataFrame with columns: datetime, {station_name}_c13, {station_name}_c02
    """
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df[f"{station_name}_c13"] = pd.to_numeric(df["bt_c13_raw"], errors="coerce") / c13_divisor
    df[f"{station_name}_c02"] = pd.to_numeric(df["refl_c02_raw"], errors="coerce")
    keep = ["datetime", f"{station_name}_c13", f"{station_name}_c02"]
    return df[keep].sort_values("datetime").reset_index(drop=True)




def build_master(cfg):
    """
    Build the global master dataframe from all complete station data.
    Applies monthly C13/C02 anomaly normalization and binary CSI masks.

    v4 changes: 5-min native resolution (was 30-min), new per-pixel
    C13+C02 loader (was one bundled file), NSRDB now UTC-native (was
    LST) — see process_nsrdb_utc's source_tz parameter.

    Parameters
    ----------
    cfg : module
        The settings module (config/settings.py).

    Returns
    -------
    master : pd.DataFrame
        Full-year 5-min UTC dataframe with all features.
    """
    import pandas as pd
    import numpy as np

    print("  Loading complete station data...")
    st_s1 = process_station_utc(pd.read_csv(cfg.RAW["ghi_s1"]),
                                 **cfg.STATIONS["s1"])
    st_s2 = process_station_utc(pd.read_csv(cfg.RAW["ghi_s2"]),
                                 **cfg.STATIONS["s2"])
    st_s3 = process_station_utc(pd.read_csv(cfg.RAW["ghi_s3"]),
                                 **cfg.STATIONS["s3"])

    ns_s1 = process_nsrdb_utc(pd.read_csv(cfg.RAW["nsrdb_s1"], skiprows=2),
                               "s1", source_tz="UTC")
    ns_s2 = process_nsrdb_utc(pd.read_csv(cfg.RAW["nsrdb_s2"], skiprows=2),
                               "s2", source_tz="UTC")
    ns_s3 = process_nsrdb_utc(pd.read_csv(cfg.RAW["nsrdb_s3"], skiprows=2),
                               "s3", source_tz="UTC")

    # C13+C02 — one file per pixel now, NOT bundled. S2 and S3 share
    # the same pixel file (confirmed physical redundancy, see config.py).
    c13c02_s1 = process_c13_c02_utc(
        pd.read_csv(cfg.RAW["c13c02_s1"]), "s1",
        c13_divisor=cfg.C13_DN_TO_KELVIN_DIVISOR)
    c13c02_s2 = process_c13_c02_utc(
        pd.read_csv(cfg.RAW["c13c02_s2"]), "s2",
        c13_divisor=cfg.C13_DN_TO_KELVIN_DIVISOR)
    c13c02_s3 = process_c13_c02_utc(
        pd.read_csv(cfg.RAW["c13c02_s3"]), "s3",
        c13_divisor=cfg.C13_DN_TO_KELVIN_DIVISOR)

    print("  Merging into master dataframe...")
    # 5-min grid (was 30-min). Tolerance tightened to 3min (was 16min)
    # to match the finer resolution — a 16min tolerance on a 5-min
    # grid would let readings 3+ grid-steps away match, defeating
    # the resolution gain.
    master = pd.DataFrame({"datetime": pd.date_range(
        "2024-01-01 00:00:00", "2025-01-01 00:00:00", freq="5min", tz="UTC")})

    for st, name in [(st_s1,"s1"),(st_s2,"s2"),(st_s3,"s3")]:
        master = pd.merge_asof(
            master.sort_values("datetime"),
            st.rename(columns={"GHI": f"GHI_{name}", "GHI_clear": f"GHI_clear_{name}",
                                "solar_elev": f"solar_elev_{name}", "CSI": f"CSI_{name}"}
                      ).sort_values("datetime"),
            on="datetime", direction="nearest", tolerance=pd.Timedelta("3min"))

    for ns in [ns_s1, ns_s2, ns_s3]:
        master = pd.merge_asof(master.sort_values("datetime"),
                               ns.sort_values("datetime"),
                               on="datetime", direction="nearest",
                               tolerance=pd.Timedelta("3min"))

    for c13c02 in [c13c02_s1, c13c02_s2, c13c02_s3]:
        master = pd.merge_asof(master.sort_values("datetime"),
                               c13c02.sort_values("datetime"),
                               on="datetime", direction="nearest",
                               tolerance=pd.Timedelta("3min"))

    # Fill missing CSI/NSRDB with -1 sentinel
    for col in ["CSI_s1","CSI_s2","CSI_s3",
                "NSRDB_CSI_s1","NSRDB_CSI_s2","NSRDB_CSI_s3"]:
        master[col] = master[col].fillna(-1.0)

    # Time features
    master["datetime_naive"] = master["datetime"].dt.tz_localize(None)
    master["hour"]     = master["datetime_naive"].dt.hour + master["datetime_naive"].dt.minute / 60
    master["doy"]      = master["datetime_naive"].dt.day_of_year
    master["hour_sin"] = np.sin(2 * np.pi * master["hour"] / 24)
    master["hour_cos"] = np.cos(2 * np.pi * master["hour"] / 24)
    master["doy_sin"]  = np.sin(2 * np.pi * master["doy"] / 366)
    master["doy_cos"]  = np.cos(2 * np.pi * master["doy"] / 366)
    master["month"]    = master["datetime_naive"].dt.month

    # Monthly C13 anomaly normalization
    # Removes seasonal background temperature so C13 encodes cloud signal
    for col in ["s1_c13", "s2_c13", "s3_c13"]:
        monthly_mean = master.groupby("month")[col].transform("mean")
        monthly_std  = master.groupby("month")[col].transform("std").clip(lower=1.0)
        master[col + "_norm"] = (master[col] - monthly_mean) / monthly_std
    for col in ["s1_c13_norm", "s2_c13_norm", "s3_c13_norm"]:
        master[col] = master[col].ffill(limit=2).fillna(0.0)

    # Monthly C02 anomaly normalization — DAYTIME ONLY (v4 fix)
    #
    # Unlike C13 (thermal IR, nonzero day and night), C02 is visible-
    # band reflectance — exactly 0 at night (no sunlight to reflect)
    # for roughly half of every day. Computing monthly mean/std over
    # ALL hours (as C13 correctly does) drags the mean toward zero
    # and inflates the std with a huge mass of exact-zero nighttime
    # readings, leaving the normalized daytime signal skewed (verified:
    # produced mean=+0.73 instead of ~0, and made S2/S3's already-
    # shared-pixel C02 values redundant in a way that likely
    # contributed to the hide_s3 pretraining R^2 dropping to 0.32).
    #
    # Fix: compute monthly mean/std from DAYTIME rows only (using each
    # station's own solar_elev > 0 as the daytime mask), apply that
    # normalization to daytime rows, and set nighttime C02_norm to a
    # fixed sentinel (0.0) directly — consistent with how nighttime
    # CSI is already excluded from its own normalization elsewhere
    # in this function.
    for col, solar_col in [("s1_c02", "solar_elev_s1"),
                            ("s2_c02", "solar_elev_s2"),
                            ("s3_c02", "solar_elev_s3")]:
        is_day = master[solar_col] > 0
        day_mean = master.loc[is_day].groupby("month")[col].transform("mean")
        day_std  = master.loc[is_day].groupby("month")[col].transform("std").clip(lower=1.0)
        master[col + "_norm"] = 0.0
        master.loc[is_day, col + "_norm"] = (
            (master.loc[is_day, col] - day_mean) / day_std
        )
    for col in ["s1_c02_norm", "s2_c02_norm", "s3_c02_norm"]:
        master[col] = master[col].ffill(limit=2).fillna(0.0)

    # Binary validity masks
    for col in ["CSI_s1", "CSI_s2", "CSI_s3"]:
        master[col + "_mask"] = (master[col] >= 0).astype(np.float32)

    master = master.reset_index(drop=True)
    print(f"  master shape: {master.shape}")
    return master


class CSIDataset(Dataset):
    """PyTorch dataset wrapper for (X, y) numpy arrays."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]



# ════════════════════════════════════════════════════════════
# utils/sequence_builder.py
# Sliding window sequence construction for pretraining and fine-tuning.
# All sequence building goes through here — never inline it in scripts.
# ════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd


def build_pretrain_sequences(df, target, anchor1, anchor2,
                              seq_len=72, center=35,
                              anchor_bad_thresh=0.9,
                              c13_gap_thresh=0.3):
    """
    Build (seq_len, 17) sequences for pretraining — hide one complete station.

    v4 feature layout (17 features, was 14 in v2/v3):
        [0]  anchor1_CSI        [1]  anchor1_mask
        [2]  anchor2_CSI        [3]  anchor2_mask
        [4]  NSRDB_CSI_target   [5]  C13_norm_target
        [6]  NSRDB_CSI_anchor1  [7]  C13_norm_anchor1
        [8]  NSRDB_CSI_anchor2  [9]  C13_norm_anchor2
        [10] hour_sin           [11] hour_cos
        [12] doy_sin            [13] doy_cos
        [14] C02_norm_target    [15] C02_norm_anchor1
        [16] C02_norm_anchor2

    Parameters
    ----------
    df          : pd.DataFrame  — the global master dataframe
    target      : str           — station being hidden ("s1","s2","s3")
    anchor1     : str           — first context station
    anchor2     : str           — second context station
    seq_len     : int           — window length (default 72 = 6h @ 5min)
    center      : int           — index of target timestep (default 35)
    anchor_bad_thresh : float   — max fraction of missing anchor steps (default 0.9)
    c13_gap_thresh    : float   — max fraction of zero C13 steps (default 0.3)
    """
    cols = list(df.columns)
    def ci(name): return cols.index(name)

    i_a1      = ci(f"CSI_{anchor1}")
    i_a1_mask = ci(f"CSI_{anchor1}_mask")
    i_a2      = ci(f"CSI_{anchor2}")
    i_a2_mask = ci(f"CSI_{anchor2}_mask")
    i_ns_tgt  = ci(f"NSRDB_CSI_{target}")
    i_c13_tgt = ci(f"{target}_c13_norm")
    i_ns_a1   = ci(f"NSRDB_CSI_{anchor1}")
    i_c13_a1  = ci(f"{anchor1}_c13_norm")
    i_ns_a2   = ci(f"NSRDB_CSI_{anchor2}")
    i_c13_a2  = ci(f"{anchor2}_c13_norm")
    i_hs      = ci("hour_sin")
    i_hc      = ci("hour_cos")
    i_ds      = ci("doy_sin")
    i_dc      = ci("doy_cos")
    i_c02_tgt = ci(f"{target}_c02_norm")    # v4 new
    i_c02_a1  = ci(f"{anchor1}_c02_norm")   # v4 new
    i_c02_a2  = ci(f"{anchor2}_c02_norm")   # v4 new
    i_tgt     = ci(f"CSI_{target}")

    data = df.values
    X_list, y_list, meta_list = [], [], []
    skipped_night = skipped_anchor = skipped_c13 = 0

    for i in range(len(df) - seq_len + 1):
        w     = data[i : i + seq_len]
        y_val = float(w[center, i_tgt])

        if y_val <= 0:
            skipped_night += 1; continue

        frac_a1_bad = (w[:, i_a1_mask] == 0).mean()
        frac_a2_bad = (w[:, i_a2_mask] == 0).mean()
        if frac_a1_bad > anchor_bad_thresh or frac_a2_bad > anchor_bad_thresh:
            skipped_anchor += 1; continue

        if (w[:, i_c13_tgt] == 0).mean() > c13_gap_thresh:
            skipped_c13 += 1; continue

        X = np.stack([
            w[:, i_a1],      w[:, i_a1_mask],
            w[:, i_a2],      w[:, i_a2_mask],
            w[:, i_ns_tgt],  w[:, i_c13_tgt],
            w[:, i_ns_a1],   w[:, i_c13_a1],
            w[:, i_ns_a2],   w[:, i_c13_a2],
            w[:, i_hs],      w[:, i_hc],
            w[:, i_ds],      w[:, i_dc],
            w[:, i_c02_tgt], w[:, i_c02_a1],  w[:, i_c02_a2],
        ], axis=1).astype(np.float32)


        X_list.append(X)
        y_list.append(np.float32(y_val))
        meta_list.append({
            "task":            f"hide_{target}",
            "target_station":  target,
            "anchor1":         anchor1,
            "anchor2":         anchor2,
            "datetime_center": str(df["datetime_naive"].iloc[i + center]),
        })

    print(f"  hide_{target}: {len(X_list):>6,} sequences | "
          f"skipped → night={skipped_night:,} "
          f"anchor={skipped_anchor:,} c13={skipped_c13:,}")
    return X_list, y_list, meta_list


def build_finetune_sequences(df, anchor1, anchor2,
                              seq_len=72, center=35,
                              anchor_bad_thresh=0.8,
                              has_target=True):
    """
    Build (seq_len, 17) sequences for fine-tuning on a partial station.
    Same 17-feature layout as pretraining (v4: +3 C02 slots vs v2/v3's 14).
    When has_target=False → inference mode (no y returned).

    Parameters
    ----------
    df          : pd.DataFrame  — ft_master dataframe
    anchor1     : str           — nearest complete station (e.g. "s3")
    anchor2     : str           — second nearest (e.g. "s1")
    seq_len     : int           — window length (default 72 = 6h @ 5min)
    center      : int           — index of target timestep (default 35)
    has_target  : bool          — True for training, False for imputation
    """
    cols = list(df.columns)
    def ci(name): return cols.index(name)

    i_a1      = ci(f"CSI_{anchor1}")
    i_a1_mask = ci(f"CSI_{anchor1}_mask")
    i_a2      = ci(f"CSI_{anchor2}")
    i_a2_mask = ci(f"CSI_{anchor2}_mask")
    i_ns_new  = ci("NSRDB_CSI_new")
    i_c13_new = ci("c13_new_norm")
    i_ns_a1   = ci(f"NSRDB_CSI_{anchor1}")
    i_c13_a1  = ci(f"{anchor1}_c13_norm")
    i_ns_a2   = ci(f"NSRDB_CSI_{anchor2}")
    i_c13_a2  = ci(f"{anchor2}_c13_norm")
    i_hs      = ci("hour_sin")
    i_hc      = ci("hour_cos")
    i_ds      = ci("doy_sin")
    i_dc      = ci("doy_cos")
    i_c02_new = ci("c02_new_norm")           # v4 new
    i_c02_a1  = ci(f"{anchor1}_c02_norm")    # v4 new
    i_c02_a2  = ci(f"{anchor2}_c02_norm")    # v4 new
    if has_target:
        i_tgt = ci("CSI_new")

    data = df.values
    X_list, y_list, dt_list = [], [], []
    skipped = 0

    for i in range(len(df) - seq_len + 1):
        w = data[i : i + seq_len]

        if has_target:
            y_val = float(w[center, i_tgt])
            if y_val <= 0:
                skipped += 1; continue

        if (w[:, i_a1_mask] == 0).mean() > anchor_bad_thresh:
            skipped += 1; continue

        X = np.stack([
            w[:, i_a1],      w[:, i_a1_mask],
            w[:, i_a2],      w[:, i_a2_mask],
            w[:, i_ns_new],  w[:, i_c13_new],
            w[:, i_ns_a1],   w[:, i_c13_a1],
            w[:, i_ns_a2],   w[:, i_c13_a2],
            w[:, i_hs],      w[:, i_hc],
            w[:, i_ds],      w[:, i_dc],
            w[:, i_c02_new], w[:, i_c02_a1],  w[:, i_c02_a2],
        ], axis=1).astype(np.float32)

        X_list.append(X)
        dt_list.append(df["datetime_naive"].iloc[i + center])
        if has_target:
            y_list.append(np.float32(y_val))

    print(f"  {len(X_list):,} sequences kept | {skipped:,} skipped")
    X_arr = np.stack(X_list)
    return (X_arr, np.array(y_list), dt_list) if has_target \
           else (X_arr, dt_list)


def fix_missing_anchor(X_imp, center=35,
                        a1_csi_idx=0, a1_mask_idx=1,
                        a2_csi_idx=2, a2_mask_idx=3,
                        a1_nsrdb_idx=6, a1_c13_idx=7,
                        a2_nsrdb_idx=8, a2_c13_idx=9,
                        a1_c02_idx=15, a2_c02_idx=16):
    """
    Replace missing anchor1 with anchor2 values in imputation sequences.
    Applied when anchor1 (S3) has data gaps.
    Copies CSI, mask, NSRDB, C13, and C02 (v4 new) from anchor2 → anchor1.

    Returns fixed copy of X_imp and count of fixed sequences.
    """
    X_fixed = X_imp.copy()
    a1_mask = X_fixed[:, center, a1_mask_idx]
    a2_mask = X_fixed[:, center, a2_mask_idx]
    broken  = (a1_mask < 0.5) & (a2_mask > 0.5)
    n_fixed = broken.sum()

    if n_fixed > 0:
        X_fixed[broken, :, a1_csi_idx]   = X_fixed[broken, :, a2_csi_idx]
        X_fixed[broken, :, a1_mask_idx]  = X_fixed[broken, :, a2_mask_idx]
        X_fixed[broken, :, a1_nsrdb_idx] = X_fixed[broken, :, a2_nsrdb_idx]
        X_fixed[broken, :, a1_c13_idx]   = X_fixed[broken, :, a2_c13_idx]
        X_fixed[broken, :, a1_c02_idx]   = X_fixed[broken, :, a2_c02_idx]

    print(f"  fix_missing_anchor: {n_fixed} sequences fixed "
          f"({n_fixed/len(X_imp)*100:.1f}%)")
    return X_fixed, n_fixed


