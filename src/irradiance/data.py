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


def process_station_utc(df_raw, lat, lon, alt=120):
    """
    Process raw Ambient Weather CSV → 30-min UTC GHI + CSI dataframe.
    Sets CSI=NaN for nighttime (GHI_clear < 10 W/m²).

    IMPORTANT — resampling method (fixed from earlier .mean() version):
    Raw sensor data is logged roughly every 5 minutes (with some jitter —
    occasional readings land a minute or two off the 5-min mark due to
    logger clock drift). The original pipeline used
    `.resample("30min").mean()`, which averages all ~6 readings inside
    each 30-min window. This systematically destroys brief cloud-edge
    enhancement spikes and genuine clear-sky peaks: verified on real
    station data, mean-resampling cuts the yearly max GHI by 16% (S1),
    17% (S2), and 42% (S3) compared to the true 5-min readings.

    Fix: snapshot the reading nearest each half-hour mark (tolerance
    3 minutes, matching the jitter window observed in the raw logs)
    instead of averaging. This preserves the true instantaneous GHI
    at each 30-min timestamp — physically appropriate for an imputation
    target that should match what an actual 30-min-cadence sensor would
    have recorded, rather than a smoothed mean.
    """
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["Date"], utc=True)
    df["GHI"]      = pd.to_numeric(df["Solar Radiation (W/m^2)"], errors="coerce")
    df = df[["datetime", "GHI"]].sort_values("datetime").reset_index(drop=True)

    # Canonical 30-min UTC grid spanning the data's own range
    grid = pd.date_range(
        df["datetime"].min().floor("30min"),
        df["datetime"].max().ceil("30min"),
        freq="30min", tz="UTC"
    )
    grid_df = pd.DataFrame({"datetime": grid})

    # Snapshot: nearest raw reading within 3 minutes of each grid timestamp
    df = pd.merge_asof(
        grid_df, df, on="datetime",
        direction="nearest", tolerance=pd.Timedelta("3min")
    )

    site  = pvlib.location.Location(lat, lon, tz="UTC", altitude=alt)
    times = pd.DatetimeIndex(df["datetime"])
    cs    = site.get_clearsky(times, model="ineichen")
    sp    = site.get_solarposition(times)
    df["GHI_clear"]  = cs["ghi"].values
    df["solar_elev"] = sp["apparent_elevation"].values
    df["CSI"]        = (df["GHI"] / df["GHI_clear"]).clip(0, 2)
    df.loc[df["GHI_clear"] < 10, "CSI"] = np.nan
    return df


def process_nsrdb_utc(df_raw, station_name):
    """
    Process NSRDB CSV (skiprows=2) → 30-min UTC NSRDB_CSI dataframe.
    NSRDB timestamps are LST (UTC-8) → convert to UTC by adding 8h.
    """
    df = df_raw.copy()
    for col in ["Year","Month","Day","Hour","Minute","GHI","Clearsky GHI"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["datetime"] = (pd.to_datetime({
        "year":   df["Year"].astype(int),
        "month":  df["Month"].astype(int),
        "day":    df["Day"].astype(int),
        "hour":   df["Hour"].astype(int),
        "minute": df["Minute"].astype(int),
    }) + pd.Timedelta(hours=8)).dt.tz_localize("UTC")
    key = f"NSRDB_CSI_{station_name}"
    df[key] = (df["GHI"] / df["Clearsky GHI"]).clip(0, 10)
    df.loc[df["Clearsky GHI"] < 10, key] = np.nan
    return df[["datetime", key]].sort_values("datetime").reset_index(drop=True)


def process_c13_utc(df_raw, cols=None):
    """
    Process GOES-18 C13 GEE export CSV → UTC datetime + brightness temp columns.
    cols: list of column names to keep (e.g. ["s1_c13","s2_c13","s3_c13"]).
    """
    if cols is None:
        cols = ["s1_c13", "s2_c13", "s3_c13"]
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime_local"], utc=True)
    keep = ["datetime"]
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep.append(col)
    return df[keep].sort_values("datetime").reset_index(drop=True)


def build_master(cfg):
    """
    Build the global master dataframe from all complete station data.
    Applies monthly C13 anomaly normalization and binary CSI masks.

    Parameters
    ----------
    cfg : module
        The settings module (config/settings.py).

    Returns
    -------
    master : pd.DataFrame
        Full-year 30-min UTC dataframe with all features.
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

    ns_s1 = process_nsrdb_utc(pd.read_csv(cfg.RAW["nsrdb_s1"], skiprows=2), "s1")
    ns_s2 = process_nsrdb_utc(pd.read_csv(cfg.RAW["nsrdb_s2"], skiprows=2), "s2")
    ns_s3 = process_nsrdb_utc(pd.read_csv(cfg.RAW["nsrdb_s3"], skiprows=2), "s3")

    c13   = process_c13_utc(pd.read_csv(cfg.RAW["c13_complete"]))

    print("  Merging into master dataframe...")
    master = pd.DataFrame({"datetime": pd.date_range(
        "2024-01-01 08:00:00", "2025-01-01 07:30:00", freq="30min", tz="UTC")})

    for st, name in [(st_s1,"s1"),(st_s2,"s2"),(st_s3,"s3")]:
        master = pd.merge_asof(
            master.sort_values("datetime"),
            st.rename(columns={"GHI": f"GHI_{name}", "GHI_clear": f"GHI_clear_{name}",
                                "solar_elev": f"solar_elev_{name}", "CSI": f"CSI_{name}"}
                      ).sort_values("datetime"),
            on="datetime", direction="nearest", tolerance=pd.Timedelta("16min"))

    for ns in [ns_s1, ns_s2, ns_s3]:
        master = pd.merge_asof(master.sort_values("datetime"),
                               ns.sort_values("datetime"),
                               on="datetime", direction="nearest",
                               tolerance=pd.Timedelta("16min"))

    master = pd.merge_asof(master.sort_values("datetime"),
                           c13.sort_values("datetime"),
                           on="datetime", direction="nearest",
                           tolerance=pd.Timedelta("16min"))

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
                              seq_len=48, center=23,
                              anchor_bad_thresh=0.9,
                              c13_gap_thresh=0.3):
    """
    Build (48, 14) sequences for pretraining — hide one complete station.

    Feature layout:
        [0]  anchor1_CSI        [1]  anchor1_mask
        [2]  anchor2_CSI        [3]  anchor2_mask
        [4]  NSRDB_CSI_target   [5]  C13_norm_target
        [6]  NSRDB_CSI_anchor1  [7]  C13_norm_anchor1
        [8]  NSRDB_CSI_anchor2  [9]  C13_norm_anchor2
        [10] hour_sin           [11] hour_cos
        [12] doy_sin            [13] doy_cos

    Parameters
    ----------
    df          : pd.DataFrame  — the global master dataframe
    target      : str           — station being hidden ("s1","s2","s3")
    anchor1     : str           — first context station
    anchor2     : str           — second context station
    seq_len     : int           — window length (default 48)
    center      : int           — index of target timestep (default 23)
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
                              seq_len=48, center=23,
                              anchor_bad_thresh=0.8,
                              has_target=True):
    """
    Build (48, 14) sequences for fine-tuning on a partial station.
    Same 14-feature layout as pretraining.
    When has_target=False → inference mode (no y returned).

    Parameters
    ----------
    df          : pd.DataFrame  — ft_master dataframe
    anchor1     : str           — nearest complete station (e.g. "s3")
    anchor2     : str           — second nearest (e.g. "s1")
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
        ], axis=1).astype(np.float32)

        X_list.append(X)
        dt_list.append(df["datetime_naive"].iloc[i + center])
        if has_target:
            y_list.append(np.float32(y_val))

    print(f"  {len(X_list):,} sequences kept | {skipped:,} skipped")
    X_arr = np.stack(X_list)
    return (X_arr, np.array(y_list), dt_list) if has_target \
           else (X_arr, dt_list)


def fix_missing_anchor(X_imp, center=23,
                        a1_csi_idx=0, a1_mask_idx=1,
                        a2_csi_idx=2, a2_mask_idx=3,
                        a1_nsrdb_idx=6, a1_c13_idx=7,
                        a2_nsrdb_idx=8, a2_c13_idx=9):
    """
    Replace missing anchor1 with anchor2 values in imputation sequences.
    Applied when anchor1 (S3) has data gaps in Oct-Nov.
    Copies CSI, mask, NSRDB, and C13 from anchor2 → anchor1.

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

    print(f"  fix_missing_anchor: {n_fixed} sequences fixed "
          f"({n_fixed/len(X_imp)*100:.1f}%)")
    return X_fixed, n_fixed