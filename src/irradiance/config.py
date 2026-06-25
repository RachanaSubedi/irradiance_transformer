# ════════════════════════════════════════════════════════════
# config/settings.py
# Single source of truth for all paths and hyperparameters.
# Import this in every script — never hardcode paths elsewhere.
# ════════════════════════════════════════════════════════════

import os
import torch

# ── Base data directory ──────────────────────────────────────
BASE_PATH = os.path.join(
    r"C:\Users\C838122727\Documents\CSU\research\IEEE 9500 bus Journal",
    "IEEE9500", "New 9500", "irradiance_transformer_project",
    "irradiance_transformer_project", "datasets"
)


def _p(filename: str) -> str:
    """
    Build a path under BASE_PATH using os.path.join, never string
    concatenation. This is the fix for the recurring PermissionError —
    the old code did `BASE_PATH + "/" + filename`, which mixes Windows
    backslashes with forward slashes and produces a malformed path
    that Windows sometimes refuses to open even though it looks valid
    when printed.
    """
    return os.path.normpath(os.path.join(BASE_PATH, filename))


# ── Raw data files ───────────────────────────────────────────
RAW = {
    # Local GHI (Ambient Weather, 5-min → resampled 30-min via snapshot,
    # NOT mean — see data.py process_station_utc for why)
    "ghi_s1": _p("46.59, -119.15 2024.csv"),
    "ghi_s2": _p("46.82, -119.15 2024.csv"),
    "ghi_s3": _p("46.82, -119.16 2024.csv"),
    "ghi_p2": _p("46.78, -119.22 2024.csv"),   # partial station

    # NSRDB (30-min, skiprows=2, LST → +8h → UTC)
    "nsrdb_s1": _p("768989_46.60_-119.15_2024.csv"),
    "nsrdb_s2": _p("767670_46.82_-119.17_2024.csv"),
    "nsrdb_s3": _p("768978_46.82_-119.15_2024.csv"),
    "nsrdb_p2": _p("763752_46.78_-119.23_2024.csv"),

    # GOES-18 C13 brightness temperature (30-min, GEE export)
    "c13_complete": _p("goes18_c13_s2_s3_30min_2024_gee.csv"),   # s1,s2,s3
    "c13_p2":       _p("goes18_c13_s1_30min_2024_gee_new.csv"),  # s5_c13 = P2
}

# ── Station coordinates ──────────────────────────────────────
STATIONS = {
    "s1": {"lat": 46.594029,  "lon": -119.152367, "alt": 120},
    "s2": {"lat": 46.823242,  "lon": -119.163197, "alt": 120},
    "s3": {"lat": 46.821036,  "lon": -119.150761, "alt": 120},
    "p2": {"lat": 46.780547,  "lon": -119.228783, "alt": 120},
}

# ── C13 column names per station ─────────────────────────────
C13_COLS = {
    "s1": "s1_c13",
    "s2": "s2_c13",
    "s3": "s3_c13",
    "p2": "s5_c13",
}

# ── Partial station (P2) fine-tuning config ──────────────────
P2 = {
    "anchor1":        "s3",               # nearest complete station
    "anchor2":        "s1",               # second nearest
    "overlap_start":  "2024-11-19 22:00", # UTC — when P2 data begins
    "overlap_end":    "2025-01-01 08:00", # UTC
    "val_start":      "2024-12-25 08:00", # UTC — hold-out validation
    "imp_start":      "2024-01-01 08:00", # UTC — start of missing period
    "imp_end":        "2024-11-19 22:00", # UTC — end of missing period
}

# ── Model architecture ───────────────────────────────────────
MODEL = {
    "n_features":  14,   # v2: 14 features (was 10 in v1)
    "seq_len":     48,   # 24 hours at 30-min resolution
    "center":      23,   # index of center timestep in window
    "d_model":     64,
    "n_heads":     4,
    "n_layers":    2,
    "d_ff":        128,
    "dropout":     0.1,
}

# ── Training hyperparameters ─────────────────────────────────
TRAIN = {
    "epochs":       100,
    "batch_size":   128,
    "lr":           3e-4,
    "patience":     15,
    "seed":         42,
}

FINETUNE = {
    "lr":              3e-5,
    "patience":        15,
    "real_repeat":     8,      # repeat real overlap sequences N times
    "batch_size":      64,
    # ── Tail-weighting for CSI loss (fixes high-CSI compression) ──
    # Samples with CSI above this threshold get extra loss weight,
    # which counteracts the systematic under-prediction of clear-sky
    # peaks that occurs with plain MSE/Huber on an imbalanced dataset
    # where >0.85 CSI samples are only ~5-7% of training data.
    "tail_threshold":  0.85,
    "tail_weight":     3.0,    # weight multiplier for samples above threshold
}

# ── Feature names (must match sequence builder order) ────────
FEATURE_NAMES = [
    "anchor1_CSI",       # [0]
    "anchor1_mask",      # [1]
    "anchor2_CSI",       # [2]
    "anchor2_mask",      # [3]
    "NSRDB_CSI_target",  # [4]
    "C13_norm_target",   # [5]
    "NSRDB_CSI_anchor1", # [6]  ← v2 new
    "C13_norm_anchor1",  # [7]  ← v2 new
    "NSRDB_CSI_anchor2", # [8]  ← v2 new
    "C13_norm_anchor2",  # [9]  ← v2 new
    "hour_sin",          # [10]
    "hour_cos",          # [11]
    "doy_sin",           # [12]
    "doy_cos",           # [13]
]

# ── Pretraining tasks (target, anchor1, anchor2) ─────────────
PRETRAIN_TASKS = [
    ("s1", "s2", "s3"),
    ("s2", "s3", "s1"),
    ("s3", "s2", "s1"),
]

# ── Saved artifact paths ─────────────────────────────────────
ARTIFACTS = {
    # v1
    "pretrain_model_v1":   _p("pretrain_best_model.pt"),
    "ghi_csv_v1":          _p("station_46_78_full_year_GHI_v1_RMSE0996.csv"),
    "ft_model_v1":         _p("finetune_frozen_46_78_synth_v1_RMSE0996.pt"),

    # v2
    "x_pretrain_v2":       _p("X_pretrain_v2.npy"),
    "y_pretrain_v2":       _p("y_pretrain_v2.npy"),
    "meta_pretrain_v2":    _p("meta_pretrain_v2.csv"),
    "pretrain_model_v2":   _p("pretrain_best_model_v2.pt"),
    "ft_model_v2":         _p("finetune_frozen_46_78_synth_v2.pt"),
    "ghi_csv_v2":          _p("station_46_78_full_year_GHI_v2.csv"),
    "plot_v2":             _p("full_year_GHI_analysis_v3.png"),

    # v3 (tail-weighted retrain — this audit)
    "x_pretrain_v3":       _p("X_pretrain_v3.npy"),
    "y_pretrain_v3":       _p("y_pretrain_v3.npy"),
    "meta_pretrain_v3":    _p("meta_pretrain_v3.csv"),
    "pretrain_model_v3":   _p("pretrain_best_model_v3.pt"),
    "ft_model_v3":         _p("finetune_frozen_46_78_synth_v3.pt"),
    "ghi_csv_v3":          _p("station_46_78_full_year_GHI_v3.csv"),
    "plot_v3":             _p("finetune_results_v3.png"),
}

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")