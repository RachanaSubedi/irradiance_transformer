# ════════════════════════════════════════════════════════════
# config/settings.py
# Single source of truth for all paths and hyperparameters.
# Import this in every script — never hardcode paths elsewhere.
# ════════════════════════════════════════════════════════════

import os
import torch

# ── Base data directory ──────────────────────────────────────
# Auto-detects OS so this same file works unmodified on both your
# local Windows machine and Cashew (or any Linux cluster) — no
# manual editing needed when switching environments.
#
# If you clone this repo to a different path on Cashew than
# /home/C838122727/irradiance_transformer, update the Linux
# branch below to match.
if os.name == "nt":
    # Windows (local machine)
    BASE_PATH = os.path.join(
        r"C:\Users\C838122727\Documents\CSU\research\IEEE 9500 bus Journal",
        "IEEE9500", "New 9500", "irradiance_transformer_project",
        "irradiance_transformer_project", "datasets"
    )
else:
    # Linux (Cashew / any HPC cluster)
    BASE_PATH = "/home/C838122727/irradiance_transformer/datasets"


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
    # Local GHI (Ambient Weather, native 5-min — unchanged format,
    # now used at native resolution instead of downsampled to 30-min)
    "ghi_s1": _p("46.59, -119.15 2024.csv"),
    "ghi_s2": _p("46.82, -119.15 2024.csv"),
    "ghi_s3": _p("46.82, -119.16 2024.csv"),
    "ghi_p2": _p("46.78, -119.22 2024.csv"),   # partial station

    # NSRDB — NEW 5-min files, UTC timestamps (no +8h shift needed,
    # unlike the old 30-min files which were LST/UTC-8). Verified via
    # solar-noon check: all 4 files peak Clearsky GHI at 19:55 UTC on
    # Jan 1, confirming UTC despite the file header's metadata still
    # showing "Time Zone, -8" (that field is unused/stale in this
    # export, not reflective of the actual timestamp convention).
    "nsrdb_s1": _p("174133_46.58_-119.15_2024.csv"),
    "nsrdb_s2": _p("174121_46.82_-119.15_2024.csv"),
    "nsrdb_s3": _p("173362_46.82_-119.17_2024.csv"),
    "nsrdb_p2": _p("171848_46.78_-119.21_2024.csv"),

    # GOES-18 C13+C02 — NEW one-file-per-pixel structure (5-min).
    # S2 and S3 share the SAME pixel file (confirmed: they fall in
    # the same satellite grid cell at this resolution, same as the
    # old 30-min data — this is a real physical redundancy, not a
    # bug). Raw values are uncalibrated DN; C13 is divided by 10 to
    # recover Kelvin inside process_c13_c02_utc (verified: matches
    # the old calibrated mean of ~290-291K for all 4 pixels). C02
    # has no documented divisor — used as raw DN, z-scored same as
    # C13. S1's C02 has ~0.95% missing values spread evenly across
    # all hours (benign satellite scan gaps, not a systematic issue)
    # — handled the same way C13 gaps already are.
    "c13c02_s1": _p("goes18_c13_c02_px_46p594595_n119p155673.csv"),
    "c13c02_s2": _p("goes18_c13_c02_px_46p828829_n119p155673.csv"),
    "c13c02_s3": _p("goes18_c13_c02_px_46p828829_n119p155673.csv"),  # same file as S2
    "c13c02_p2": _p("goes18_c13_c02_px_46p774775_n119p234828.csv"),
}

# ── GOES-18 calibration ───────────────────────────────────────
# bt_c13_raw is uncalibrated DN (digital counts), not Kelvin.
# Verified: dividing by 10 recovers values matching the OLD
# pipeline's already-calibrated C13 (mean ~291K both before and
# after this conversion on the new data).
C13_DN_TO_KELVIN_DIVISOR = 10.0
# C02 (reflectance band) has no documented divisor — used as raw
# DN, z-scored the same way C13 is. Revisit if a calibration
# coefficient becomes available.

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
# v4: switched from 30-min to 5-min native resolution + added C02.
#
# WINDOW LENGTH — IMPORTANT DECISION:
# v1-v3 used seq_len=48 (24 hours at 30-min). Naively keeping "24
# hours of context" at 5-min would mean seq_len=288 — a 6x longer
# sequence, costing ~36x more attention compute per layer (cost
# scales with seq_len^2), and forcing the model to learn very
# long-range dependencies (cloud cover 23 hours away is unlikely to
# be predictive of right now) in a single window.
#
# Default here is a 6-HOUR window (seq_len=72) instead — short
# enough to keep attention cost manageable, long enough to capture
# the relevant cloud-evolution timescale. CENTER is the middle of
# whatever seq_len you choose: center = seq_len // 2 - 1.
#
# If you want to test the full 24h-at-5min approach instead, set
# seq_len=288, center=143, and expect significantly longer training
# time — and expect to re-tune d_model/n_layers, since 288 tokens
# may need more capacity than 64-dim embeddings provide.
MODEL = {
    "n_features":  17,   # v4: 17 features (14 + 3 C02 slots: target/anchor1/anchor2)
    "seq_len":     72,   # 6 hours at 5-min resolution (was 48 @ 30-min = 24h)
    "center":      35,   # seq_len // 2 - 1
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
    "lr":           1e-4,   # lowered from 3e-4 — val loss was oscillating
                              # with increasing amplitude after epoch 2,
                              # classic too-high-LR signature on the
                              # larger 5-min dataset
    "patience":     6,       # tightened from 15 for faster iteration
                              # while tuning; raise back once LR is stable
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
# v4: added C02_norm_target/anchor1/anchor2, parallel to the
# existing C13_norm_* pattern (new reflectance feature from the
# C13+C02 combined GOES extraction).
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
    "C02_norm_target",   # [14] ← v4 new
    "C02_norm_anchor1",  # [15] ← v4 new
    "C02_norm_anchor2",  # [16] ← v4 new
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

    # v4 (5-min resolution + C02 feature)
    "x_pretrain_v4":       _p("X_pretrain_v4.npy"),
    "y_pretrain_v4":       _p("y_pretrain_v4.npy"),
    "meta_pretrain_v4":    _p("meta_pretrain_v4.csv"),
    "pretrain_model_v4":   _p("pretrain_best_model_v4.pt"),
    "ft_model_v4":         _p("finetune_frozen_46_78_synth_v4.pt"),
    "ghi_csv_v4":          _p("station_46_78_full_year_GHI_v4.csv"),
    "plot_v4":             _p("finetune_results_v4.png"),
}

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")