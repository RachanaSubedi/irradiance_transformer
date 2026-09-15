"""Dynamic window dataset for target-aware irradiance imputation.

This module joins, without copying the annual stores:
1. station_model_ready_2024.npz (station observations and physics), and
2. goes_token_store_2024/*.npy (quality-aware 5x5 GOES tokens).

It does not select a model architecture or normalize from the full year.
Split masks and train-only normalization belong to the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


STATION_NAMES = ("S1", "S2", "S3", "P2")


@dataclass(frozen=True)
class WindowSpec:
    seq_len: int = 72
    center: int = 35

    def __post_init__(self) -> None:
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if not 0 <= self.center < self.seq_len:
            raise ValueError("center must lie inside the sequence")

    @property
    def left(self) -> int:
        return self.center

    @property
    def right(self) -> int:
        return self.seq_len - self.center - 1


def _copy_tensor(array: np.ndarray, dtype: torch.dtype | None = None) -> torch.Tensor:
    # Memmap views are read-only. Copying only the requested window avoids
    # PyTorch's non-writable-array warning and leaves the annual store untouched.
    tensor = torch.from_numpy(np.array(array, copy=True))
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _load_station_store(path: Path) -> dict[str, np.ndarray]:
    required = {
        "time_utc_ns",
        "station_names",
        "station_csi",
        "station_csi_target",
        "station_csi_valid_mask",
        "time_since_csi_observed_minutes",
        "clear_sky_ghi",
        "cos_zenith",
        "meteorology",
        "meteorology_valid_mask",
        "time_encoding",
        "station_latitude",
        "station_longitude",
        "station_elevation_m",
        "pairwise_distance_km",
        "pairwise_east_km",
        "pairwise_north_km",
        "pixel_relative_y",
        "pixel_relative_x",
        "idw_csi",
        "idw_available_mask",
        "idw_source_count",
    }
    with np.load(path) as z:
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(f"{path}: missing arrays {missing}")
        return {name: np.array(z[name], copy=True) for name in required}


class SpatiotemporalWindowDataset(Dataset):
    """Return quality-aware station/cloud windows for one target at a time.

    Parameters
    ----------
    allowed_time_mask:
        Boolean mask over the annual timeline. A sample is retained only when
        every timestamp in its window belongs to the allowed split. This
        prevents train/validation boundary leakage.
    target_stations:
        Stations cycled as reconstruction targets. Use S1/S2/S3 for
        pseudo-target pretraining and P2 for evaluation/inference.
    whole_target_blackout:
        If True, all target-station CSI values in the input window are hidden.
        Truth remains available separately for loss calculation.
    allow_p2_as_source:
        Defaults False, preventing measured P2 values from leaking into
        pseudo-target pretraining. P2 GOES patches and geometry remain present.
    include_idw:
        Defaults False. When False, IDW is not returned as an active input.
        Enabling it later supports a controlled IDW ablation without rebuilding.
    """

    def __init__(
        self,
        station_store: str | Path,
        cloud_store: str | Path,
        *,
        window: WindowSpec = WindowSpec(),
        target_stations: Sequence[str] = ("S1", "S2", "S3"),
        allowed_time_mask: np.ndarray | None = None,
        whole_target_blackout: bool = True,
        allow_p2_as_source: bool = False,
        include_idw: bool = False,
        require_center_truth: bool = True,
        daylight_only: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.station_path = Path(station_store)
        self.cloud_root = Path(cloud_store)
        self.window = window
        self.whole_target_blackout = whole_target_blackout
        self.allow_p2_as_source = allow_p2_as_source
        self.include_idw = include_idw

        self.station = _load_station_store(self.station_path)
        names = tuple(str(x).upper() for x in self.station["station_names"])
        if names != STATION_NAMES:
            raise ValueError(f"station order {names}; expected {STATION_NAMES}")
        self.station_names = names
        self.station_to_index = {name: i for i, name in enumerate(names)}
        self.target_indices = tuple(
            self.station_to_index[name.upper()] for name in target_stations
        )

        self.cloud = {
            "continuous": np.load(
                self.cloud_root / "cloud_continuous.npy", mmap_mode="r"
            ),
            "continuous_mask": np.load(
                self.cloud_root / "cloud_continuous_valid_mask.npy",
                mmap_mode="r",
            ),
            "categorical": np.load(
                self.cloud_root / "cloud_categorical_token.npy", mmap_mode="r"
            ),
            "categorical_mask": np.load(
                self.cloud_root / "cloud_categorical_valid_mask.npy",
                mmap_mode="r",
            ),
            "cod_quality_token": np.load(
                self.cloud_root / "cod_quality_token.npy", mmap_mode="r"
            ),
            "cod_is_night": np.load(
                self.cloud_root / "cod_is_night.npy", mmap_mode="r"
            ),
            "phase_quality_bits": np.load(
                self.cloud_root / "phase_quality_bits.npy", mmap_mode="r"
            ),
            "product_matched_mask": np.load(
                self.cloud_root / "product_matched_mask.npy", mmap_mode="r"
            ),
        }
        cloud_time = np.load(self.cloud_root / "time_utc_ns.npy", mmap_mode="r")
        station_time = self.station["time_utc_ns"]
        if not np.array_equal(cloud_time, station_time):
            raise ValueError("station and GOES timelines are not identical")
        self.time_utc_ns = station_time
        self.n_time = len(station_time)

        if allowed_time_mask is None:
            allowed = np.ones(self.n_time, dtype=bool)
        else:
            allowed = np.asarray(allowed_time_mask, dtype=bool)
            if allowed.shape != (self.n_time,):
                raise ValueError(
                    f"allowed_time_mask shape {allowed.shape}; "
                    f"expected {(self.n_time,)}"
                )
        self.allowed_time_mask = allowed

        # Prefix sum permits O(1) verification that the complete window lies
        # inside one split, rather than merely checking its center timestamp.
        disallowed_prefix = np.concatenate(
            ([0], np.cumsum(~allowed, dtype=np.int64))
        )
        centers = np.arange(
            window.left,
            self.n_time - window.right,
            dtype=np.int64,
        )
        starts = centers - window.left
        stops = centers + window.right + 1
        complete_window_allowed = (
            disallowed_prefix[stops] - disallowed_prefix[starts]
        ) == 0
        centers = centers[complete_window_allowed]

        truth_mask = self.station["station_csi_valid_mask"].astype(bool)
        records: list[np.ndarray] = []
        for target in self.target_indices:
            keep = np.ones(len(centers), dtype=bool)
            if require_center_truth:
                keep &= truth_mask[centers, target]
            if daylight_only:
                keep &= self.station["clear_sky_ghi"][centers, target] >= 10.0
            selected = centers[keep]
            target_column = np.full(len(selected), target, dtype=np.int64)
            records.append(np.column_stack((selected, target_column)))

        self.records = (
            np.concatenate(records, axis=0)
            if records
            else np.empty((0, 2), dtype=np.int64)
        )
        if max_samples is not None:
            if max_samples < 0:
                raise ValueError("max_samples must be nonnegative")
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def _source_station_mask(self, target: int) -> np.ndarray:
        allowed = np.ones(len(self.station_names), dtype=np.uint8)
        allowed[target] = 0
        if not self.allow_p2_as_source and target != self.station_to_index["P2"]:
            allowed[self.station_to_index["P2"]] = 0
        return allowed

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        center, target = (int(x) for x in self.records[item])
        start = center - self.window.left
        stop = center + self.window.right + 1
        sl = slice(start, stop)

        truth = self.station["station_csi_target"][sl]
        truth_mask = self.station["station_csi_valid_mask"][sl].astype(np.uint8)
        input_csi = np.array(self.station["station_csi"][sl], copy=True)
        input_mask = np.array(truth_mask, copy=True)
        elapsed = np.array(
            self.station["time_since_csi_observed_minutes"][sl], copy=True
        )
        meteorology = np.array(self.station["meteorology"][sl], copy=True)
        meteorology_mask = np.array(
            self.station["meteorology_valid_mask"][sl], copy=True
        )

        source_station_mask = self._source_station_mask(target)
        hidden_station_mask = source_station_mask == 0

        if self.whole_target_blackout:
            input_csi[:, target] = 0.0
            input_mask[:, target] = 0
            elapsed[:, target] = 1440.0
            # Simulate a genuinely unavailable target station. P2 has no
            # target-site meteorology during its long pre-installation gap.
            meteorology[:, target] = 0.0
            meteorology_mask[:, target] = 0

        # P2 truth is reserved from pseudo-target pretraining by default.
        for station_index in np.flatnonzero(hidden_station_mask):
            if station_index != target:
                input_csi[:, station_index] = 0.0
                input_mask[:, station_index] = 0
                elapsed[:, station_index] = 1440.0
                meteorology[:, station_index] = 0.0
                meteorology_mask[:, station_index] = 0

        result = {
            "station_csi": _copy_tensor(input_csi, torch.float32),
            "station_csi_mask": _copy_tensor(input_mask, torch.bool),
            "time_since_observed_minutes": _copy_tensor(elapsed, torch.float32),
            "clear_sky_ghi": _copy_tensor(
                self.station["clear_sky_ghi"][sl], torch.float32
            ),
            "cos_zenith": _copy_tensor(
                self.station["cos_zenith"][sl], torch.float32
            ),
            "meteorology": _copy_tensor(
                meteorology, torch.float32
            ),
            "meteorology_mask": _copy_tensor(
                meteorology_mask, torch.bool
            ),
            "time_encoding": _copy_tensor(
                self.station["time_encoding"][sl], torch.float32
            ),
            "cloud_continuous": _copy_tensor(
                self.cloud["continuous"][sl], torch.float32
            ),
            "cloud_continuous_mask": _copy_tensor(
                self.cloud["continuous_mask"][sl], torch.bool
            ),
            "cloud_categorical_token": _copy_tensor(
                self.cloud["categorical"][sl], torch.long
            ),
            "cloud_categorical_mask": _copy_tensor(
                self.cloud["categorical_mask"][sl], torch.bool
            ),
            "cod_quality_token": _copy_tensor(
                self.cloud["cod_quality_token"][sl], torch.long
            ),
            "cod_is_night": _copy_tensor(
                self.cloud["cod_is_night"][sl], torch.bool
            ),
            "phase_quality_bits": _copy_tensor(
                self.cloud["phase_quality_bits"][sl], torch.bool
            ),
            "product_matched_mask": _copy_tensor(
                self.cloud["product_matched_mask"][sl], torch.bool
            ),
            "target_csi_sequence": _copy_tensor(
                truth[:, target], torch.float32
            ),
            "target_truth_mask": _copy_tensor(
                truth_mask[:, target], torch.bool
            ),
            "target_csi_center": _copy_tensor(
                np.asarray(truth[self.window.center, target]), torch.float32
            ),
            "target_station_index": torch.tensor(target, dtype=torch.long),
            "source_station_mask": _copy_tensor(
                source_station_mask, torch.bool
            ),
            "center_time_utc_ns": torch.tensor(
                int(self.time_utc_ns[center]), dtype=torch.long
            ),
            "center_index": torch.tensor(center, dtype=torch.long),
            # Static geometry is returned per sample so default DataLoader
            # collation works without a custom batch object.
            "station_coordinates": _copy_tensor(
                np.stack(
                    [
                        self.station["station_latitude"],
                        self.station["station_longitude"],
                        self.station["station_elevation_m"],
                    ],
                    axis=1,
                ),
                torch.float32,
            ),
            "pairwise_distance_km": _copy_tensor(
                self.station["pairwise_distance_km"], torch.float32
            ),
            "pairwise_east_km": _copy_tensor(
                self.station["pairwise_east_km"], torch.float32
            ),
            "pairwise_north_km": _copy_tensor(
                self.station["pairwise_north_km"], torch.float32
            ),
            "pixel_relative_y": _copy_tensor(
                self.station["pixel_relative_y"], torch.float32
            ),
            "pixel_relative_x": _copy_tensor(
                self.station["pixel_relative_x"], torch.float32
            ),
        }

        if self.include_idw:
            result.update(
                {
                    "idw_csi": _copy_tensor(
                        self.station["idw_csi"][sl, target], torch.float32
                    ),
                    "idw_available_mask": _copy_tensor(
                        self.station["idw_available_mask"][sl, target],
                        torch.bool,
                    ),
                    "idw_source_count": _copy_tensor(
                        self.station["idw_source_count"][sl, target],
                        torch.long,
                    ),
                }
            )

        return result


def contiguous_time_mask(
    time_utc_ns: np.ndarray,
    start: str,
    stop: str,
) -> np.ndarray:
    """Return [start, stop) UTC mask for split-before-window construction."""
    start_ns = np.datetime64(start, "ns").astype(np.int64)
    stop_ns = np.datetime64(stop, "ns").astype(np.int64)
    if stop_ns <= start_ns:
        raise ValueError("stop must be later than start")
    times = np.asarray(time_utc_ns, dtype=np.int64)
    return (times >= start_ns) & (times < stop_ns)
