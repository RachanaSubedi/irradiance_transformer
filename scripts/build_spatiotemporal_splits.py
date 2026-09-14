#!/usr/bin/env python3
"""Create reproducible blocked-day splits and train-only feature statistics.

Each calendar month contributes one contiguous validation block and one
contiguous test block. The same timestamp masks are used for every pseudo-target
station. Dynamic windows are later retained only when every timestamp in the
window belongs to one split, preventing boundary leakage.

Normalization statistics use training timestamps and S1/S2/S3 only. P2 labels
and P2 satellite pixels do not influence preprocessing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from irradiance import config as cfg


TRAIN_STATION_INDICES = (0, 1, 2)
SPLIT_NAMES = ("train", "validation", "test")


class OnlineMoments:
    def __init__(self, n_features: int) -> None:
        self.count = np.zeros(n_features, dtype=np.int64)
        self.total = np.zeros(n_features, dtype=np.float64)
        self.total_sq = np.zeros(n_features, dtype=np.float64)

    def update(self, values: np.ndarray, mask: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        if values.shape != mask.shape:
            raise ValueError("value/mask shape mismatch")
        flat_values = values.reshape(-1, values.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        safe = np.where(flat_mask, flat_values, 0.0)
        self.count += flat_mask.sum(axis=0)
        self.total += safe.sum(axis=0)
        self.total_sq += (safe * safe).sum(axis=0)

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.any(self.count == 0):
            raise ValueError(f"features with zero observations: {self.count}")
        mean = self.total / self.count
        variance = np.maximum(self.total_sq / self.count - mean * mean, 0.0)
        std = np.sqrt(variance)
        std = np.where(std < 1e-8, 1.0, std)
        return mean, std, self.count


def choose_monthly_blocks(
    timeline: pd.DatetimeIndex,
    block_days: int,
    seed: int,
    separation_days: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    normalized_days = timeline.normalize()
    assignments = np.full(len(timeline), "train", dtype="<U10")
    selected: dict[str, list[str]] = {"validation": [], "test": []}

    for month in range(1, 13):
        month_days = pd.DatetimeIndex(
            sorted(normalized_days[normalized_days.month == month].unique())
        )
        if len(month_days) < 2 * block_days + separation_days:
            raise ValueError(f"month {month}: insufficient days")

        starts = np.arange(0, len(month_days) - block_days + 1)
        test_start = int(rng.choice(starts))
        test_positions = set(range(test_start, test_start + block_days))

        valid_starts = []
        for start in starts:
            positions = set(range(int(start), int(start) + block_days))
            expanded_test = set()
            for position in test_positions:
                expanded_test.update(
                    range(
                        max(0, position - separation_days),
                        min(len(month_days), position + separation_days + 1),
                    )
                )
            if positions.isdisjoint(expanded_test):
                valid_starts.append(int(start))
        if not valid_starts:
            raise RuntimeError(f"month {month}: unable to place validation block")

        validation_start = int(rng.choice(valid_starts))
        blocks = {
            "test": month_days[test_start : test_start + block_days],
            "validation": month_days[
                validation_start : validation_start + block_days
            ],
        }

        for split, days in blocks.items():
            selected[split].extend(day.strftime("%Y-%m-%d") for day in days)
            assignments[np.isin(normalized_days, days)] = split

    masks = {name: assignments == name for name in SPLIT_NAMES}
    if np.any(sum(mask.astype(np.uint8) for mask in masks.values()) != 1):
        raise AssertionError("split masks are not mutually exclusive/exhaustive")
    return masks, selected


def complete_window_centers(mask: np.ndarray, seq_len: int, center: int) -> int:
    left = center
    right = seq_len - center - 1
    candidate = np.arange(left, len(mask) - right)
    starts = candidate - left
    stops = candidate + right + 1
    prefix = np.concatenate(([0], np.cumsum(~mask, dtype=np.int64)))
    return int(np.sum((prefix[stops] - prefix[starts]) == 0))


def compute_statistics(
    station_path: Path,
    cloud_root: Path,
    train_mask: np.ndarray,
    chunk_rows: int,
) -> dict:
    with np.load(station_path) as z:
        meteorology = np.asarray(z["meteorology"], dtype=np.float32)
        meteorology_mask = np.asarray(
            z["meteorology_valid_mask"], dtype=bool
        )
        meteorology_names = [str(x) for x in z["meteorology_names"]]

    cloud = np.load(cloud_root / "cloud_continuous.npy", mmap_mode="r")
    cloud_mask = np.load(
        cloud_root / "cloud_continuous_valid_mask.npy", mmap_mode="r"
    )
    cloud_names = json.loads(
        (cloud_root / "manifest.json").read_text()
    )["cloud_continuous_channels"]

    cloud_moments = OnlineMoments(len(cloud_names))
    met_moments = OnlineMoments(len(meteorology_names))
    train_indices = np.flatnonzero(train_mask)

    for first in range(0, len(train_indices), chunk_rows):
        rows = train_indices[first : first + chunk_rows]

        cloud_values = np.asarray(
            cloud[rows][:, TRAIN_STATION_INDICES], dtype=np.float32
        )
        cloud_valid = np.asarray(
            cloud_mask[rows][:, TRAIN_STATION_INDICES], dtype=bool
        )
        cloud_moments.update(cloud_values, cloud_valid)

        met_values = meteorology[rows][:, TRAIN_STATION_INDICES]
        met_valid = meteorology_mask[rows][:, TRAIN_STATION_INDICES]
        met_moments.update(met_values, met_valid)

    cloud_mean, cloud_std, cloud_count = cloud_moments.result()
    met_mean, met_std, met_count = met_moments.result()

    def pack(names, mean, std, count):
        return {
            name: {
                "mean": float(mean[i]),
                "std": float(std[i]),
                "count": int(count[i]),
            }
            for i, name in enumerate(names)
        }

    return {
        "cloud_continuous": pack(
            cloud_names, cloud_mean, cloud_std, cloud_count
        ),
        "meteorology": pack(
            meteorology_names, met_mean, met_std, met_count
        ),
        "fixed_scaling": {
            "station_csi": {
                "operation": "none",
                "expected_range": [0.0, 1.3],
            },
            "clear_sky_ghi": {
                "operation": "divide",
                "divisor": 1000.0,
            },
            "cos_zenith": {
                "operation": "none",
                "expected_range": [0.0, 1.0],
            },
            "time_since_observed_minutes": {
                "operation": "log1p_then_divide",
                "divisor": float(np.log1p(1440.0)),
                "cap_minutes": 1440.0,
            },
            "time_encoding": {
                "operation": "none",
                "expected_range": [-1.0, 1.0],
            },
        },
    }


def main(args: argparse.Namespace) -> None:
    with np.load(args.station_store) as z:
        times_ns = np.asarray(z["time_utc_ns"], dtype=np.int64)
        csi_valid = np.asarray(z["station_csi_valid_mask"], dtype=bool)

    timeline = pd.to_datetime(times_ns, utc=True)
    masks, selected_days = choose_monthly_blocks(
        timeline,
        block_days=args.block_days,
        seed=args.seed,
        separation_days=args.separation_days,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        np.save(args.output / f"{name}_time_mask.npy", mask.astype(np.uint8))

    statistics = compute_statistics(
        args.station_store,
        args.cloud_store,
        masks["train"],
        args.chunk_rows,
    )
    (args.output / "normalization_train_only.json").write_text(
        json.dumps(statistics, indent=2) + "\n"
    )

    split_summary = {}
    for name, mask in masks.items():
        split_summary[name] = {
            "timestamps": int(mask.sum()),
            "days": int(pd.DatetimeIndex(timeline[mask]).normalize().nunique()),
            "complete_window_centers": complete_window_centers(
                mask, args.seq_len, args.center
            ),
            "valid_center_targets": {
                station: int(np.sum(mask & csi_valid[:, i]))
                for i, station in enumerate(("S1", "S2", "S3", "P2"))
            },
        }

    manifest = {
        "complete": True,
        "year": 2024,
        "strategy": (
            "one deterministic contiguous validation block and one test "
            "block per calendar month; remaining days train"
        ),
        "seed": args.seed,
        "block_days": args.block_days,
        "separation_days_between_validation_and_test": args.separation_days,
        "interval_convention": "UTC calendar days; masks are mutually exclusive",
        "window_rule": (
            "retain a center only if its entire sequence window belongs "
            "to the same split"
        ),
        "seq_len": args.seq_len,
        "center": args.center,
        "shared_across_pseudo_targets": True,
        "normalization_scope": {
            "timestamps": "training mask only",
            "stations": ["S1", "S2", "S3"],
            "p2_excluded": True,
        },
        "selected_days": selected_days,
        "summary": split_summary,
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(json.dumps(manifest, indent=2))
    print(f"SAVED {args.output / 'split_manifest.json'}")
    print(f"SAVED {args.output / 'normalization_train_only.json'}")


def parse_args() -> argparse.Namespace:
    root = (
        Path(cfg.BASE_PATH)
        / "spatiotemporal_dataset"
        / "washington_2024"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--station-store",
        type=Path,
        default=root / "station_model_ready_2024.npz",
    )
    parser.add_argument(
        "--cloud-store",
        type=Path,
        default=root / "goes_token_store_2024",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "splits_v1",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-days", type=int, default=3)
    parser.add_argument("--separation-days", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--center", type=int, default=35)
    parser.add_argument("--chunk-rows", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
