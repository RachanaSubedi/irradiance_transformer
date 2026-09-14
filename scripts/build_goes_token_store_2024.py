#!/usr/bin/env python3
"""Convert synchronized daily GOES L2 patches into an annual token store.

The store is deliberately model-agnostic:
* continuous cloud variables are float32 and zero-filled with explicit masks;
* categorical products use 0 for missing and positive integer class tokens;
* packed DQF values are decoded into meaningful categories/bits;
* arrays are ordinary .npy files opened by PyTorch through numpy memmap.

No station measurements, NSRDB values, IDW values, normalization statistics, or
sliding-window duplication are written here.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from irradiance import config as cfg


STATIONS = ("S1", "S2", "S3", "P2")
N_STATIONS = 4
PATCH_SIDE = 5
N_PIXELS = 25
N_SLOTS_PER_DAY = 288
FIVE_MINUTES_NS = 300_000_000_000
COD_USABLE_STATUS = np.asarray((0, 2, 4, 10, 16), dtype=np.uint8)


def annual_timeline_ns(year: int) -> np.ndarray:
    start = np.datetime64(f"{year}-01-01T00:00:00", "ns").astype(np.int64)
    stop = np.datetime64(f"{year + 1}-01-01T00:00:00", "ns").astype(np.int64)
    count = int((stop - start) // FIVE_MINUTES_NS)
    return start + np.arange(count, dtype=np.int64) * FIVE_MINUTES_NS


def open_output_arrays(root: Path, n_time: int) -> dict[str, np.memmap]:
    specs = {
        # Last axis: [log1p(COD), cloud probability].
        "cloud_continuous.npy": ((n_time, N_STATIONS, N_PIXELS, 2), np.float32),
        "cloud_continuous_valid_mask.npy": (
            (n_time, N_STATIONS, N_PIXELS, 2), np.uint8
        ),
        # Last axis: [ACM, BCM, phase]. Token 0 is missing.
        "cloud_categorical_token.npy": (
            (n_time, N_STATIONS, N_PIXELS, 3), np.uint8
        ),
        "cloud_categorical_valid_mask.npy": (
            (n_time, N_STATIONS, N_PIXELS, 3), np.uint8
        ),
        # COD DQF status token: 0 missing, otherwise status/2 + 1.
        "cod_quality_token.npy": ((n_time, N_STATIONS, N_PIXELS), np.uint8),
        "cod_dqf_valid_mask.npy": ((n_time, N_STATIONS, N_PIXELS), np.uint8),
        "cod_is_night.npy": ((n_time, N_STATIONS, N_PIXELS), np.uint8),
        "cod_strict_quality_mask.npy": (
            (n_time, N_STATIONS, N_PIXELS), np.uint8
        ),
        "cod_usable_quality_mask.npy": (
            (n_time, N_STATIONS, N_PIXELS), np.uint8
        ),
        # Phase bit order documented in manifest.
        "phase_quality_bits.npy": (
            (n_time, N_STATIONS, N_PIXELS, 6), np.uint8
        ),
        "phase_dqf_valid_mask.npy": (
            (n_time, N_STATIONS, N_PIXELS), np.uint8
        ),
        "phase_strict_quality_mask.npy": (
            (n_time, N_STATIONS, N_PIXELS), np.uint8
        ),
        # Product order: CODC, ACMC, ACTPC.
        "product_matched_mask.npy": ((n_time, 3), np.uint8),
        "scan_time_offset_seconds.npy": ((n_time, 3), np.int32),
    }
    return {
        name: np.lib.format.open_memmap(
            root / name, mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in specs.items()
    }


def finite_dqf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    decoded = np.zeros(values.shape, dtype=np.uint8)
    decoded[finite] = np.rint(values[finite]).astype(np.uint8)
    return decoded, finite


def valid_numeric(
    values: np.ndarray,
    supplied_mask: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    return (
        supplied_mask.astype(bool)
        & np.isfinite(values)
        & (values >= minimum)
        & (values <= maximum)
    )


def valid_category(
    values: np.ndarray,
    supplied_mask: np.ndarray,
    minimum: int,
    maximum: int,
) -> np.ndarray:
    return (
        supplied_mask.astype(bool)
        & np.isfinite(values)
        & (values >= minimum)
        & (values <= maximum)
        & (values == np.rint(values))
    )


def flatten_patch(values: np.ndarray) -> np.ndarray:
    if values.shape[1:] != (N_STATIONS, PATCH_SIDE, PATCH_SIDE):
        raise ValueError(f"unexpected patch shape {values.shape}")
    return values.reshape(values.shape[0], N_STATIONS, N_PIXELS)


def process_day(
    source: Path,
    output: dict[str, np.memmap],
    expected_time: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, int]:
    with np.load(source) as z:
        required = {
            "time_utc_ns", "station_names",
            "cod", "cod_dqf", "cloud_probability", "acm", "bcm",
            "acm_dqf", "phase", "phase_dqf",
            "cod_valid_mask", "cloud_probability_valid_mask",
            "acm_valid_mask", "bcm_valid_mask", "phase_valid_mask",
            "codc_scan_matched", "codc_time_offset_seconds",
            "acmc_scan_matched", "acmc_time_offset_seconds",
            "actpc_scan_matched", "actpc_time_offset_seconds",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(f"{source}: missing arrays {missing}")

        day_time = np.asarray(z["time_utc_ns"], dtype=np.int64)
        if len(day_time) != N_SLOTS_PER_DAY:
            raise ValueError(f"{source}: expected 288 rows, found {len(day_time)}")
        if not np.array_equal(day_time, expected_time[start:stop]):
            raise ValueError(f"{source}: timeline does not match annual 5-min grid")

        names = tuple(str(x).upper() for x in z["station_names"])
        if names != STATIONS:
            raise ValueError(f"{source}: station order {names}, expected {STATIONS}")

        cod = flatten_patch(np.asarray(z["cod"], dtype=np.float32))
        probability = flatten_patch(
            np.asarray(z["cloud_probability"], dtype=np.float32)
        )
        acm = flatten_patch(np.asarray(z["acm"], dtype=np.float32))
        bcm = flatten_patch(np.asarray(z["bcm"], dtype=np.float32))
        phase = flatten_patch(np.asarray(z["phase"], dtype=np.float32))

        cod_supplied = flatten_patch(np.asarray(z["cod_valid_mask"]))
        probability_supplied = flatten_patch(
            np.asarray(z["cloud_probability_valid_mask"])
        )
        acm_supplied = flatten_patch(np.asarray(z["acm_valid_mask"]))
        bcm_supplied = flatten_patch(np.asarray(z["bcm_valid_mask"]))
        phase_supplied = flatten_patch(np.asarray(z["phase_valid_mask"]))

        cod_dqf, cod_dqf_valid = finite_dqf(
            flatten_patch(np.asarray(z["cod_dqf"], dtype=np.float32))
        )
        phase_dqf, phase_dqf_valid = finite_dqf(
            flatten_patch(np.asarray(z["phase_dqf"], dtype=np.float32))
        )

        cod_status = cod_dqf & np.uint8(30)
        cod_observed = valid_numeric(cod, cod_supplied, 0.0, 200.0)
        cod_usable = (
            cod_observed
            & cod_dqf_valid
            & np.isin(cod_status, COD_USABLE_STATUS)
        )
        cod_strict = cod_observed & cod_dqf_valid & (cod_status == 0)
        probability_valid = valid_numeric(
            probability, probability_supplied, 0.0, 1.0
        )
        acm_valid = valid_category(acm, acm_supplied, 0, 3)
        bcm_valid = valid_category(bcm, bcm_supplied, 0, 1)
        phase_valid = valid_category(phase, phase_supplied, 0, 5)

        continuous = np.zeros((*cod.shape, 2), dtype=np.float32)
        continuous[..., 0][cod_usable] = np.log1p(cod[cod_usable])
        continuous[..., 1][probability_valid] = probability[probability_valid]
        output["cloud_continuous.npy"][start:stop] = continuous
        output["cloud_continuous_valid_mask.npy"][start:stop, ..., 0] = cod_usable
        output["cloud_continuous_valid_mask.npy"][start:stop, ..., 1] = (
            probability_valid
        )

        categorical = np.zeros((*cod.shape, 3), dtype=np.uint8)
        categorical[..., 0][acm_valid] = (
            np.rint(acm[acm_valid]).astype(np.uint8) + 1
        )
        categorical[..., 1][bcm_valid] = (
            np.rint(bcm[bcm_valid]).astype(np.uint8) + 1
        )
        categorical[..., 2][phase_valid] = (
            np.rint(phase[phase_valid]).astype(np.uint8) + 1
        )
        output["cloud_categorical_token.npy"][start:stop] = categorical
        output["cloud_categorical_valid_mask.npy"][start:stop, ..., 0] = acm_valid
        output["cloud_categorical_valid_mask.npy"][start:stop, ..., 1] = bcm_valid
        output["cloud_categorical_valid_mask.npy"][start:stop, ..., 2] = (
            phase_valid
        )

        quality_token = np.zeros(cod.shape, dtype=np.uint8)
        quality_token[cod_dqf_valid] = (
            cod_status[cod_dqf_valid] // np.uint8(2) + np.uint8(1)
        )
        output["cod_quality_token.npy"][start:stop] = quality_token
        output["cod_dqf_valid_mask.npy"][start:stop] = cod_dqf_valid
        output["cod_is_night.npy"][start:stop] = (
            cod_dqf_valid & ((cod_dqf & np.uint8(1)) != 0)
        )
        output["cod_strict_quality_mask.npy"][start:stop] = cod_strict
        output["cod_usable_quality_mask.npy"][start:stop] = cod_usable

        bits = np.asarray((1, 2, 4, 8, 16, 32), dtype=np.uint8)
        phase_quality = (
            (phase_dqf[..., None] & bits[None, None, None, :]) != 0
        )
        phase_quality &= phase_dqf_valid[..., None]
        output["phase_quality_bits.npy"][start:stop] = phase_quality
        output["phase_dqf_valid_mask.npy"][start:stop] = phase_dqf_valid
        output["phase_strict_quality_mask.npy"][start:stop] = (
            phase_valid & phase_dqf_valid & (phase_dqf == 0)
        )

        matched = np.stack(
            [
                np.asarray(z["codc_scan_matched"], dtype=np.uint8),
                np.asarray(z["acmc_scan_matched"], dtype=np.uint8),
                np.asarray(z["actpc_scan_matched"], dtype=np.uint8),
            ],
            axis=1,
        )
        offsets = np.stack(
            [
                np.asarray(z["codc_time_offset_seconds"], dtype=np.int32),
                np.asarray(z["acmc_time_offset_seconds"], dtype=np.int32),
                np.asarray(z["actpc_time_offset_seconds"], dtype=np.int32),
            ],
            axis=1,
        )
        output["product_matched_mask.npy"][start:stop] = matched
        output["scan_time_offset_seconds.npy"][start:stop] = offsets

        return {
            "cod_usable": int(cod_usable.sum()),
            "cod_strict": int(cod_strict.sum()),
            "cloud_probability_valid": int(probability_valid.sum()),
            "acm_valid": int(acm_valid.sum()),
            "bcm_valid": int(bcm_valid.sum()),
            "phase_valid": int(phase_valid.sum()),
            "phase_strict": int(
                (phase_valid & phase_dqf_valid & (phase_dqf == 0)).sum()
            ),
            "codc_matched": int(matched[:, 0].sum()),
            "acmc_matched": int(matched[:, 1].sum()),
            "actpc_matched": int(matched[:, 2].sum()),
        }


def build(args: argparse.Namespace) -> None:
    timeline = annual_timeline_ns(args.year)
    expected_days = len(timeline) // N_SLOTS_PER_DAY
    if expected_days not in (365, 366):
        raise AssertionError("invalid annual timeline")

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output)

    temporary = args.output.with_name(args.output.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    output = open_output_arrays(temporary, len(timeline))
    np.save(temporary / "time_utc_ns.npy", timeline)
    totals: dict[str, int] = {}

    try:
        for doy in range(1, expected_days + 1):
            source = (
                args.input
                / f"goes_l2_model_ready_{args.year}_doy{doy:03d}.npz"
            )
            if not source.exists():
                raise FileNotFoundError(source)

            start = (doy - 1) * N_SLOTS_PER_DAY
            stop = start + N_SLOTS_PER_DAY
            counts = process_day(source, output, timeline, start, stop)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value

            if doy == 1 or doy % 25 == 0 or doy == expected_days:
                print(f"DOY {doy:03d}/{expected_days}: complete", flush=True)

        for array in output.values():
            array.flush()
        output.clear()

        manifest = {
            "complete": True,
            "year": args.year,
            "timeline_rows": len(timeline),
            "station_names": list(STATIONS),
            "patch_shape": [PATCH_SIDE, PATCH_SIDE],
            "pixels_per_patch": N_PIXELS,
            "cloud_continuous_channels": [
                "cod_log1p",
                "cloud_probability",
            ],
            "cloud_categorical_channels": [
                "acm_token",
                "bcm_token",
                "phase_token",
            ],
            "categorical_missing_token": 0,
            "categorical_token_maps": {
                "acm_token": {
                    "1": "clear",
                    "2": "probably_clear",
                    "3": "probably_cloudy",
                    "4": "cloudy",
                },
                "bcm_token": {
                    "1": "clear_or_probably_clear",
                    "2": "cloudy_or_probably_cloudy",
                },
                "phase_token": {
                    "1": "clear_sky",
                    "2": "liquid_water",
                    "3": "super_cooled_liquid_water",
                    "4": "mixed_phase",
                    "5": "ice",
                    "6": "unknown",
                },
            },
            "cod_quality_token_rule": "0=missing; otherwise (DQF & 30)/2 + 1",
            "cod_usable_status_values": COD_USABLE_STATUS.tolist(),
            "cod_strict_status_value": 0,
            "phase_quality_bit_order": [
                "overall_degraded",
                "degraded_l1b",
                "degraded_beta_ratio",
                "weak_ice_signal",
                "degraded_surface_emissivity",
                "satellite_zenith_above_threshold",
            ],
            "product_order": ["CODC", "ACMC", "ACTPC"],
            "station_measurements_included": False,
            "nsrdb_included": False,
            "idw_included": False,
            "normalization_applied": {
                "cod": "log1p only; fit standardization on training split later",
                "cloud_probability": "none",
            },
            "counts": totals,
            "source": str(args.input),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        temporary.replace(args.output)
    except Exception:
        output.clear()
        raise

    print(f"SAVED {args.output}")
    print(json.dumps(totals, indent=2))


def parse_args() -> argparse.Namespace:
    data_root = Path(cfg.BASE_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            data_root / "goes_l2_model_ready" / "washington_2024"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            data_root
            / "spatiotemporal_dataset"
            / "washington_2024"
            / "goes_token_store_2024"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
