#!/usr/bin/env python3
"""Build the quality-preserving station half of the new GOES-patch dataset.

This intentionally does NOT use build_master(): that legacy pipeline adds
NSRDB/C13/C02 features and fills station gaps. Here, raw observations stay raw.
The existing daily GOES model-ready NPZ files remain the cloud-patch store; this
script creates the small annual station/physics store joined to the same 5-min
UTC timeline.

IDW is saved as an optional baseline/ablation field. It is not an active model
input unless a later training configuration explicitly enables it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from irradiance import config as cfg


STATIONS = ("s1", "s2", "s3", "p2")
FIVE_MINUTES_NS = 300_000_000_000
GHI_COLUMN = "Solar Radiation (W/m^2)"
MET_COLUMNS = {
    "temperature_c": "Outdoor Temperature (°F)",
    "relative_humidity": "Humidity (%)",
    "pressure_hpa": "Relative Pressure (inHg)",
}


def annual_timeline_ns(year: int) -> np.ndarray:
    start = pd.Timestamp(f"{year}-01-01T00:00:00Z")
    stop = pd.Timestamp(f"{year + 1}-01-01T00:00:00Z")
    count = int((stop.value - start.value) // FIVE_MINUTES_NS)
    return start.value + np.arange(count, dtype=np.int64) * FIVE_MINUTES_NS


def load_raw_station(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"Date", GHI_COLUMN}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")

    out = pd.DataFrame()
    out["datetime"] = pd.to_datetime(raw["Date"], utc=True, errors="coerce")
    out["ghi"] = pd.to_numeric(raw[GHI_COLUMN], errors="coerce")

    for output_name, source_name in MET_COLUMNS.items():
        out[output_name] = (
            pd.to_numeric(raw[source_name], errors="coerce")
            if source_name in raw.columns
            else np.nan
        )

    # Unit conversions. Relative humidity is retained as a 0-1 fraction.
    out["temperature_c"] = (out["temperature_c"] - 32.0) * (5.0 / 9.0)
    out["relative_humidity"] = out["relative_humidity"] / 100.0
    out["pressure_hpa"] = out["pressure_hpa"] * 33.8638866667

    out = out.dropna(subset=["datetime"]).sort_values("datetime")
    duplicate_count = int(out["datetime"].duplicated(keep=False).sum())
    if duplicate_count:
        print(f"  {path.name}: resolving {duplicate_count} duplicate timestamp rows")
        out = out.drop_duplicates("datetime", keep="last")
    return out.reset_index(drop=True)


def align_station(raw: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    # Pandas may parse ISO CSV timestamps at microsecond resolution while the
    # synthetic timeline is nanosecond resolution. merge_asof requires exact
    # dtype agreement, so normalize precision explicitly without rounding.
    left = timeline.copy()
    right = raw.copy()
    left["datetime"] = pd.Series(
        pd.DatetimeIndex(left["datetime"]).as_unit("ns"),
        index=left.index,
    )
    right["datetime"] = pd.Series(
        pd.DatetimeIndex(right["datetime"]).as_unit("ns"),
        index=right.index,
    )

    # Logger timestamps should already lie on the 5-min grid. A 90-second
    # tolerance permits clock jitter without borrowing an adjacent reading.
    return pd.merge_asof(
        left,
        right,
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=90),
    )


def clear_sky_and_solar(
    timeline: pd.DatetimeIndex, lat: float, lon: float, altitude: float
) -> tuple[np.ndarray, np.ndarray]:
    site = pvlib.location.Location(lat, lon, tz="UTC", altitude=altitude)
    clear = site.get_clearsky(timeline, model="ineichen")["ghi"].to_numpy(float)
    position = site.get_solarposition(timeline)
    cos_zenith = np.cos(np.deg2rad(position["apparent_zenith"].to_numpy(float)))
    return clear, np.clip(cos_zenith, 0.0, 1.0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def pairwise_geometry(
    latitudes: np.ndarray, longitudes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(latitudes)
    distance = np.zeros((n, n), dtype=np.float32)
    east = np.zeros((n, n), dtype=np.float32)
    north = np.zeros((n, n), dtype=np.float32)

    for target in range(n):
        for source in range(n):
            distance[target, source] = haversine_km(
                latitudes[target], longitudes[target],
                latitudes[source], longitudes[source],
            )
            mean_lat = math.radians((latitudes[target] + latitudes[source]) / 2)
            east[target, source] = (
                (longitudes[source] - longitudes[target])
                * 111.320 * math.cos(mean_lat)
            )
            north[target, source] = (
                (latitudes[source] - latitudes[target]) * 110.574
            )
    return distance, east, north


def compute_idw(
    csi: np.ndarray,
    valid: np.ndarray,
    distance_km: np.ndarray,
    power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target-specific IDW using only other stations valid at that timestamp."""
    n_time, n_station = csi.shape
    values = np.full((n_time, n_station), np.nan, dtype=np.float32)
    available = np.zeros((n_time, n_station), dtype=np.uint8)
    source_count = np.zeros((n_time, n_station), dtype=np.uint8)

    for target in range(n_station):
        candidates = np.arange(n_station) != target
        weights = np.zeros(n_station, dtype=np.float64)
        weights[candidates] = np.power(
            np.maximum(distance_km[target, candidates], 1e-6), -power
        )

        active = valid & candidates[None, :]
        weighted_sum = np.where(active, csi * weights[None, :], 0.0).sum(axis=1)
        weight_sum = np.where(active, weights[None, :], 0.0).sum(axis=1)
        count = active.sum(axis=1)

        ok = weight_sum > 0
        values[ok, target] = (weighted_sum[ok] / weight_sum[ok]).astype(np.float32)
        available[ok, target] = 1
        source_count[:, target] = count.astype(np.uint8)

    return values, available, source_count


def time_since_valid_minutes(valid: np.ndarray) -> np.ndarray:
    result = np.zeros(valid.shape, dtype=np.float32)
    for station in range(valid.shape[1]):
        elapsed = 0.0
        for t in range(valid.shape[0]):
            if valid[t, station]:
                elapsed = 0.0
            else:
                elapsed += 5.0
            result[t, station] = elapsed
    return result


def relative_patch_coordinates(
    geometry_path: Path, station_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(geometry_path.read_text())
    station_geometry = report["station_geometry"]
    row_offsets = np.arange(-2, 3, dtype=np.float32)
    col_offsets = np.arange(-2, 3, dtype=np.float32)
    relative_y = np.empty((len(station_names), 5, 5), dtype=np.float32)
    relative_x = np.empty_like(relative_y)

    for index, station in enumerate(station_names):
        item = station_geometry[station]
        fractional_y = float(item["fy"]) - float(item["center_y_index"])
        fractional_x = float(item["fx"]) - float(item["center_x_index"])
        relative_y[index] = row_offsets[:, None] - fractional_y
        relative_x[index] = col_offsets[None, :] - fractional_x

    return relative_y, relative_x


def build(args: argparse.Namespace) -> None:
    timeline_ns = annual_timeline_ns(args.year)
    timeline_index = pd.to_datetime(timeline_ns, utc=True)
    timeline = pd.DataFrame({"datetime": timeline_index})
    n_time = len(timeline)
    n_station = len(STATIONS)

    ghi = np.full((n_time, n_station), np.nan, dtype=np.float32)
    ghi_observed = np.zeros((n_time, n_station), dtype=np.uint8)
    csi = np.full((n_time, n_station), np.nan, dtype=np.float32)
    csi_valid = np.zeros((n_time, n_station), dtype=np.uint8)
    clear_sky_ghi = np.zeros((n_time, n_station), dtype=np.float32)
    cos_zenith = np.zeros((n_time, n_station), dtype=np.float32)
    meteorology = np.full((n_time, n_station, 3), np.nan, dtype=np.float32)
    meteorology_valid = np.zeros((n_time, n_station, 3), dtype=np.uint8)

    latitudes = np.array([cfg.STATIONS[s]["lat"] for s in STATIONS], dtype=np.float32)
    longitudes = np.array([cfg.STATIONS[s]["lon"] for s in STATIONS], dtype=np.float32)
    elevations = np.array([cfg.STATIONS[s].get("alt", 0.0) for s in STATIONS], dtype=np.float32)

    for index, station in enumerate(STATIONS):
        path = Path(cfg.RAW[f"ghi_{station}"])
        print(f"Loading {station.upper()}: {path}")
        aligned = align_station(load_raw_station(path), timeline)

        observed = np.isfinite(aligned["ghi"].to_numpy(float))
        station_ghi = aligned["ghi"].to_numpy(float)
        # Negative irradiance is physically invalid and is not marked observed.
        observed &= station_ghi >= 0.0

        clear, cosine = clear_sky_and_solar(
            timeline_index,
            float(latitudes[index]),
            float(longitudes[index]),
            float(elevations[index]),
        )
        daytime = clear >= args.daylight_threshold
        station_csi = np.divide(
            station_ghi,
            clear,
            out=np.full(n_time, np.nan, dtype=float),
            where=observed & daytime,
        )
        valid_csi = observed & daytime & np.isfinite(station_csi)
        station_csi[valid_csi] = np.clip(
            station_csi[valid_csi], args.csi_min, args.csi_max
        )

        ghi[:, index] = station_ghi.astype(np.float32)
        ghi_observed[:, index] = observed.astype(np.uint8)
        csi[:, index] = station_csi.astype(np.float32)
        csi_valid[:, index] = valid_csi.astype(np.uint8)
        clear_sky_ghi[:, index] = clear.astype(np.float32)
        cos_zenith[:, index] = cosine.astype(np.float32)

        for feature, column in enumerate(MET_COLUMNS):
            values = aligned[column].to_numpy(float)
            meteorology[:, index, feature] = values.astype(np.float32)
            meteorology_valid[:, index, feature] = np.isfinite(values).astype(np.uint8)

        print(
            f"  GHI observed={observed.mean():.3%}; "
            f"daytime CSI valid={valid_csi.sum():,}"
        )

    distance, east, north = pairwise_geometry(latitudes, longitudes)
    idw_csi, idw_available, idw_count = compute_idw(
        csi, csi_valid.astype(bool), distance, args.idw_power
    )

    time_since = time_since_valid_minutes(csi_valid.astype(bool))
    hours = (
        timeline_index.hour.to_numpy()
        + timeline_index.minute.to_numpy() / 60.0
    )
    doy = timeline_index.dayofyear.to_numpy()
    time_encoding = np.stack(
        [
            np.sin(2 * np.pi * hours / 24.0),
            np.cos(2 * np.pi * hours / 24.0),
            np.sin(2 * np.pi * doy / 366.0),
            np.cos(2 * np.pi * doy / 366.0),
        ],
        axis=1,
    ).astype(np.float32)

    relative_y, relative_x = relative_patch_coordinates(
        args.geometry, STATIONS
    )

    # Model inputs should use these zero-filled values together with masks.
    station_csi_input = np.nan_to_num(csi, nan=0.0).astype(np.float32)
    meteorology_input = np.nan_to_num(meteorology, nan=0.0).astype(np.float32)
    idw_csi_input = np.nan_to_num(idw_csi, nan=0.0).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            time_utc_ns=timeline_ns,
            station_names=np.asarray([s.upper() for s in STATIONS]),
            station_csi=station_csi_input,
            station_csi_target=csi,
            station_csi_valid_mask=csi_valid,
            station_ghi=ghi,
            station_ghi_observed_mask=ghi_observed,
            time_since_csi_observed_minutes=time_since,
            clear_sky_ghi=clear_sky_ghi,
            cos_zenith=cos_zenith,
            meteorology=meteorology_input,
            meteorology_valid_mask=meteorology_valid,
            meteorology_names=np.asarray(tuple(MET_COLUMNS)),
            time_encoding=time_encoding,
            time_encoding_names=np.asarray(
                ("hour_sin", "hour_cos", "doy_sin", "doy_cos")
            ),
            station_latitude=latitudes,
            station_longitude=longitudes,
            station_elevation_m=elevations,
            pairwise_distance_km=distance,
            pairwise_east_km=east,
            pairwise_north_km=north,
            pixel_relative_y=relative_y,
            pixel_relative_x=relative_x,
            idw_csi=idw_csi_input,
            idw_available_mask=idw_available,
            idw_source_count=idw_count,
        )
    temp_path.replace(args.output)

    report = {
        "complete": True,
        "year": args.year,
        "timeline_rows": n_time,
        "station_names": [s.upper() for s in STATIONS],
        "station_order_is_axis_1": True,
        "station_csi_input_fill": 0.0,
        "station_gaps_filled": False,
        "nsrdb_included": False,
        "idw_saved_but_enabled": False,
        "idw_power": args.idw_power,
        "daylight_threshold_wm2": args.daylight_threshold,
        "csi_clip": [args.csi_min, args.csi_max],
        "meteorology_names": list(MET_COLUMNS),
        "time_encoding_names": ["hour_sin", "hour_cos", "doy_sin", "doy_cos"],
        "csi_valid_counts": {
            station.upper(): int(csi_valid[:, i].sum())
            for i, station in enumerate(STATIONS)
        },
        "idw_available_counts": {
            station.upper(): int(idw_available[:, i].sum())
            for i, station in enumerate(STATIONS)
        },
        "goes_store": str(args.goes_root),
        "geometry_source": str(args.geometry),
        "notes": [
            "Use station_csi with station_csi_valid_mask as model input.",
            "Use station_csi_target only where station_csi_valid_mask is one.",
            "Do not enable idw_csi in the first direct-CSI experiment.",
            "GOES patches remain in the daily model-ready files and are not duplicated here.",
        ],
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"SAVED {args.output}")
    print(f"SAVED {args.output.with_suffix('.json')}")


def parse_args() -> argparse.Namespace:
    root = Path(cfg.BASE_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--goes-root",
        type=Path,
        default=root / "goes_l2_model_ready" / "washington_2024",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=root / "goes_l2_patches" / "washington_2024" / "codc" / "geometry.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "spatiotemporal_dataset" / "washington_2024" / "station_model_ready_2024.npz",
    )
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--daylight-threshold", type=float, default=10.0)
    parser.add_argument("--csi-min", type=float, default=0.0)
    parser.add_argument("--csi-max", type=float, default=1.3)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
