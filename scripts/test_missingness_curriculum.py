#!/usr/bin/env python3
"""Test artificial missingness against real station/GOES windows."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from irradiance import config as cfg
from irradiance.missingness import BLOCK, POINT, WHOLE_TARGET, MissingnessCurriculum
from irradiance.spatiotemporal_dataset import SpatiotemporalWindowDataset, WindowSpec
from irradiance.station_embedding import MissingnessAwareStationEmbedding


def main() -> None:
    root = Path(cfg.BASE_PATH) / "spatiotemporal_dataset" / "washington_2024"
    dataset = SpatiotemporalWindowDataset(
        root / "station_model_ready_2024.npz",
        root / "goes_token_store_2024",
        window=WindowSpec(72, 35),
        target_stations=("S1", "S2", "S3"),
        whole_target_blackout=False,
        allow_p2_as_source=False,
        include_idw=False,
        max_samples=512,
    )
    # Keep the smoke-test batch small because the general dataset also returns
    # all 25 GOES tokens even though this test exercises only station inputs.
    batch = next(iter(DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)))
    augmented = MissingnessCurriculum(seed=42)(batch, center=35)

    modes = augmented["missingness_mode"]
    names = {POINT: "point", BLOCK: "block", WHOLE_TARGET: "whole_target"}
    print("Mode counts:")
    for code, name in names.items():
        print(f"  {name:14s}: {int((modes == code).sum())}")

    target = augmented["target_station_index"]
    row = torch.arange(len(target))
    center_hidden = ~augmented["station_csi_mask"][row, 35, target]
    loss_is_truth = (
        ~augmented["imputation_loss_mask"]
        | augmented["target_truth_mask"]
    ).all()

    assert bool(center_hidden.all())
    assert bool(augmented["imputation_loss_mask"][:, 35].all())
    assert bool(loss_is_truth)
    assert not bool(augmented["source_station_mask"][:, 3].any())
    assert "idw_csi" not in augmented

    embedding = MissingnessAwareStationEmbedding.from_normalization_json(
        root / "splits_v1" / "normalization_train_only.json"
    )
    nodes = embedding(
        station_csi=augmented["station_csi"],
        station_csi_mask=augmented["station_csi_mask"],
        time_since_observed_minutes=augmented["time_since_observed_minutes"],
        clear_sky_ghi=augmented["clear_sky_ghi"],
        cos_zenith=augmented["cos_zenith"],
        meteorology=augmented["meteorology"],
        meteorology_mask=augmented["meteorology_mask"],
        time_encoding=augmented["time_encoding"],
        target_station_index=target,
        source_station_mask=augmented["source_station_mask"],
        station_coordinates=augmented["station_coordinates"],
        pairwise_distance_km=augmented["pairwise_distance_km"],
        pairwise_east_km=augmented["pairwise_east_km"],
        pairwise_north_km=augmented["pairwise_north_km"],
    )
    assert nodes.shape == (16, 72, 4, 64)
    assert torch.isfinite(nodes).all()

    print("Center hidden for every sample:", bool(center_hidden.all()))
    print("Loss mask contains truth only:", bool(loss_is_truth))
    print("P2 excluded as source:", not bool(augmented["source_station_mask"][:, 3].any()))
    print("IDW active:", "idw_csi" in augmented)
    print("Node embedding shape:", tuple(nodes.shape))
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
