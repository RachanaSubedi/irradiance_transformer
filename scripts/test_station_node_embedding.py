#!/usr/bin/env python3
"""Smoke-test the missingness-aware station-node embedding."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from irradiance import config as cfg
from irradiance.spatiotemporal_dataset import (
    SpatiotemporalWindowDataset,
    WindowSpec,
)
from irradiance.station_embedding import MissingnessAwareStationEmbedding


def main() -> None:
    root = Path(cfg.BASE_PATH) / "spatiotemporal_dataset" / "washington_2024"
    dataset = SpatiotemporalWindowDataset(
        root / "station_model_ready_2024.npz",
        root / "goes_token_store_2024",
        window=WindowSpec(72, 35),
        target_stations=("S1", "S2", "S3"),
        whole_target_blackout=True,
        allow_p2_as_source=False,
        include_idw=False,
        max_samples=8,
    )
    batch = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
    )
    model = MissingnessAwareStationEmbedding.from_normalization_json(
        root / "splits_v1" / "normalization_train_only.json"
    )
    model.train()

    embeddings = model(
        station_csi=batch["station_csi"],
        station_csi_mask=batch["station_csi_mask"],
        time_since_observed_minutes=batch["time_since_observed_minutes"],
        clear_sky_ghi=batch["clear_sky_ghi"],
        cos_zenith=batch["cos_zenith"],
        meteorology=batch["meteorology"],
        meteorology_mask=batch["meteorology_mask"],
        time_encoding=batch["time_encoding"],
        target_station_index=batch["target_station_index"],
        source_station_mask=batch["source_station_mask"],
        station_coordinates=batch["station_coordinates"],
        pairwise_distance_km=batch["pairwise_distance_km"],
        pairwise_east_km=batch["pairwise_east_km"],
        pairwise_north_km=batch["pairwise_north_km"],
    )

    print("Station CSI input :", tuple(batch["station_csi"].shape))
    print("Node embeddings   :", tuple(embeddings.shape))
    print(
        "Trainable parameters:",
        sum(parameter.numel() for parameter in model.parameters()),
    )
    print(
        "Target CSI visible:",
        bool(
            batch["station_csi_mask"][
                torch.arange(len(batch["target_station_index"]))[:, None],
                torch.arange(batch["station_csi"].shape[1])[None, :],
                batch["target_station_index"][:, None],
            ].any()
        ),
    )
    print("Fixed station-ID embedding present: False")
    print("IDW present:", "idw_csi" in batch)

    assert embeddings.shape == (2, 72, 4, 64)
    assert torch.isfinite(embeddings).all()

    loss = embeddings.square().mean()
    loss.backward()
    gradients_ok = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("Finite forward pass:", bool(torch.isfinite(embeddings).all()))
    print("Finite backward pass:", gradients_ok)
    print("OVERALL RESULT:", "PASS" if gradients_ok else "FAIL")


if __name__ == "__main__":
    main()
