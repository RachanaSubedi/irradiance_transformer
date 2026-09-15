#!/usr/bin/env python3
"""Test the shared temporal encoder for every pseudo-target station."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from irradiance import config as cfg
from irradiance.missingness import MissingnessCurriculum
from irradiance.spatiotemporal_dataset import (
    SpatiotemporalWindowDataset,
    WindowSpec,
)
from irradiance.station_embedding import MissingnessAwareStationEmbedding
from irradiance.temporal_encoder import SharedTemporalEncoder


def embed_station_batch(batch, embedding):
    return embedding(
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


def main() -> None:
    root = Path(cfg.BASE_PATH) / "spatiotemporal_dataset" / "washington_2024"
    station_embedding = MissingnessAwareStationEmbedding.from_normalization_json(
        root / "splits_v1" / "normalization_train_only.json"
    )
    temporal_encoder = SharedTemporalEncoder()
    masker = MissingnessCurriculum(seed=42)

    outputs = []
    for target_name in ("S1", "S2", "S3"):
        dataset = SpatiotemporalWindowDataset(
            root / "station_model_ready_2024.npz",
            root / "goes_token_store_2024",
            window=WindowSpec(72, 35),
            target_stations=(target_name,),
            whole_target_blackout=False,
            allow_p2_as_source=False,
            include_idw=False,
            max_samples=4,
        )
        batch = next(
            iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
        )
        augmented = masker(batch, center=35)
        nodes = embed_station_batch(augmented, station_embedding)
        temporal = temporal_encoder(nodes)

        expected_index = {"S1": 0, "S2": 1, "S3": 2}[target_name]
        assert bool(
            (augmented["target_station_index"] == expected_index).all()
        )
        assert nodes.shape == (2, 72, 4, 64)
        assert temporal.shape == nodes.shape
        assert torch.isfinite(temporal).all()
        outputs.append(temporal)

        print(
            f"{target_name}: target_index={expected_index}, "
            f"nodes={tuple(nodes.shape)}, temporal={tuple(temporal.shape)}"
        )

    # Verify that the temporal module itself is equivariant to station order:
    # permuting nodes, applying the shared encoder, and undoing the permutation
    # must reproduce the same output when dropout is disabled.
    temporal_encoder.eval()
    probe = outputs[0].detach()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = temporal_encoder(probe)
        permuted = temporal_encoder(probe[:, :, permutation])
        restored = permuted[:, :, inverse]
    equivariance_error = float((reference - restored).abs().max())
    assert equivariance_error < 1e-5

    station_embedding.train()
    temporal_encoder.train()
    total = sum(value.square().mean() for value in outputs)
    total.backward()
    parameters = list(station_embedding.parameters()) + list(
        temporal_encoder.parameters()
    )
    gradients_ok = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in parameters
        if parameter.requires_grad
    )

    print(
        "Temporal trainable parameters:",
        sum(p.numel() for p in temporal_encoder.parameters()),
    )
    print("Station-order equivariance max error:", equivariance_error)
    print("Finite backward pass:", gradients_ok)
    print("All S1/S2/S3 targets tested: True")
    print("IDW active: False")
    print("OVERALL RESULT:", "PASS" if gradients_ok else "FAIL")


if __name__ == "__main__":
    main()
