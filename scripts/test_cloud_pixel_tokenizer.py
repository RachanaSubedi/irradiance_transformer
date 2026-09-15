#!/usr/bin/env python3
"""Smoke-test the GOES cloud-pixel tokenizer on real dataset windows."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from irradiance import config as cfg
from irradiance.cloud_tokenizer import CloudPixelTokenizer
from irradiance.spatiotemporal_dataset import (
    SpatiotemporalWindowDataset,
    WindowSpec,
)


def main() -> None:
    root = Path(cfg.BASE_PATH) / "spatiotemporal_dataset" / "washington_2024"
    dataset = SpatiotemporalWindowDataset(
        root / "station_model_ready_2024.npz",
        root / "goes_token_store_2024",
        window=WindowSpec(72, 35),
        target_stations=("S1", "S2", "S3"),
        include_idw=False,
        allow_p2_as_source=False,
        max_samples=8,
    )
    batch = next(
        iter(DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0))
    )
    tokenizer = CloudPixelTokenizer.from_normalization_json(
        root / "splits_v1" / "normalization_train_only.json"
    )
    tokenizer.train()

    tokens, available = tokenizer(
        continuous=batch["cloud_continuous"],
        continuous_mask=batch["cloud_continuous_mask"],
        categorical_token=batch["cloud_categorical_token"],
        categorical_mask=batch["cloud_categorical_mask"],
        cod_quality_token=batch["cod_quality_token"],
        cod_is_night=batch["cod_is_night"],
        phase_quality_bits=batch["phase_quality_bits"],
        pixel_relative_y=batch["pixel_relative_y"],
        pixel_relative_x=batch["pixel_relative_x"],
    )

    print("Input continuous :", tuple(batch["cloud_continuous"].shape))
    print("Input categorical:", tuple(batch["cloud_categorical_token"].shape))
    print("Cloud tokens     :", tuple(tokens.shape))
    print("Pixel available  :", tuple(available.shape))
    print("Tokenizer input dimension:", tokenizer.input_dim)
    print(
        "Trainable parameters:",
        sum(parameter.numel() for parameter in tokenizer.parameters()),
    )

    assert tokens.shape == (2, 72, 4, 25, 64)
    assert available.shape == (2, 72, 4, 25)
    assert torch.isfinite(tokens).all()
    assert available.dtype == torch.bool

    loss = tokens.square().mean()
    loss.backward()
    gradients_ok = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in tokenizer.parameters()
        if parameter.requires_grad
    )
    print("Finite forward pass:", bool(torch.isfinite(tokens).all()))
    print("Finite backward pass:", gradients_ok)
    print("IDW present:", "idw_csi" in batch)
    print("OVERALL RESULT:", "PASS" if gradients_ok else "FAIL")


if __name__ == "__main__":
    main()
