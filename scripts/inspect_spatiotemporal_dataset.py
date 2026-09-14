#!/usr/bin/env python3
"""Smoke-test the dynamic station + GOES window dataset."""

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from irradiance import config as cfg
from irradiance.spatiotemporal_dataset import (
    SpatiotemporalWindowDataset,
    WindowSpec,
)


def main() -> None:
    root = Path(cfg.BASE_PATH) / "spatiotemporal_dataset" / "washington_2024"
    dataset = SpatiotemporalWindowDataset(
        root / "station_model_ready_2024.npz",
        root / "goes_token_store_2024",
        window=WindowSpec(seq_len=72, center=35),
        target_stations=("S1", "S2", "S3"),
        whole_target_blackout=True,
        allow_p2_as_source=False,
        include_idw=False,
        max_samples=12,
    )

    print("Dataset smoke-test records:", len(dataset))
    if not dataset:
        raise RuntimeError("dataset contains no valid samples")

    sample = dataset[0]
    print("\nSingle-sample tensors:")
    for name, value in sample.items():
        print(
            f"{name:38s} shape={str(tuple(value.shape)):22s} "
            f"dtype={value.dtype}"
        )

    target = int(sample["target_station_index"])
    assert sample["station_csi"].shape == (72, 4)
    assert sample["cloud_continuous"].shape == (72, 4, 25, 2)
    assert sample["cloud_categorical_token"].shape == (72, 4, 25, 3)
    assert sample["phase_quality_bits"].shape == (72, 4, 25, 6)
    assert not sample["station_csi_mask"][:, target].any()
    assert np.isfinite(float(sample["target_csi_center"]))
    assert bool(sample["target_truth_mask"][35])
    assert "idw_csi" not in sample

    # For S1/S2/S3 pseudo-target training, P2 is not an allowed source.
    assert not bool(sample["source_station_mask"][3])
    assert not sample["station_csi_mask"][:, 3].any()

    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("\nFirst batch:")
    print("station_csi:", tuple(batch["station_csi"].shape))
    print("cloud_continuous:", tuple(batch["cloud_continuous"].shape))
    print(
        "cloud_categorical_token:",
        tuple(batch["cloud_categorical_token"].shape),
    )
    print("target_csi_center:", tuple(batch["target_csi_center"].shape))
    print("target_station_index:", batch["target_station_index"].tolist())

    print("\nIDW active input:", "idw_csi" in sample)
    print("P2 permitted as pretraining source:", bool(sample["source_station_mask"][3]))
    print("OVERALL RESULT: PASS")


if __name__ == "__main__":
    main()
