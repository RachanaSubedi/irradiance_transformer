"""Missingness-aware, target-relative station-node embedding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class StationEmbeddingConfig:
    d_model: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1
    distance_scale_km: float = 25.0
    elevation_scale_m: float = 1000.0
    elapsed_cap_minutes: float = 1440.0


class MissingnessAwareStationEmbedding(nn.Module):
    """Create one transferable station token per node and timestamp.

    No station-ID lookup table is used. Each token is the sum of four terms:

      content: CSI, observation mask, elapsed time, solar and meteorology
      time:    cyclic hour/day encodings
      geometry: target-relative east/north/distance/elevation difference
      role:    target indicator and allowed-source indicator

    The target-relative representation allows the same weights to be used for
    unseen locations. Absolute Washington station names are never embedded.
    """

    def __init__(
        self,
        meteorology_mean: torch.Tensor,
        meteorology_std: torch.Tensor,
        config: StationEmbeddingConfig = StationEmbeddingConfig(),
    ) -> None:
        super().__init__()
        if tuple(meteorology_mean.shape) != (3,):
            raise ValueError("meteorology_mean must contain three channels")
        if tuple(meteorology_std.shape) != (3,):
            raise ValueError("meteorology_std must contain three channels")
        if torch.any(meteorology_std <= 0):
            raise ValueError("meteorology_std must be positive")
        if config.distance_scale_km <= 0 or config.elevation_scale_m <= 0:
            raise ValueError("geometry scales must be positive")
        if config.elapsed_cap_minutes <= 0:
            raise ValueError("elapsed cap must be positive")

        self.config = config
        self.register_buffer(
            "meteorology_mean", meteorology_mean.detach().float().clone()
        )
        self.register_buffer(
            "meteorology_std", meteorology_std.detach().float().clone()
        )

        # 1 CSI + 1 observed mask + 1 elapsed + 1 clear-sky GHI
        # + 1 cos(zenith) + 3 meteorology + 3 meteorology masks = 11.
        self.content_dim = 11
        self.content_encoder = nn.Sequential(
            nn.Linear(self.content_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.d_model),
        )
        self.time_encoder = nn.Linear(4, config.d_model)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(4, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.d_model),
        )
        self.role_encoder = nn.Linear(2, config.d_model)
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_dropout = nn.Dropout(config.dropout)

    @classmethod
    def from_normalization_json(
        cls,
        path: str | Path,
        config: StationEmbeddingConfig = StationEmbeddingConfig(),
    ) -> "MissingnessAwareStationEmbedding":
        report = json.loads(Path(path).read_text())
        stats = report["meteorology"]
        names = ("temperature_c", "relative_humidity", "pressure_hpa")
        mean = torch.tensor([stats[name]["mean"] for name in names])
        std = torch.tensor([stats[name]["std"] for name in names])
        return cls(mean, std, config)

    @staticmethod
    def _target_relative_geometry(
        pairwise_distance_km: torch.Tensor,
        pairwise_east_km: torch.Tensor,
        pairwise_north_km: torch.Tensor,
        station_coordinates: torch.Tensor,
        target_station_index: torch.Tensor,
        distance_scale_km: float,
        elevation_scale_m: float,
    ) -> torch.Tensor:
        if pairwise_distance_km.ndim != 3:
            raise ValueError("pairwise geometry must have shape [B,L,L]")
        if pairwise_east_km.shape != pairwise_distance_km.shape:
            raise ValueError("pairwise east shape mismatch")
        if pairwise_north_km.shape != pairwise_distance_km.shape:
            raise ValueError("pairwise north shape mismatch")
        batch, stations, other = pairwise_distance_km.shape
        if stations != other:
            raise ValueError("pairwise geometry must be square")
        if station_coordinates.shape != (batch, stations, 3):
            raise ValueError("station_coordinates must have shape [B,L,3]")
        if target_station_index.shape != (batch,):
            raise ValueError("target_station_index must have shape [B]")

        row = torch.arange(batch, device=pairwise_distance_km.device)
        distance = pairwise_distance_km[row, target_station_index]
        east = pairwise_east_km[row, target_station_index]
        north = pairwise_north_km[row, target_station_index]

        elevation = station_coordinates[..., 2]
        target_elevation = elevation[row, target_station_index].unsqueeze(1)
        elevation_difference = elevation - target_elevation

        return torch.stack(
            (
                east / distance_scale_km,
                north / distance_scale_km,
                distance / distance_scale_km,
                elevation_difference / elevation_scale_m,
            ),
            dim=-1,
        )

    def forward(
        self,
        *,
        station_csi: torch.Tensor,
        station_csi_mask: torch.Tensor,
        time_since_observed_minutes: torch.Tensor,
        clear_sky_ghi: torch.Tensor,
        cos_zenith: torch.Tensor,
        meteorology: torch.Tensor,
        meteorology_mask: torch.Tensor,
        time_encoding: torch.Tensor,
        target_station_index: torch.Tensor,
        source_station_mask: torch.Tensor,
        station_coordinates: torch.Tensor,
        pairwise_distance_km: torch.Tensor,
        pairwise_east_km: torch.Tensor,
        pairwise_north_km: torch.Tensor,
    ) -> torch.Tensor:
        """Return node embeddings with shape [B,T,L,D]."""
        if station_csi.ndim != 3:
            raise ValueError("station_csi must have shape [B,T,L]")
        batch, time, stations = station_csi.shape
        expected_node_shape = (batch, time, stations)

        for name, tensor in (
            ("station_csi_mask", station_csi_mask),
            ("time_since_observed_minutes", time_since_observed_minutes),
            ("clear_sky_ghi", clear_sky_ghi),
            ("cos_zenith", cos_zenith),
        ):
            if tensor.shape != expected_node_shape:
                raise ValueError(f"{name} shape mismatch")

        if meteorology.shape != (batch, time, stations, 3):
            raise ValueError("meteorology must have shape [B,T,L,3]")
        if meteorology_mask.shape != meteorology.shape:
            raise ValueError("meteorology mask shape mismatch")
        if time_encoding.shape != (batch, time, 4):
            raise ValueError("time_encoding must have shape [B,T,4]")
        if source_station_mask.shape != (batch, stations):
            raise ValueError("source_station_mask must have shape [B,L]")

        dtype = station_csi.dtype
        observed = station_csi_mask.to(dtype)
        met_mask = meteorology_mask.to(dtype)

        normalized_met = (
            meteorology - self.meteorology_mean
        ) / self.meteorology_std
        normalized_met = torch.where(
            meteorology_mask,
            normalized_met,
            torch.zeros_like(normalized_met),
        )

        elapsed = torch.log1p(
            time_since_observed_minutes.clamp(
                min=0.0, max=self.config.elapsed_cap_minutes
            )
        ) / torch.log1p(
            torch.tensor(
                self.config.elapsed_cap_minutes,
                device=station_csi.device,
                dtype=dtype,
            )
        )

        content = torch.cat(
            (
                station_csi.unsqueeze(-1),
                observed.unsqueeze(-1),
                elapsed.unsqueeze(-1),
                (clear_sky_ghi / 1000.0).unsqueeze(-1),
                cos_zenith.unsqueeze(-1),
                normalized_met,
                met_mask,
            ),
            dim=-1,
        )
        if content.shape[-1] != self.content_dim:
            raise AssertionError("station content dimension changed unexpectedly")
        content_embedding = self.content_encoder(content)

        time_embedding = self.time_encoder(time_encoding)
        time_embedding = time_embedding[:, :, None, :]

        geometry = self._target_relative_geometry(
            pairwise_distance_km,
            pairwise_east_km,
            pairwise_north_km,
            station_coordinates,
            target_station_index,
            self.config.distance_scale_km,
            self.config.elevation_scale_m,
        )
        geometry_embedding = self.geometry_encoder(geometry)[:, None, :, :]

        station_indices = torch.arange(stations, device=station_csi.device)
        is_target = station_indices[None, :] == target_station_index[:, None]
        role = torch.stack(
            (
                is_target.to(dtype),
                source_station_mask.to(dtype),
            ),
            dim=-1,
        )
        role_embedding = self.role_encoder(role)[:, None, :, :]

        combined = (
            content_embedding
            + time_embedding
            + geometry_embedding
            + role_embedding
        )
        return self.output_dropout(self.output_norm(combined))
