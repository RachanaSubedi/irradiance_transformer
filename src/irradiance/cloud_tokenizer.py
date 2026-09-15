"""Neural components for target-aware GOES patch representation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn


@dataclass(frozen=True)
class CloudTokenizerConfig:
    d_model: int = 64
    hidden_dim: int = 64
    dropout: float = 0.1
    acm_embedding_dim: int = 4
    bcm_embedding_dim: int = 2
    phase_embedding_dim: int = 4
    cod_quality_embedding_dim: int = 4


class CloudPixelTokenizer(nn.Module):
    """Encode each GOES pixel as one spatially indexed cloud token.

    Expected channel order follows goes_token_store_2024/manifest.json:
      continuous: [COD log1p, cloud probability]
      categorical: [ACM token, BCM token, phase token]

    Token 0 is reserved for a missing categorical value. The same network is
    shared over every time, station and pixel; station-relative coordinates
    preserve where the station lies within each nominally identical patch.
    """

    def __init__(
        self,
        continuous_mean: torch.Tensor,
        continuous_std: torch.Tensor,
        config: CloudTokenizerConfig = CloudTokenizerConfig(),
    ) -> None:
        super().__init__()
        if tuple(continuous_mean.shape) != (2,):
            raise ValueError("continuous_mean must contain two channels")
        if tuple(continuous_std.shape) != (2,):
            raise ValueError("continuous_std must contain two channels")
        if torch.any(continuous_std <= 0):
            raise ValueError("continuous_std must be positive")

        self.config = config
        self.register_buffer(
            "continuous_mean", continuous_mean.detach().float().clone()
        )
        self.register_buffer(
            "continuous_std", continuous_std.detach().float().clone()
        )

        self.acm_embedding = nn.Embedding(
            5, config.acm_embedding_dim, padding_idx=0
        )
        self.bcm_embedding = nn.Embedding(
            3, config.bcm_embedding_dim, padding_idx=0
        )
        self.phase_embedding = nn.Embedding(
            7, config.phase_embedding_dim, padding_idx=0
        )
        # COD quality: 0 missing; possible decoded tokens 1 through 9.
        self.cod_quality_embedding = nn.Embedding(
            10, config.cod_quality_embedding_dim, padding_idx=0
        )

        input_dim = (
            2  # standardized continuous values
            + 2  # continuous validity
            + config.acm_embedding_dim
            + config.bcm_embedding_dim
            + config.phase_embedding_dim
            + 3  # categorical validity
            + config.cod_quality_embedding_dim
            + 1  # COD day/night algorithm
            + 6  # decoded phase-quality bits
            + 2  # station-relative pixel y/x
        )
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )

    @classmethod
    def from_normalization_json(
        cls,
        path: str | Path,
        config: CloudTokenizerConfig = CloudTokenizerConfig(),
    ) -> "CloudPixelTokenizer":
        report = json.loads(Path(path).read_text())
        stats = report["cloud_continuous"]
        names = ("cod_log1p", "cloud_probability")
        mean = torch.tensor([stats[name]["mean"] for name in names])
        std = torch.tensor([stats[name]["std"] for name in names])
        return cls(mean, std, config)

    @staticmethod
    def _relative_coordinates(
        relative_y: torch.Tensor,
        relative_x: torch.Tensor,
        batch: int,
        stations: int,
        pixels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Dataset batches provide [B,L,5,5]; direct calls may provide [L,5,5].
        if relative_y.ndim == 3:
            relative_y = relative_y.unsqueeze(0).expand(batch, -1, -1, -1)
            relative_x = relative_x.unsqueeze(0).expand(batch, -1, -1, -1)
        if relative_y.ndim != 4 or relative_x.shape != relative_y.shape:
            raise ValueError("relative coordinates must be [L,5,5] or [B,L,5,5]")
        if relative_y.shape[:2] != (batch, stations):
            raise ValueError("relative-coordinate batch/station dimensions mismatch")
        if relative_y.shape[-2] * relative_y.shape[-1] != pixels:
            raise ValueError("relative-coordinate pixel count mismatch")

        # Fixed physical scaling; no validation information is used.
        y = relative_y.reshape(batch, stations, pixels).to(device, dtype) / 2.5
        x = relative_x.reshape(batch, stations, pixels).to(device, dtype) / 2.5
        return torch.stack((y, x), dim=-1)

    def forward(
        self,
        *,
        continuous: torch.Tensor,
        continuous_mask: torch.Tensor,
        categorical_token: torch.Tensor,
        categorical_mask: torch.Tensor,
        cod_quality_token: torch.Tensor,
        cod_is_night: torch.Tensor,
        phase_quality_bits: torch.Tensor,
        pixel_relative_y: torch.Tensor,
        pixel_relative_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tokens [B,T,L,25,D] and pixel availability [B,T,L,25]."""
        if continuous.ndim != 5 or continuous.shape[-1] != 2:
            raise ValueError("continuous must have shape [B,T,L,25,2]")
        if continuous_mask.shape != continuous.shape:
            raise ValueError("continuous mask shape mismatch")
        if categorical_token.shape != (*continuous.shape[:-1], 3):
            raise ValueError("categorical token shape mismatch")
        if categorical_mask.shape != categorical_token.shape:
            raise ValueError("categorical mask shape mismatch")
        if cod_quality_token.shape != continuous.shape[:-1]:
            raise ValueError("COD quality shape mismatch")
        if cod_is_night.shape != continuous.shape[:-1]:
            raise ValueError("COD day/night shape mismatch")
        if phase_quality_bits.shape != (*continuous.shape[:-1], 6):
            raise ValueError("phase-quality shape mismatch")

        batch, time, stations, pixels, _ = continuous.shape
        dtype = continuous.dtype
        device = continuous.device

        continuous_mask_f = continuous_mask.to(dtype)
        normalized = (
            (continuous - self.continuous_mean)
            / self.continuous_std
        )
        # Invalid channels remain exactly zero rather than receiving a
        # standardized representation of the storage fill value.
        normalized = torch.where(
            continuous_mask, normalized, torch.zeros_like(normalized)
        )

        acm = categorical_token[..., 0]
        bcm = categorical_token[..., 1]
        phase = categorical_token[..., 2]
        if torch.any((acm < 0) | (acm > 4)):
            raise ValueError("ACM token outside 0..4")
        if torch.any((bcm < 0) | (bcm > 2)):
            raise ValueError("BCM token outside 0..2")
        if torch.any((phase < 0) | (phase > 6)):
            raise ValueError("phase token outside 0..6")
        if torch.any((cod_quality_token < 0) | (cod_quality_token > 9)):
            raise ValueError("COD-quality token outside 0..9")

        relative = self._relative_coordinates(
            pixel_relative_y,
            pixel_relative_x,
            batch,
            stations,
            pixels,
            device,
            dtype,
        )
        relative = relative[:, None].expand(-1, time, -1, -1, -1)

        features = torch.cat(
            [
                normalized,
                continuous_mask_f,
                self.acm_embedding(acm),
                self.bcm_embedding(bcm),
                self.phase_embedding(phase),
                categorical_mask.to(dtype),
                self.cod_quality_embedding(cod_quality_token),
                cod_is_night.unsqueeze(-1).to(dtype),
                phase_quality_bits.to(dtype),
                relative,
            ],
            dim=-1,
        )
        if features.shape[-1] != self.input_dim:
            raise AssertionError("tokenizer input dimension changed unexpectedly")

        tokens = self.network(features)
        pixel_available = (
            continuous_mask.any(dim=-1)
            | categorical_mask.any(dim=-1)
            | (cod_quality_token != 0)
        )
        return tokens, pixel_available
