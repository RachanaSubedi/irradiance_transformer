"""Shared bidirectional temporal encoder for station-node sequences."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalEncoderConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    max_seq_len: int = 288


def sinusoidal_position_encoding(
    max_seq_len: int, d_model: int
) -> torch.Tensor:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if d_model <= 0 or d_model % 2:
        raise ValueError("d_model must be a positive even number")

    position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / d_model)
    )
    encoding = torch.zeros(max_seq_len, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * divisor)
    encoding[:, 1::2] = torch.cos(position * divisor)
    return encoding


class SharedTemporalEncoder(nn.Module):
    """Apply one shared, bidirectional Transformer along time for every node.

    Input/output shape is [B,T,L,D]. Internally station nodes are folded into
    the batch dimension, producing [B*L,T,D]. Consequently there is no spatial
    information exchange here; all stations share exactly the same temporal
    parameters. Spatial and cloud interaction occurs in later modules.
    """

    def __init__(
        self,
        config: TemporalEncoderConfig = TemporalEncoderConfig(),
    ) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.config = config
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(
                config.max_seq_len, config.d_model
            ),
            persistent=True,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.input_dropout = nn.Dropout(config.dropout)

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if node_embeddings.ndim != 4:
            raise ValueError("node_embeddings must have shape [B,T,L,D]")
        batch, time, stations, features = node_embeddings.shape
        if features != self.config.d_model:
            raise ValueError(
                f"embedding dimension {features}; "
                f"expected {self.config.d_model}"
            )
        if time > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {time} exceeds "
                f"max_seq_len={self.config.max_seq_len}"
            )

        # [B,T,L,D] -> [B,L,T,D] -> [B*L,T,D].
        sequence = (
            node_embeddings.permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch * stations, time, features)
        )
        sequence = sequence + self.position_encoding[:time].to(
            device=sequence.device,
            dtype=sequence.dtype,
        ).unsqueeze(0)
        encoded = self.encoder(self.input_dropout(sequence))

        # Restore canonical station axis.
        return (
            encoded.reshape(batch, stations, time, features)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
