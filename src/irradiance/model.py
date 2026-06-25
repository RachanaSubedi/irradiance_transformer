# ════════════════════════════════════════════════════════════
# utils/model.py
# Transformer architecture for CSI spatial imputation.
# Import this everywhere — never redefine the model in scripts.
# ════════════════════════════════════════════════════════════

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).
    Adds relative position-in-window awareness to embeddings.
    Note: explicit time features (hour_sin/cos, doy_sin/cos) are also
    present in input, so PE adds sequence position, not calendar time.
    """
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerImputer(nn.Module):
    """
    Transformer encoder for spatial GHI imputation.

    Architecture
    ────────────
    Input  : (batch, seq_len, input_dim)  — 48 timesteps × 14 features
    Output : (batch,)                     — predicted CSI at center step

    input_proj  : Linear(input_dim → d_model)
    pos_enc     : Sinusoidal positional encoding
    encoder     : N × TransformerEncoderLayer  (Pre-LN, batch_first=True)
    head        : MLP(d_model → 128 → 64 → 1) with ReLU + Dropout
    clamp       : output ∈ [0, 1.5] (physically valid CSI range)

    Version history
    ───────────────
    v1 : input_dim=10  (anchor CSI/mask, target NSRDB+C13, time)
    v2 : input_dim=14  (+ anchor1 NSRDB+C13, anchor2 NSRDB+C13)
    """

    def __init__(
        self,
        input_dim:  int = 14,
        d_model:    int = 64,
        nhead:      int = 4,
        num_layers: int = 2,
        d_ff:       int = 128,
        dropout:    float = 0.1,
        center:     int = 23,
    ):
        super().__init__()
        self.center     = center
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = d_ff,
            dropout         = dropout,
            batch_first     = True,   # (batch, seq, features)
            norm_first      = True,   # Pre-LN for stable training
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers           = num_layers,
            enable_nested_tensor = False,
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),      nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)           # (B, 48, d_model)
        x = self.pos_enc(x)
        x = self.encoder(x)              # (B, 48, d_model)
        out = self.head(x[:, self.center]).squeeze(-1)  # (B,)
        return out.clamp(0.0, 1.5)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_checkpoint(cls, path: str, device=None) -> "TransformerImputer":
        """
        Load a saved model from checkpoint.
        Usage:
            model = TransformerImputer.from_checkpoint("pretrain_best_model_v2.pt")
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        cfg  = ckpt["config"]
        model = cls(
            input_dim  = cfg["input_dim"],
            d_model    = cfg["d_model"],
            nhead      = cfg["nhead"],
            num_layers = cfg["num_layers"],
            d_ff       = cfg["d_ff"],
            dropout    = cfg["dropout"],
            center     = cfg["center"],
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        print(f"  Loaded model from {path}")
        print(f"  Version: {ckpt.get('version','unknown')} | "
              f"Test RMSE: {ckpt.get('test_rmse', ckpt.get('ft_rmse','?')):.4f}")
        return model, ckpt