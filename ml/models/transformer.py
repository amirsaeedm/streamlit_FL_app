# ml/models/transformer.py
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (batch_first: [B, L, D])"""
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)          # [max_len, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, D]
        self.register_buffer('pe', pe)  # not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Encoder-decoder Transformer for multi-step forecasting:
      - Encoder attends over the past L steps.
      - Decoder uses learned forecast queries (one per horizon step)
        attending to the encoder memory to produce t+1..t+H.
    Shapes:
      input  : [B, L, input_size]
      output : [B, forecast_steps, output_size]
    """
    def __init__(
        self,
        input_size: int,           # D_in
        output_size: int,          # D_out (e.g., 5 targets)
        forecast_steps: int = 6,   # H
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.d_model = d_model

        # Project inputs into model dimension
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encodings
        self.enc_pos = PositionalEncoding(d_model, dropout)
        self.dec_pos = PositionalEncoding(d_model, dropout)

        # Transformer encoder/decoder (batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)

        # Learned queries: one vector per forecast step
        self.query_embed = nn.Parameter(torch.randn(1, forecast_steps, d_model))

        # Output projection per step
        self.output_proj = nn.Linear(d_model, output_size)

        # Optional weight init
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, input_size]
        returns: [B, forecast_steps, output_size]
        """
        B = x.size(0)

        # Encoder
        enc_in = self.input_proj(x)           # [B, L, d_model]
        enc_in = self.enc_pos(enc_in)         # add PE
        memory = self.encoder(enc_in)         # [B, L, d_model]

        # Decoder queries
        tgt = self.query_embed.expand(B, -1, -1)  # [B, H, d_model]
        tgt = self.dec_pos(tgt)

        # Decode with cross-attention to memory
        dec_out = self.decoder(tgt=tgt, memory=memory)  # [B, H, d_model]

        # Project each step to outputs
        y = self.output_proj(dec_out)          # [B, H, output_size]
        return y
