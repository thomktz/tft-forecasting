import math
import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """
    Positional encoder for the transformer model.

    A much faster implementation of
    https://github.com/mattsherar/Temporal_Fusion_Transform/blob/master/tft_model.py#L122
    """

    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        self.scale = math.sqrt(d_model)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).detach()
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Forward pass of the positional encoder."""
        x = x * self.scale
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
