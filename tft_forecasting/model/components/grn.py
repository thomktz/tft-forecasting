"""Gated Residual Network (GRN) implementation."""

import torch
import torch.nn as nn
from typing import Optional
from .gate_add_norm import GateAddNorm


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) implementation."""

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float = 0.1) -> None:
        super(GatedResidualNetwork, self).__init__()

        self.elu = nn.ELU()
        self.gate = GateAddNorm(input_size=hidden_size, hidden_size=input_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_size, hidden_size)  # Input size hidden_size, output size hidden_size
        self.linear2 = nn.Linear(input_size, hidden_size)  # Input size a, output size hidden_size
        self.linear3 = nn.Linear(input_size, hidden_size, bias=False)  # Input size c=a, output size hidden_size

    def forward(self, a: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the GRN."""
        if c is None:
            eta_2 = self.elu(self.linear2(a))
        else:
            eta_2 = self.elu(self.linear2(a) + self.linear3(c))

        # Page 7 - "During training, dropout is applied before the gating layer
        # and layer normalization – i.e. to η1 in Eq. (3)."
        eta_1 = self.dropout(self.linear1(eta_2))
        return self.gate(eta_1, a)
