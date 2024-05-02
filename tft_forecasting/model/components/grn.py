"""Gated Residual Network (GRN) implementation."""

import torch
import torch.nn as nn
from typing import Optional
from .gate_add_norm import GateAddNorm
from ..utils import TimeDistributed


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) implementation."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0.1,
        context_size: Optional[int] = None,
        batch_first: bool = False,
    ) -> None:
        super().__init__()

        self.elu = nn.ELU()
        # To add 'a' to eta_1, we need to make sure that the dimensions match
        self.gate = GateAddNorm(input_size=output_size, time_distributed=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = TimeDistributed(nn.Linear(hidden_size, output_size))
        self.linear2 = TimeDistributed(nn.Linear(input_size, hidden_size))

        if context_size is not None:
            self.context = TimeDistributed(nn.Linear(context_size, hidden_size), batch_first=batch_first)

        self.map_input = input_size != output_size
        if self.map_input:
            self.input_mapper = TimeDistributed(nn.Linear(input_size, output_size))

    def forward(self, a: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the GRN."""
        # Ensure that 'a' and 'eta_1' have the same dimensions
        residual = self.input_mapper(a) if self.map_input else a

        if c is None:
            eta_2 = self.elu(self.linear2(a))
        else:
            eta_2 = self.elu(self.linear2(a) + self.context(c))

        # Page 7 - "During training, dropout is applied before the gating layer
        # and layer normalization – i.e. to η1 in Eq. (3)."
        eta_1 = self.dropout(self.linear1(eta_2))
        return self.gate(eta_1, residual)
