import torch
import torch.nn as nn
from .glu import GatedLinearUnit


class AddAndNorm(nn.Module):
    """Add and Norm layer implementation."""

    def __init__(self, input_size: int) -> None:
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Add and Norm layer."""
        return self.layer_norm(x + y)


class GateAddNorm(nn.Module):
    """Gate, Add and Norm layer implementation."""

    def __init__(self, input_size: int) -> None:
        super(GateAddNorm, self).__init__()
        self.add_and_norm = AddAndNorm(input_size)
        self.gate = GatedLinearUnit(input_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Gate. Apply gate to x."""
        return self.add_and_norm(self.gate(x), y)
