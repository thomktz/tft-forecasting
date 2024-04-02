import torch
import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) implementation."""

    def __init__(self, input_size: int) -> None:
        super(GatedLinearUnit, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self) -> None:
        """Initialize weights."""
        for n, p in self.named_parameters():
            if "bias" in n:
                torch.nn.init.zeros_(p)
            elif "fc" in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, gamma: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GLU."""
        return self.sigmoid(self.linear1(gamma)) * self.linear2(gamma)
