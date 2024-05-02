"""Variable selection component for the model."""

import torch
import torch.nn as nn
from typing import Optional
from tft_forecasting.model.components.grn import GatedResidualNetwork


class VariableSelection(nn.Module):
    """Variable selection component to learn the importance of each feature."""

    def __init__(
        self, mX: int, input_size: int, hidden_size: int, dropout_rate: float = 0.1, context_size: Optional[int] = None
    ) -> None:
        """
        Initialize the variable selection component.

        Parameters
        ----------
        mX: int
            Number of inputs
        input_size: int
            Number of features
        hidden_size: int
            Size of the hidden layer
        dropout_rate: float
            Dropout rate
        """
        super().__init__()

        self.mX = mX
        self.input_size = input_size

        self.softmax = nn.Softmax(dim=1)
        self.weights_grn = GatedResidualNetwork(
            input_size * mX, hidden_size, mX, dropout_rate, context_size=context_size
        )
        self.grns = nn.ModuleList(
            [GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout_rate) for _ in range(mX)]
        )

    def forward(self, input_matrix: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the variable selection component.

        Parameters
        ----------
        input_matrix: torch.Tensor[batch_size, mX, input_size]
            Input matrix
        """
        weights = self.softmax(self.weights_grn(input_matrix, c=c)).unsqueeze(2)

        var_outputs = []
        for i in range(self.mX):
            var_outputs.append(self.grns[i](input_matrix[:, :, (i * self.input_size) : (i + 1) * self.input_size]))

        outputs = (torch.stack(var_outputs, axis=-1) * weights).sum(axis=-1)

        return outputs, weights
