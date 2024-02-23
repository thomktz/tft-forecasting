"""Variable selection component for the model."""

import torch
import torch.nn as nn
from typing import Optional
from tft_forecasting.model.components.grn import GatedResidualNetwork


class VariableSelection(nn.Module):
    """Variable selection component to learn the importance of each feature."""

    def __init__(self, mX: int, input_size: int, hidden_size: int, dropout_rate: float = 0.1) -> None:
        """
        Initialize the variable selection component.

        Parameters
        ----------
        mX: int
            Number of features (interated as 'j' in the paper)
        input_size: int
            Number of inputs (iterated as 'i' in the paper)
        hidden_size: int
            Size of the hidden layer ('d_model' in the paper)
        dropout_rate: float
            Dropout rate
        """
        super(VariableSelection, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.weights_grn = GatedResidualNetwork(input_size * mX, hidden_size, dropout_rate)
        self.grns = nn.ModuleList([GatedResidualNetwork(input_size, hidden_size, dropout_rate) for _ in range(mX)])

    def forward(self, input_matrix: torch.Tensor, c: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the variable selection component.

        Parameters
        ----------
        input_matrix: torch.Tensor[batch_size, mX, input_size]
            Input matrix
        """
        flattened = input_matrix.view(input_matrix.size(0), -1)
        weights = self.softmax(self.weights_grn(flattened, c=c))

        # Apply each GRN to the corresponding slice of input_matrix and multiply by weights
        weighted_values = [weights[:, i : (i + 1)] * grn(input_matrix[:, i]) for i, grn in enumerate(self.grns)]

        return torch.stack(weighted_values, dim=1).sum(dim=1)
