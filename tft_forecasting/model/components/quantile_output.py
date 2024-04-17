"""Quantile Forecast component for the model."""

import torch
import torch.nn as nn
from typing import List, Callable, Optional
from torch.optim.optimizer import Optimizer
from utils import default_quantiles


class MultiOutputQuantileRegression(nn.Module):
    """Multi Output Quantile Regression model implementation."""

    def __init__(self, input_size: int, quantiles: Optional[List[float]] = None):
        super(MultiOutputQuantileRegression, self).__init__()

        quantiles = quantiles if quantiles is not None else default_quantiles

        self.input_size = input_size
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.fc = nn.Linear(input_size, self.num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the model."""
        return self.fc(x)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, optimizer: Optimizer, epochs: int = 100) -> None:
        """Fit the model to the data."""
        criterion = self._quantile_loss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the output for the given input."""
        return self(X)

    def _quantile_loss(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Quantile loss function."""

        def loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            total_loss = 0
            for i in range(self.num_quantiles):
                error = outputs[:, i] - targets[:, i]
                total_loss += torch.mean(torch.max((self.quantiles[i] - 1) * error, self.quantiles[i] * error))
            return total_loss / self.num_quantiles

        return loss
