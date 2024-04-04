"""Quantile Forecast component for the model."""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import List, Callable


class MultiOutputQuantileRegression(nn.Module):
    def __init__(self, input_size: int, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super(MultiOutputQuantileRegression, self).__init__()
        self.input_size = input_size
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.fc = nn.Linear(input_size, self.num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, optimizer: Optimizer, epochs: int = 100) -> None:
        criterion = self._quantile_loss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, Y)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        return self(X)

    def _quantile_loss(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        def loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            total_loss = 0
            for i in range(self.num_quantiles):
                error = outputs[:, i] - targets[:, i]
                total_loss += torch.mean(torch.max((self.quantiles[i] - 1) * error, self.quantiles[i] * error))
            return total_loss / self.num_quantiles
        return loss
