"""Quantile Forecast component for the model."""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from typing import List, Callable


class MultiOutputQuantileRegression(nn.Module):
    """
    A PyTorch module for multi-output quantile regression.

    Args:
        input_size (int): The size of the input features.
        quantiles (List[float], optional): The quantile levels to predict. Defaults to [0.1, 0.5, 0.9].

    Attributes:
        input_size (int): The size of the input features.
        quantiles (List[float]): The quantile levels to predict.
        num_quantiles (int): The number of quantiles to predict.
        fc (nn.Linear): The fully connected layer for predicting quantiles.

    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Performs a forward pass through the network.

        fit(X: torch.Tensor, Y: torch.Tensor, optimizer: Optimizer, epochs: int = 100) -> None:
            Trains the model on the given input-output pairs.

        predict(X: torch.Tensor) -> torch.Tensor:
            Predicts the quantiles for the given input.

    """

    def __init__(self, input_size: int, quantiles: List[float] = [0.1, 0.5, 0.9]):
        """
        Initializes a MultiOutputQuantileRegression object.

        Args:
            input_size (int): The size of the input.
            quantiles (List[float], optional): A list of quantiles to predict. Defaults to [0.1, 0.5, 0.9].
        """
        super(MultiOutputQuantileRegression, self).__init__()
        self.input_size = input_size
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.fc = nn.Linear(input_size, self.num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the QuantileOutput module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.fc(x)

    def fit(self, X: torch.Tensor, Y: torch.Tensor, optimizer: Optimizer, epochs: int = 100) -> None:
        """
        Fits the model to the training data.

        Args:
            X (torch.Tensor): The input data.
            Y (torch.Tensor): The target data.
            optimizer (Optimizer): The optimizer used for training.
            epochs (int, optional): The number of training epochs. Defaults to 100.
        """
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
        """
        Predicts the output for the given input tensor.

        Args:
            X (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The predicted output tensor.
        """
        return self(X)

    def _quantile_loss(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns a loss function that calculates the quantile loss.

        The quantile loss is calculated as the mean of the maximum of two terms:
        - (quantile - 1) * error, if error is negative
        - quantile * error, if error is non-negative

        Args:
            outputs (torch.Tensor): The predicted outputs.
            targets (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The quantile loss.

        """

        def loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Calculates the loss between the predicted outputs and the target values.

            Args:
                outputs (torch.Tensor): The predicted outputs from the model.
                targets (torch.Tensor): The target values.

            Returns:
                torch.Tensor: The calculated loss.

            """
            total_loss = 0
            for i in range(self.num_quantiles):
                error = outputs[:, i] - targets[:, i]
                total_loss += torch.mean(torch.max((self.quantiles[i] - 1) * error, self.quantiles[i] * error))
            return total_loss / self.num_quantiles

        return loss
