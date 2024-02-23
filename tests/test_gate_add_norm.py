import pytest
import torch
from tft_forecasting.model.components.gate_add_norm import AddAndNorm, GateAddNorm


@pytest.fixture
def input_tensors():
    """Fixture to generate input tensors for testing."""
    x = torch.randn(5, 10)  # Example tensor
    y = torch.randn(5, 10)  # Another tensor with the same shape
    h = torch.randn(5, 20)  # Hidden state tensor
    return x, y, h


@pytest.fixture
def add_and_norm_model():
    """Fixture to create an AddAndNorm model instance."""
    input_size = 10  # Match the second dimension of input tensors
    return AddAndNorm(input_size)


@pytest.fixture
def gate_add_norm_model():
    """Fixture to create a GateAddNorm model instance."""
    input_size = 10
    hidden_size = 20  # Example hidden size
    return GateAddNorm(input_size, hidden_size)


def test_add_and_norm_forward(input_tensors, add_and_norm_model):
    x, y, _ = input_tensors
    output = add_and_norm_model(x, y)
    assert output.shape == x.shape, "Output shape should match input shape."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."


def test_gate_add_norm_forward(input_tensors, gate_add_norm_model):
    x, _, h = input_tensors
    output = gate_add_norm_model(x, h)
    assert output.shape == h.shape, "Output shape should match hidden shape."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."
