import pytest
import torch
from tft_forecasting.model.components.gate_add_norm import AddAndNorm, GateAddNorm


@pytest.fixture
def input_tensors():
    """Fixture to generate input tensors for testing."""
    x = torch.randn(5, 10)
    y = torch.randn(5, 10)
    return x, y


@pytest.fixture
def add_and_norm_model():
    """Fixture to create an AddAndNorm model instance."""
    input_size = 10
    return AddAndNorm(input_size)


@pytest.fixture
def gate_add_norm_model():
    """Fixture to create a GateAddNorm model instance."""
    input_size = 10
    return GateAddNorm(input_size)


def test_add_and_norm_forward(input_tensors, add_and_norm_model):
    x, y = input_tensors
    output = add_and_norm_model(x, y)
    assert output.shape == x.shape, "Output shape should match input shape."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."


def test_gate_add_norm_forward(input_tensors, gate_add_norm_model):
    x, y = input_tensors
    output = gate_add_norm_model(x, y)
    assert output.shape == y.shape, "Output shape should match input shape."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."
