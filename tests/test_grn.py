import pytest
import torch
from tft_forecasting.model.components.grn import GatedResidualNetwork


@pytest.fixture
def grn_input():
    """Fixture to generate sample input tensors for GRN testing."""
    batch_size = 5
    input_size = 10
    hidden_size = 20
    dropout_rate = 0.1
    a = torch.randn(batch_size, input_size)
    c = torch.randn(batch_size, input_size)
    return a, c, input_size, hidden_size, dropout_rate


@pytest.fixture
def grn_model(grn_input):
    """Fixture to create a GatedResidualNetwork model instance."""
    _, _, input_size, hidden_size, dropout_rate = grn_input
    return GatedResidualNetwork(input_size, hidden_size, dropout_rate)


def test_grn_forward_with_c(grn_input, grn_model):
    a, c, _, _, _ = grn_input
    output = grn_model(a, c)
    assert (
        output.shape == a.shape
    ), "Output shape should match (batch_size, input_size) when context tensor is provided."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."


def test_grn_forward_without_c(grn_input, grn_model):
    a, _, _, _, _ = grn_input
    output = grn_model(a)
    assert (
        output.shape == a.shape
    ), "Output shape should match (batch_size, input_size) when no context tensor is provided."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."
