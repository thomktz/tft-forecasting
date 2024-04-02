import pytest
import torch
from tft_forecasting.model.components.grn import GatedResidualNetwork


@pytest.fixture
def grn_input():
    """Fixture to generate sample input tensors for GRN testing."""
    input_size = 10
    hidden_size = 20
    output_size = 15
    time_steps = 4
    batch_size = 7
    dropout_rate = 0.1
    return input_size, hidden_size, output_size, batch_size, time_steps, dropout_rate


@pytest.fixture
def grn_model(grn_input):
    """Fixture to create a GatedResidualNetwork model instance."""
    input_size, hidden_size, output_size, _, _, dropout_rate = grn_input
    return GatedResidualNetwork(input_size, hidden_size, output_size, dropout_rate)


def test_grn_forward_with_c(grn_input, grn_model):
    input_size, _, output_size, batch_size, time_steps, _ = grn_input
    a = torch.randn(batch_size, time_steps, input_size)
    c = torch.randn(batch_size, time_steps, input_size)
    output = grn_model(a, c)
    assert output.shape == (
        batch_size,
        time_steps,
        output_size,
    ), "Output shape should match (batch_size, time_steps, output_size) when context tensor is provided."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."


def test_grn_forward_without_c(grn_input, grn_model):
    input_size, _, output_size, batch_size, time_steps, _ = grn_input
    a = torch.randn(batch_size, time_steps, input_size)
    output = grn_model(a)
    assert output.shape == (
        batch_size,
        time_steps,
        output_size,
    ), "Output shape should match (batch_size, time_steps, output_size) when no context tensor is provided."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."
