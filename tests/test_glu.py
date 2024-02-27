import pytest
import torch
from tft_forecasting.model.components import GatedLinearUnit


@pytest.fixture
def input_tensor():
    """Fixture to generate a sample input tensor for GLU testing."""
    batch_size = 5
    input_size = 10
    hidden_size = 20
    x = torch.randn(batch_size, input_size)
    return x, input_size, hidden_size


@pytest.fixture
def glu_model(input_tensor):
    """Fixture to create a GatedLinearUnit model instance."""
    _, input_size, _ = input_tensor
    return GatedLinearUnit(input_size)


@pytest.fixture
def glu_model_hidden(input_tensor):
    _, input_size, hidden_size = input_tensor
    return GatedLinearUnit(input_size, hidden_size)


def test_glu_forward(input_tensor, glu_model):
    gamma, _, _ = input_tensor
    output = glu_model(gamma)
    assert output.shape == gamma.shape, "Output shape should match input shape."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."


def test_glu_hidden_forward(input_tensor, glu_model_hidden):
    gamma, _, hidden_size = input_tensor
    output = glu_model_hidden(gamma)
    assert output.shape[1] == hidden_size, "Output shape should match hidden shape."
    assert torch.is_tensor(output), "Output should be a torch.Tensor."
