import pytest
import torch
from tft_forecasting.model.components.variable_selection import VariableSelection


@pytest.fixture
def variable_selection_setup():
    mX = 4  # Number of features
    input_size = 10  # Number of inputs per feature
    hidden_size = 20
    batch_size = 5
    dropout_rate = 0.1

    # Create an input matrix with shape [batch_size, mX, input_size]
    input_matrix = torch.randn(batch_size, mX, input_size)
    # Initialize the VariableSelection model
    model = VariableSelection(mX, input_size, hidden_size, dropout_rate)
    return input_matrix, model, batch_size, input_size


def test_variable_selection_forward(variable_selection_setup):
    input_matrix, model, batch_size, input_size = variable_selection_setup
    # Perform the forward pass
    output = model(input_matrix)
    # Verify the output shape
    expected_output_shape = (batch_size, input_size)
    assert output.shape == expected_output_shape, "Output shape should match (batch_size, hidden_size)."
    # Ensure the output is a torch.Tensor
    assert torch.is_tensor(output), "Output should be a torch.Tensor."
