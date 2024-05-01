import pytest
import torch
from tft_forecasting.model.components.variable_selection import VariableSelection


@pytest.fixture
def variable_selection_setup():
    mX = 4
    hidden_size = 20
    batch_size = 5
    dropout_rate = 0.1
    time_steps = 7

    # Create an input matrix with shape [batch_size, time_steps, mX]
    input_matrix = torch.randn(batch_size, time_steps, mX)
    # Initialize the VariableSelection model
    model = VariableSelection(mX, time_steps, hidden_size, dropout_rate)
    return input_matrix, model, batch_size, time_steps, hidden_size


def test_variable_selection_forward(variable_selection_setup):
    input_matrix, model, batch_size, time_steps, hidden_size = variable_selection_setup
    # Perform the forward pass
    output = model(input_matrix)
    # Verify the output shape
    print(input_matrix.shape)
    assert output.shape == (batch_size, hidden_size), "Output shape should match (batch_size, hidden_size)."
    # Ensure the output is a torch.Tensor
    assert torch.is_tensor(output), "Output should be a torch.Tensor."
