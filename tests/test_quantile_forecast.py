import torch
import torch.optim as optim
import pytest
from tft_forecasting.model.components.quantile_output import MultiOutputQuantileRegression

# Sample data for testing
X_train = torch.randn(100, 5)  # 100 samples, 5 features
Y_train = torch.randn(100, 3)  # 100 samples, 3 quantiles


@pytest.fixture
def model():
    return MultiOutputQuantileRegression(input_size=5)


@pytest.fixture
def optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01)


def test_model_training(model, optimizer):
    model.fit(X_train, Y_train, optimizer, epochs=10)
    assert True  # Add assertions to verify training


def test_model_prediction(model):
    X_test = torch.randn(10, 5)  # 10 samples for testing
    predictions = model.predict(X_test)
    assert predictions.shape == (10, 3)  # Ensure correct output shape
