import torch
import pytest
from tft_forecasting.model.components.interpretable_multi_head import InterpretableMultiHeadAttention


@pytest.fixture
def attention_model():
    n_head = 2
    d_model = 8
    return InterpretableMultiHeadAttention(n_head, d_model)


def test_forward_method(attention_model):
    batch_size = 10
    seq_length = 5
    d_model = 8

    # Generate sample input tensors
    q = torch.randn(batch_size, seq_length, d_model)
    k = torch.randn(batch_size, seq_length, d_model)
    v = torch.randn(batch_size, seq_length, d_model)

    # Perform forward pass
    output, attn = attention_model(q, k, v)

    # Assertions
    assert output.shape == (batch_size, seq_length, d_model), "Output shape mismatch"
    assert attn.shape == (batch_size, attention_model.n_head, seq_length, seq_length), "Attention shape mismatch"


def test_masked_forward_method(attention_model):
    batch_size = 10
    seq_length = 5
    d_model = 8

    # Generate sample input tensors
    q = torch.randn(batch_size, seq_length, d_model)
    k = torch.randn(batch_size, seq_length, d_model)
    v = torch.randn(batch_size, seq_length, d_model)
    mask = torch.randint(2, size=(batch_size, seq_length, seq_length)).bool()

    # Perform forward pass with mask
    output, attn = attention_model(q, k, v, mask=mask)

    # Assertions
    assert output.shape == (batch_size, seq_length, d_model), "Output shape mismatch"
    assert attn.shape == (batch_size, attention_model.n_head, seq_length, seq_length), "Attention shape mismatch"
