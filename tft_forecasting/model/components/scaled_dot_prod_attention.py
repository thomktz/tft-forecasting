"""Scaled Dot Product Attention component for the model."""

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot Product Attention component for the model."""

    def __init__(self, scale: bool = True):
        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        """Forward pass for the model."""
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        output = torch.bmm(attn, v)
        return output, attn
