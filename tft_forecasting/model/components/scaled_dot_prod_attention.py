"""Scaled Dot Product Attention component for the model."""

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module.

    Args:
        scale (bool, optional): Whether to scale the attention scores by the square root of the dimension.
            Defaults to True.

    Inputs:
        q (torch.Tensor): The query tensor of shape (batch_size, seq_len_q, d_model).
        k (torch.Tensor): The key tensor of shape (batch_size, seq_len_k, d_model).
        v (torch.Tensor): The value tensor of shape (batch_size, seq_len_v, d_model).
        mask (torch.Tensor, optional): The mask tensor of shape (batch_size, seq_len_q, seq_len_k),
            where True values indicate positions to be masked. Defaults to None.

    Returns:
        output (torch.Tensor): The output tensor of shape (batch_size, seq_len_q, d_model).
        attn (torch.Tensor): The attention tensor of shape (batch_size, seq_len_q, seq_len_k).
    """

    def __init__(self, scale: bool = True):
        """
        Initializes the ScaledDotProdAttention module.

        Args:
            scale (bool, optional): Whether to scale the dot product attention scores by the square root
                of the feature dimension. Defaults to True.
        """

        super().__init__()
        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        """
        Performs the forward pass of the scaled dot product attention mechanism.

        Args:
            q (torch.Tensor): The query tensor of shape (batch_size, seq_len_q, d_model).
            k (torch.Tensor): The key tensor of shape (batch_size, seq_len_k, d_model).
            v (torch.Tensor): The value tensor of shape (batch_size, seq_len_v, d_model).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, seq_len_q, seq_len_k).
                Defaults to None.

        Returns:
            output (torch.Tensor): The output tensor of shape (batch_size, seq_len_q, d_model).
            attn (torch.Tensor): The attention weights tensor of shape (batch_size, seq_len_q, seq_len_k).
        """
        attn = torch.bmm(q, k.permute(0, 2, 1))

        if self.scale:
            dimension = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32))
            attn = attn / dimension

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)
        attn = self.softmax(attn)

        output = torch.bmm(attn, v)
        return output, attn
