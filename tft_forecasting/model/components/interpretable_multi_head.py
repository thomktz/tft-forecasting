"""Masked Interpreatable Multihead Attention component for the model."""

import torch
import torch.nn as nn
from typing import Tuple
from .scaled_dot_prod_attention import ScaledDotProductAttention


class InterpretableMultiHeadAttention(nn.Module):
    """Maked Interpretable Multi-Head Attention (MIMHA) implementation."""

    def __init__(self, n_head: int, d_model: int):
        """
        Initializes an instance of the InterpretableMultiHead class.

        Args:
            n_head (int): The number of attention heads.
            d_model (int): The dimensionality of the model.

        """
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head

        self.v_layer = nn.Linear(self.d_model, self.d_v)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_q) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.attention = ScaledDotProductAttention()
        self.w_h = nn.Linear(self.d_v, self.d_model, bias=False)

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the model.

        This method initializes the weights of the model using Xavier uniform initialization for non-bias parameters
        and zero initialization for bias parameters.
        """
        for name, p in self.named_parameters():
            if "bias" not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the InterpretableMultiHead module.

        Args:
            q (torch.Tensor): The query tensor of shape (batch_size, seq_length, d_model).
            k (torch.Tensor): The key tensor of shape (batch_size, seq_length, d_model).
            v (torch.Tensor): The value tensor of shape (batch_size, seq_length, d_model).
            mask (torch.Tensor, optional): The attention mask tensor of shape (batch_size, seq_length, seq_length).
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output tensor of shape (batch_size, d_model)
            and the attention tensor of shape (batch_size, n_head, seq_length, seq_length).
        """
        heads = []
        attns = []
        # (batch_size, seq_length, d_model)
        vs = self.v_layer(v)
        # (batch_size, d_model, seq_length)
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            heads.append(head)
            attns.append(attn)

        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2).permute(0, 2, 1, 3)

        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        outputs = self.w_h(outputs)

        return outputs, attn
