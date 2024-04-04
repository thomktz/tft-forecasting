import torch.nn as nn
from torch.nn import LSTM
from .components.variable_selection import VariableSelection
from .components.gate_add_norm import GateAddNorm
from .components.grn import GatedResidualNetwork
from .components.interpretable_multi_head import InterpretableMultiHeadAttention
from .components.quantile_output import MultiOutputQuantileRegression
from typing import List


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer model implementation."""

    def __init__(
        self,
        mX_encoder: int,
        mX_decoder: int,
        input_size: int,
        n_past_inputs,
        n_known_future_inputs,
        hidden_size : int,
        quantiles: List[float] = [0.1, 0.5, 0.9],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder_var_selec = VariableSelection(
            mx=mX_encoder,
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.decoder_var_selec = VariableSelection(
            mx=mX_decoder,
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.lstm_encoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=self.lstm_layers,
            dropout=dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=self.lstm_layers,
            dropout=dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.post_lstm_gan = GateAddNorm(input_size=hidden_size, hidden_size=hidden_size)

        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout,
        )

        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            n_head=self.num_attention_heads,
        )

        self.feed_forward_block = GateAddNorm(input_size=hidden_size, hidden_size=hidden_size)

        self.output_layer = MultiOutputQuantileRegression(input_size=hidden_size, quantiles=quantiles)
