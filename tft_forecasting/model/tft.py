import torch.nn as nn
from torch.nn import LSTM
from .components.variable_selection import VariableSelection
from .components.gate_add_norm import GateAddNorm


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer model implementation."""

    def __init__(
        self,
        mX_encoder,
        mX_decoder,
        input_size,
        n_past_inputs,
        n_known_future_inputs,
        hidden_size,
    ) -> None:
        super().__init__()

        self.encoder_var_selec = VariableSelection(
            mx=mX_encoder,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        self.decoder_var_selec = VariableSelection(
            mx=mX_decoder,
            input_size=input_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        )

        self.lstm_encoder = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.lstm_decoder = LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0,
            batch_first=True,
        )

        self.post_lstm_gan = GateAddNorm(input_size=self.hidden_size, hidden_size=self.hidden_size)
