"""Gated Residual Network (GRN) implementation."""

import torch.nn as nn


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) implementation."""

    def __init__(self, input_size):
        super(GatedLinearUnit, self).__init__()

        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, gamma):
        """Forward pass of the GLU."""
        return self.sigmoid(self.linear1(gamma)) * self.linear2(gamma)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) implementation."""

    def __init__(self, input_size, dropout_rate=0.1):
        super(GatedResidualNetwork, self).__init__()

        self.elu = nn.ELU()
        self.glu = GatedLinearUnit(input_size)
        self.layer_norm = nn.LayerNorm(input_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.linear3 = nn.Linear(input_size, input_size)

    def forward(self, a, c):
        """Forward pass of the GRN."""
        eta_2 = self.elu(self.linear2(a) + self.linear3(c))

        # Page 7 - "During training, dropout is applied before the gating layer
        # and layer normalization – i.e. to η1 in Eq. (3)."
        eta_1 = self.dropout(self.linear1(eta_2))
        return self.layer_norm(a + self.glu(eta_1))
