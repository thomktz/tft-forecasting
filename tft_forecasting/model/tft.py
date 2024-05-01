import torch
import torch.nn as nn

from .components.variable_selection import VariableSelection
from .components.grn import GatedResidualNetwork
from .components.glu import GatedLinearUnit
from .components.embedding import PositionalEncoder
from .utils import default_quantiles, TimeDistributed


class TFT(nn.Module):
    """
    Main Temporal Fusion Transformer model implementation.

    Parameters
    ----------
    device: torch.device
        Device to run the model on
    batch_size: int
        Batch size
    static_variables: int
        Number of static variables
    encode_length: int
        Length of the encoder
    categorical_covariates: int
        Number of time varying categorical variables
    n_past_covariates: int
        Number of time varying real variables in the encoder
    n_future_covariates: int
        Number of time varying real variables in the decoder
    num_masked_series: int
        Number of input series to mask
    hidden_size: int
        Hidden size
    lstm_layers: int
        Number of LSTM layers
    dropout: float
        Dropout rate
    embedding_dim: int
        Embedding dimension
    attn_heads: int
        Number of attention heads
    quantiles: list[float]
        List of quantiles
    seq_length: int
        Sequence length
    """

    def __init__(
        self,
        batch_size: int,
        encode_length: int,
        static_variables: int,
        categorical_covariates: int,
        n_past_covariates: int,
        n_future_covariates: int,
        num_masked_series: int,
        hidden_size: int = 32,
        lstm_layers: int = 2,
        dropout: float = 0.05,
        embedding_dim: int = 8,
        attn_heads: int = 8,
        quantiles: list[float] = default_quantiles,
        seq_length: int = 336,
        device: torch.device = "cpu",
    ) -> None:
        super(TFT, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.static_variables = static_variables
        self.encode_length = encode_length
        self.categorical_covariates = categorical_covariates
        self.n_past_covariates = n_past_covariates
        self.n_future_covariates = n_future_covariates
        self.num_input_series_to_mask = num_masked_series
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.attn_heads = attn_heads
        self.quantiles = quantiles
        self.seq_length = seq_length

        self.num_quantiles = len(quantiles)

        self.static_embedding_layers = nn.ModuleList(
            [nn.Embedding(embedding_dim, embedding_dim).to(self.device) for _ in range(self.static_variables)]
        )

        self.time_varying_embedding_layers = nn.ModuleList(
            [
                TimeDistributed(nn.Embedding(embedding_dim, embedding_dim), batch_first=True).to(self.device)
                for _ in range(self.categorical_covariates)
            ]
        )

        self.time_varying_linear_layers = nn.ModuleList(
            [
                TimeDistributed(nn.Linear(1, embedding_dim), batch_first=True).to(self.device)
                for _ in range(self.n_past_covariates)
            ]
        )

        self.encoder_variable_selection = VariableSelection(
            input_size=embedding_dim,
            mX=(n_past_covariates + categorical_covariates),
            hidden_size=hidden_size,
            dropout_rate=dropout,
            context_size=embedding_dim * static_variables,
        )

        self.decoder_variable_selection = VariableSelection(
            input_size=embedding_dim,
            mX=(n_future_covariates + categorical_covariates),
            hidden_size=hidden_size,
            dropout_rate=dropout,
            context_size=embedding_dim * static_variables,
        )

        self.lstm_encoder_input_size = embedding_dim * (n_past_covariates + categorical_covariates + static_variables)

        self.lstm_decoder_input_size = embedding_dim * (n_future_covariates + categorical_covariates + static_variables)

        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout,
        )

        self.post_lstm_gate = TimeDistributed(GatedLinearUnit(hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(hidden_size))

        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout,
            context_size=embedding_dim * static_variables,
        )

        self.position_encoding = PositionalEncoder(hidden_size, seq_length)

        self.multihead_attn = nn.MultiheadAttention(hidden_size, attn_heads)
        self.post_attn_gate = TimeDistributed(GatedLinearUnit(hidden_size))

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(hidden_size))
        self.pos_wise_ff = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(hidden_size))
        self.pre_output_gate = TimeDistributed(GatedLinearUnit(hidden_size))

        self.output_layer = TimeDistributed(nn.Linear(hidden_size, self.num_quantiles), batch_first=True)

    def init_hidden(self):
        """Zero initialization for the hidden state of the LSTM."""
        return torch.zeros(self.lstm_layers, self.batch_size, self.hidden_size, device=self.device)

    def apply_embedding(self, x, static_embedding, apply_masking):
        """Input dimension: (batch_size, timesteps, input_size)."""
        if apply_masking:
            time_varying_real_vectors = []
            for i in range(self.n_future_covariates):
                emb = self.time_varying_linear_layers[i + self.num_input_series_to_mask](
                    x[:, :, i + self.num_input_series_to_mask].view(x.size(0), -1, 1)
                )
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        else:
            time_varying_real_vectors = []
            for i in range(self.n_past_covariates):
                emb = self.time_varying_linear_layers[i](x[:, :, i].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        time_varying_categoical_vectors = []
        for i in range(self.time_varying_categoical_variables):
            emb = self.time_varying_embedding_layers[i](
                x[:, :, self.n_past_covariates + i].view(x.size(0), -1, 1).long()
            )
            time_varying_categoical_vectors.append(emb)
        time_varying_categoical_embedding = torch.cat(time_varying_categoical_vectors, dim=2)

        static_embedding = torch.cat(time_varying_categoical_embedding.size(1) * [static_embedding])
        static_embedding = static_embedding.view(
            time_varying_categoical_embedding.size(0), time_varying_categoical_embedding.size(1), -1
        )

        embeddings = torch.cat(
            [static_embedding, time_varying_categoical_embedding, time_varying_real_embedding], dim=2
        )

        return embeddings.view(-1, x.size(0), embeddings.size(2))

    def encode(self, x, hidden=None):
        """Encoder LSTM."""
        if hidden is None:
            hidden = self.init_hidden()

        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))

        return output, hidden

    def decode(self, x, hidden=None):
        """Decoder LSTM."""
        if hidden is None:
            hidden = self.init_hidden()

        output, (hidden, cell) = self.lstm_decoder(x, (hidden, hidden))

        return output, hidden

    def forward(self, x):
        """Input order: static, time_varying_categorical, time_varying_real."""
        embedding_vectors = []
        for i in range(self.static_variables):
            emb = self.static_embedding_layers[i](x["identifier"][:, 0, i].long().to(self.device))
            embedding_vectors.append(emb)

        static_embedding = torch.cat(embedding_vectors, dim=1)
        embeddings_encoder = self.apply_embedding(
            x["inputs"][:, : self.encode_length, :].float().to(self.device), static_embedding, apply_masking=False
        )
        embeddings_decoder = self.apply_embedding(
            x["inputs"][:, self.encode_length :, :].float().to(self.device), static_embedding, apply_masking=True
        )
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(
            embeddings_encoder[:, :, : -(self.embedding_dim * self.static_variables)],
            embeddings_encoder[:, :, -(self.embedding_dim * self.static_variables) :],
        )
        embeddings_decoder, decoder_sparse_weights = self.decoder_variable_selection(
            embeddings_decoder[:, :, : -(self.embedding_dim * self.static_variables)],
            embeddings_decoder[:, :, -(self.embedding_dim * self.static_variables) :],
        )

        pe = self.position_encoding(torch.zeros(self.seq_length, 1, embeddings_encoder.size(2)).to(self.device)).to(
            self.device
        )

        embeddings_encoder = embeddings_encoder + pe[: self.encode_length, :, :]
        embeddings_decoder = embeddings_decoder + pe[self.encode_length :, :, :]

        lstm_input = torch.cat([embeddings_encoder, embeddings_decoder], dim=0)
        encoder_output, hidden = self.encode(embeddings_encoder)
        decoder_output, _ = self.decode(embeddings_decoder, hidden)
        lstm_output = torch.cat([encoder_output, decoder_output], dim=0)
        lstm_output = self.post_lstm_gate(lstm_output + lstm_input)

        # Static enrichment
        static_embedding = torch.cat(lstm_output.size(0) * [static_embedding]).view(
            lstm_output.size(0), lstm_output.size(1), -1
        )
        attn_input = self.static_enrichment(lstm_output, static_embedding)
        attn_input = self.post_lstm_norm(lstm_output)

        attn_output, attn_output_weights = self.multihead_attn(
            attn_input[self.encode_length :, :, :],
            attn_input[: self.encode_length, :, :],
            attn_input[: self.encode_length, :, :],
        )
        attn_output = self.post_attn_gate(attn_output) + attn_input[self.encode_length :, :, :]
        attn_output = self.post_attn_norm(attn_output)

        output = self.pos_wise_ff(attn_output)  # [self.encode_length:,:,:])
        output = self.pre_output_gate(output) + lstm_output[self.encode_length :, :, :]

        output = self.pre_output_norm(output)
        output = self.output_layer(output.view(self.batch_size, -1, self.hidden_size))

        return (
            output,
            encoder_output,
            decoder_output,
            attn_output,
            attn_output_weights,
            encoder_sparse_weights,
            decoder_sparse_weights,
        )
