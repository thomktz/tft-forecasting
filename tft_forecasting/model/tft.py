import torch.nn as nn


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer model implementation."""

    def __init__(
        self,
        n_past_inputs,
        n_known_future_inputs,
    ) -> None:
        super().__init__()
        self.past
