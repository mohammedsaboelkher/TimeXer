import torch
import torch.nn as nn, torch.nn.functional as F

from config import Config 
from blocks.instance_norm import InstanceNorm
from blocks.embedding import PatchEmbedding, VariateEmbedding
from blocks.encoder import Encoder


class TimeXer(nn.Module):
    """
    TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables

    Architecture:
        1. Optional instance normalization on inputs
        2. Patch embedding for the endogenous series
        3. Variate embedding for all exogenous series (with optional time marks)
        4. Transformer encoder with cross-attention
        5. Projection head for forecasting

    Args:
        config (Config): Configuration object containing model hyperparameters,
            sequence lengths, and training settings.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.norm = InstanceNorm()
        self.norm_ = InstanceNorm()

        self.embed = PatchEmbedding(
            seq_len=config.seq_len,
            patch_len=config.model.patch_len,
            patch_overlap=config.model.patch_overlap,
            d_model=config.model.d_model,
            glb_token=config.model.glb_token,
            dropout=config.model.dropout,
            bias=config.model.bias
        )

        self.embed_ = VariateEmbedding(
            seq_len=config.seq_len,
            d_model=config.model.d_model,
            dropout=config.model.dropout,
            bias=config.model.bias
        )

        self.encoder = Encoder(
            n_blocks=config.model.n_encoder_blocks,
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            d_ff=config.model.d_ff,
            dropout=config.model.dropout,
            bias=config.model.bias
        )

        self.head = nn.Sequential(
            nn.Flatten(-2),
            nn.LazyLinear(config.label_len + config.pred_len, bias=config.model.bias)
        )

    def forward(
        self,
        endo: torch.Tensor,
        exo: torch.Tensor,
        marks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass for time series forecasting.

        Processes endogenous and exogenous time series through embeddings,
        encoder, and prediction head to generate forecasts.

        Args:
            endo (torch.Tensor): Endogenous (target) input tensor of shape
                `(batch, seq_len)` representing the univariate time series to be forecasted.
            exo (torch.Tensor): Exogenous (covariate) input tensor of shape
                `(batch, seq_len, n_vars)` containing auxiliary variables that
                may help improve predictions.
            marks (torch.Tensor | None): Optional temporal markers or features
                (e.g., hour-of-day, day-of-week) of shape `(batch, seq_len, k)`.

        Returns:
            torch.Tensor: Forecasted values of shape
                `(batch, label_len + pred_len)` where `label_len` is the overlap
                with input and `pred_len` is the future horizon.
        """

        if self.config.model.use_instance_norm:
            endo = self.norm(endo)
            exo = self.norm_(exo)

        endo = self.embed(endo)
        exo = self.embed_(exo, marks)
        endo = self.encoder(endo, exo)
        endo = self.head(endo.permute(0, 2, 1))

        if self.config.model.use_instance_norm:
            endo = self.norm(endo, norm=False)

        return endo