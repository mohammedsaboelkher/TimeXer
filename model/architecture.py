import torch
import torch.nn as nn

from model.modules import (
    Encoder,
    InstanceNorm,
    PatchEmbedding,
    VariateEmbedding,
)

class TimeXer(nn.Module):
    """
    Empowering Transformers for Time Series Forecasting with Exogenous Variables.

    Args:
        n_encoder_blocks (int): Number of encoder blocks.
        patch_len (int): Length of each patch.
        patch_overlap (int): Overlap between patches.
        n_heads (int): Number of attention heads.
        d_ff (int): Dimension of the feedforward network.
        pred_len (int): Prediction length (horizon + overlap).
        d_model (int): Dimension of the model.
        use_instance_norm (bool): Whether to use instance normalization.
        dropout (float): Dropout rate.
        bias (bool): Whether to use bias in linear layers.
    """
    def __init__(
        self,
        n_encoder_blocks: int,
        patch_len: int,
        patch_overlap: int,
        n_heads: int,
        d_ff: int,
        pred_len: int,
        d_model: int,
        use_instance_norm: bool = True,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()

        if use_instance_norm:
            self.instance_norm = nn.ModuleList(
                [
                    InstanceNorm(), # for the endogenous series
                    InstanceNorm()  # for the exogenous series
                ]
            )

        self.patch_embed = PatchEmbedding(
            patch_len=patch_len,
            patch_overlap=patch_overlap,
            d_model=d_model,
            dropout=dropout,
            bias=bias
        )

        self.variate_embed = VariateEmbedding(
            d_model=d_model,
            dropout=dropout,
            bias=bias
        )

        self.encoder = Encoder(
            n_blocks=n_encoder_blocks,
            n_heads=n_heads,
            d_ff=d_ff,
            d_model=d_model,
            dropout=dropout,
            bias=bias
        )

        self.head = nn.Sequential(
            nn.Flatten(-2),
            nn.LazyLinear(pred_len, bias=bias)
        )

    def forward(
        self,
        endo: torch.Tensor,
        exo: torch.Tensor,
        ext: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            endo (torch.Tensor): Endogenous (target) input tensor of shape
                `(batch, seq_len)` representing the univariate time series to be forecasted.
            exo (torch.Tensor): Exogenous (covariate) input tensor of shape
                `(batch, seq_len, n_vars)` containing auxiliary variables that
                may help improve predictions.
            ext (torch.Tensor | None): Optional temporal markers or features of shape
                `(batch, seq_len, n_ext_vars)` that provide additional context

        Returns:
            torch.Tensor: Forecasted values of shape
                `(batch, pred_len)` where `pred_len` is overlap + horizon.
        """

        if hasattr(self, "instance_norm"):
            endo = self.instance_norm[0](endo)
            exo = self.instance_norm[1](exo)

        patch_embeddings = self.patch_embed(endo)
        variate_embeddings = self.variate_embed(exo, ext)
        embeddings = self.encoder(patch_embeddings, variate_embeddings)
        horizon = self.head(embeddings.permute(0, 2, 1))

        if hasattr(self, "instance_norm"):
            horizon = self.instance_norm[0](horizon, norm=False)

        return horizon