import torch
import torch.nn as nn

from model.modules.positional_encoding import PositionalEncoding


class PatchEmbedding(nn.Module):
    """
    Convert a multivariate time series into overlapping patches and project each
    patch into a d-dimensional embedding space.

    Args:
        patch_len (int): Length (number of timesteps) of each patch.
        patch_overlap (int): Number of timesteps that adjacent patches overlap.
            Must satisfy `0 <= patch_overlap < patch_len`.
        d_model (int): Dimension of the output embedding for each patch.
        dropout (float): Dropout probability applied to patch embeddings.
        bias (bool): Whether to use bias in the internal linear projection.
    """
    def __init__(
        self,
        patch_len: int,
        patch_overlap: int,
        d_model: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        if not 0 <= patch_overlap < patch_len:
            raise ValueError("patch_overlap must be in the range [0, patch_len)")

        self.patch_len = patch_len
        self.patch_overlap = patch_overlap

        self.patch_proj = nn.Linear(patch_len, d_model, bias=bias)
        self.positional_encoding = PositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)

        self.token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patch embeddings from the input time series.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len)`

        Returns:
            torch.Tensor: Patch embeddings with shape
                `(batch, n_patches, d_model)` where `n_patches` is the
                number of patches created from the input sequence (including the global token if used).
        """
        x = x.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.patch_len - self.patch_overlap
        )
        x = self.positional_encoding(self.patch_proj(x))
        x = torch.cat([self.token.repeat((x.shape[0], 1, 1)), x], dim=1)

        return self.dropout(x)