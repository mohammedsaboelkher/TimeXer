import math
import torch
import torch.nn as nn, torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    sinusoidal positional encoding.

    Args:
        d_model (int): Dimensionality of the model / embedding size.
        max_len (int): Maximum sequence length supported by the encoding.
    """
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000
    ):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_len, d_model)`.

        Returns:
            torch.Tensor: The input tensor with positional encodings added,
                same shape as `x`.
        """
        x = x + self.pe[:, :x.size(1)]

        return x

class PatchEmbedding(nn.Module):
    """
    Convert a multivariate time series into overlapping patches and project each
    patch into a d-dimensional embedding space.

    Args:
        patch_len (int): Length (number of timesteps) of each patch.
        patch_overlap (int): Number of timesteps that adjacent patches overlap.
            Must satisfy `0 <= patch_overlap < patch_len`.
        seq_len (int): Total input sequence length (timesteps).
        d_model (int): Dimension of the output embedding for each patch.
        dropout (float): Dropout probability applied to patch embeddings.
        bias (bool): Whether to use bias in the internal linear projection.
    """
    def __init__(
        self,
        patch_len: int,
        patch_overlap: int,
        seq_len: int,
        d_model: int,
        glb_token: bool = True,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        if not 0 <= patch_overlap < patch_len:
            raise ValueError("patch_overlap must be in the range [0, patch_len)")

        if ((seq_len - patch_len) / (patch_len - patch_overlap)) % 1 == 0:
            raise ValueError("((seq_len - patch_len) / (self.patch_len - self.patch_overlap)) % 1 == 0 must be true to create valid patches")

        self.patch_len = patch_len
        self.patch_overlap = patch_overlap
        self.seq_len = seq_len
        self.d_model = d_model

        if glb_token:
            self.token = nn.Parameter(torch.randn(1, 1, 1, d_model))

        self.proj = nn.Linear(patch_len, d_model, bias=bias)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

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

        x = self.positional_encoding(self.proj(x))

        if hasattr(self, "token"):
            batch_size = x.shape[0]
            x = torch.cat([self.token.repeat((batch_size, 1, 1, 1)), x], dim=1)

        return self.dropout(x)

class VariateEmbedding(nn.Module):
    """
    Embed exogenous (variates) signals across the sequence dimension.

    Args:
        seq_len (int): Length of the input sequences (timesteps).
        d_model (int): Output embedding dimensionality.
        dropout (float): Dropout probability applied to embeddings.
        bias (bool): Whether to use bias in the linear projection.
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        self.proj = nn.Linear(seq_len, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute variate embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_len, n_vars)`
            x_mark (torch.Tensor | None): Optional additional features/marks
                for each timestep (e.g., time-of-day, calendar features).
                Expected shape `(batch, seq_len, k)`

        Returns:
            torch.Tensor: Output embeddings of shape `(batch, n_vars, d_model)`
                (or `(batch, n_vars + k, d_model)` if `x_mark` is used).
        """
        x = x.permute(0, 2, 1)
        x = self.proj(x) if x_mark is None else self.proj(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        return self.dropout(x)