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
        n_vars (int): Number of variables / channels in the input time series.
        dropout (float): Dropout probability applied to patch embeddings.
        bias (bool): Whether to use bias in the internal linear projection.
    """
    def __init__(
        self,
        patch_len: int,
        patch_overlap: int,
        seq_len: int,
        d_model: int,
        n_vars: int,
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
        self.n_vars = n_vars

        self.proj = nn.Linear(patch_len, d_model, bias=bias)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patch embeddings from the input time series.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, n_vars)`
            if n_vars passed to the constructor == 1, then it's considered univariate
            patching is only applied to the last channel, other than n_vars == 1 it's considered multivariate and patching is applied to the whole input.

        Returns:
            torch.Tensor: Patch embeddings with shape
                `(batch, n_vars, n_patches, d_model)` where `n_patches` is the
                number of patches created from the input sequence.
        """
        x = (x if self.n_vars > 1 else x[:, :, -1].unsqueeze(-1)).permute(0, 2, 1).unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.patch_len - self.patch_overlap
        )

        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = self.positional_encoding(self.proj(x))
        x = x.reshape(-1, self.n_vars, x.shape[-2], x.shape[-1])
        x = self.dropout(x)

        return x

class VariateEmbedding(nn.Module):
    """
    Embed exogenous (variates) signals across the sequence dimension.

    This module projects each variable's time series (or a combination of the
    series and an optional `x_mark` tensor) into the `d_model` embedding
    dimension.

    Args:
        seq_len (int): Length of the input sequences (timesteps).
        d_model (int): Output embedding dimensionality.
        n_vars (int): Number of variables / channels in `x`.
        dropout (float): Dropout probability applied to embeddings.
        bias (bool): Whether to use bias in the linear projection.
    """
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_vars: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_vars = n_vars

        self.proj = nn.Linear(seq_len, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute variate embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_len, n_vars)`
                or `(batch, seq_len)` for univariate. The implementation
                expects channels last and will permute to `(batch, n_vars, seq_len)`.
            x_mark (torch.Tensor | None): Optional additional features/marks
                for each timestep (e.g., time-of-day, calendar features).
                Expected shape `(batch, seq_len, k)` and will be permuted and
                concatenated along the channel dimension when provided.

        Returns:
            torch.Tensor: Output embeddings of shape `(batch, n_vars, d_model)`
                (or `(batch, n_vars_combined, d_model)` if `x_mark` is used).
        """
        x = (x if self.n_vars > 1 else x[:, :, :-1]).permute(0, 2, 1)
        x = self.proj(x) if x_mark is None else self.proj(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)

class GlobalVariateToken(nn.Module):
    """
    Learnable global token(s) representing each variate (series).

    This module creates a small set of learnable tokens (one per variable) and
    repeats them for the batch dimension so they can be concatenated or used as
    a special token in downstream transformer layers.

    Args:
        d_model (int): Dimensionality of each token embedding.
        n_vars (int): Number of variables / tokens to maintain.
        dropout (float): Dropout probability applied to the tokens.
    """
    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.0):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the batch-repeated global tokens.

        Args:
            x (torch.Tensor): Any tensor with batch dimension at `x.shape[0]`.

        Returns:
            torch.Tensor: Tokens repeated for the batch with shape
                `(batch, n_vars, 1, d_model)`.
        """
        return self.dropout(self.token.repeat((x.shape[0], 1, 1, 1)))