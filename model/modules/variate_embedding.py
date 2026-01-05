import torch
import torch.nn as nn

class VariateEmbedding(nn.Module):
    """
    Embed exogenous (variates) signals across the sequence dimension.

    Args:
        d_model (int): Output embedding dimensionality.
        dropout (float): Dropout probability applied to embeddings.
        bias (bool): Whether to use bias in the linear projection.
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.variate_proj = nn.LazyLinear(d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        ext: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Compute variate embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_len, n_vars)`
            ext (torch.Tensor | None): Optional additional features/marks
                Expected shape `(batch, seq_len, k)`

        Returns:
            torch.Tensor: Output embeddings of shape `(batch, n_vars, d_model)`
                (or `(batch, n_vars + k, d_model)` if `ext` is used).
        """
        x = x.permute(0, 2, 1)
        x = self.variate_proj(x) if ext is None else self.variate_proj(torch.cat([x, ext.permute(0, 2, 1)], 1))

        return self.dropout(x)
