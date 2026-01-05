import math

import torch
import torch.nn as nn


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