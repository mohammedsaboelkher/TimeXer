import torch
import torch.nn as nn

from model.modules.encoder_block import EncoderBlock


class Encoder(nn.Module):
    """
    Stack of encoder blocks for transformer-based encoding.

    Args:
        n_blocks (int): Number of encoder blocks to stack.
        n_heads (int): Number of attention heads in each block.
        d_model (int): Model dimensionality / embedding size.
        d_ff (int): Hidden size of the feed-forward network in each block.
        dropout (float): Dropout probability applied in each block.
        bias (bool): Whether to use bias in linear layers.
    """
    def __init__(
        self,
        n_blocks: int,
        n_heads: int,
        d_ff: int,
        d_model: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                n_heads=n_heads,
                d_ff=d_ff,
                d_model=d_model,
                dropout=dropout,
                bias=bias
            ) for _ in range(n_blocks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        cross: torch.Tensor,
        x_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
        is_x_causal: bool = False,
        is_cross_causal: bool = False
    ) -> torch.Tensor:
        """
        Process input through all encoder blocks sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_len, d_model)`.
            cross (torch.Tensor): Cross-attention context of shape
                `(batch, cross_len, d_model)`.
            x_mask (torch.Tensor | None): Optional mask for self-attention.
            cross_mask (torch.Tensor | None): Optional mask for cross-attention.
            is_x_causal (bool): Apply causal masking in self-attention.
            is_cross_causal (bool): Apply causal masking in cross-attention.

        Returns:
            torch.Tensor: Encoded output of shape `(batch, seq_len, d_model)`.
        """
        for layer in self.layers:
            x = layer(
                x,
                cross,
                x_mask=x_mask,
                cross_mask=cross_mask,
                is_x_causal=is_x_causal,
                is_cross_causal=is_cross_causal
            )

        return x