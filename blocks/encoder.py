import torch
import torch.nn as nn, torch.nn.functional as F

from attention import MHAttention


class EncoderBlock(nn.Module):
    """
    Single encoder block with self-attention, cross-attention, and feed-forward layers.

    This block applies:
    1. Self-attention on the input sequence
    2. Cross-attention between a global token (first position) and a cross-input
    3. Position-wise feed-forward network
    Each sub-layer uses residual connections and layer normalization.

    Args:
        d_model (int): Dimensionality of the model embeddings.
        n_heads (int): Number of attention heads for multi-head attention.
        d_ff (int): Hidden dimensionality of the feed-forward network.
        dropout (float): Dropout probability applied throughout the block.
        bias (bool): Whether to use bias in linear projections.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.attn = MHAttention(
            n_heads=n_heads,
            query_embed_dim=d_model,
            key_embed_dim=d_model,
            value_embed_dim=d_model,
            attn_embed_dim=d_model,
            output_embed_dim=d_model,
            dropout=dropout,
            bias=bias
        )
        self.cross_attn = MHAttention(
            n_heads=n_heads,
            query_embed_dim=d_model,
            key_embed_dim=d_model,
            value_embed_dim=d_model,
            attn_embed_dim=d_model,
            output_embed_dim=d_model,
            dropout=dropout,
            bias=bias
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias),
            nn.Dropout(dropout)
        )
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])

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
        Process input through encoder block with self-attention and cross-attention.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch, seq_len, d_model)`.
                The first position `x[:, 0, :]` is treated as a global token.
            cross (torch.Tensor): Cross-attention context tensor of shape
                `(batch, cross_len, d_model)`.
            x_mask (torch.Tensor | None): Optional attention mask for self-attention.
            cross_mask (torch.Tensor | None): Optional attention mask for cross-attention.
            is_x_causal (bool): Whether to apply causal masking in self-attention.
            is_cross_causal (bool): Whether to apply causal masking in cross-attention.

        Returns:
            torch.Tensor: Output tensor of shape `(batch, seq_len, d_model)`.
        """
        N, L, D = x.shape
        x = self.norm[0](x + self.attn(x, x, x, attn_mask=x_mask, is_causal=is_x_causal))

        x_glb_token = x[:, 0, :].unsqueeze(1).reshape(N, 1, D)
        x_glb_token= self.norm[1](x_glb_token + self.cross_attn(
            x_glb_token,
            cross,
            cross,
            attn_mask=cross_mask,
            is_causal=is_cross_causal
        ))

        x = torch.cat([x_glb_token, x[:, 1:, :]], dim=1)
        x = self.norm[2](x + self.ff(x))

        return x


class Encoder(nn.Module):
    """
    Stack of encoder blocks for transformer-based encoding.

    Processes input sequences through multiple encoder blocks, each applying
    self-attention, cross-attention (on the global token), and feed-forward
    transformations.

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
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
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

