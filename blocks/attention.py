import torch
import torch.nn as nn, torch.nn.functional as F

class MHAttention(nn.Module):
    """
    multi-head attention module, uses torch sdpa (scaled_dot_product_attention)

    Args:
        query_embed_dim (int): size of embedding dim for query
        key_embed_dim (int): size of embedding dim for key
        value_embed_dim (int): size of embedding dim for value
        attn_embed_dim (int): total embedding dim of combined heads post input projection. Each head has dim attn_embed_dim // n_heads
        output_embed_dim (int): size of embedding dim for output
        n_heads (int): number of heads
        dropout (float, optional): dropout probability. Default: 0.0
        bias (bool, optional): whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        n_heads: int,
        query_embed_dim: int,
        key_embed_dim: int,
        value_embed_dim: int,
        attn_embed_dim: int,
        output_embed_dim: int,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        if attn_embed_dim % n_heads != 0:
            raise ValueError("attn_embed_dim must be divisible by n_heads")

        self.n_heads = nheads
        self._qkv_same_embed_dim = query_embed_dim == key_embed_dim and query_embed_dim == value_embed_dim

        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(query_embed_dim, attn_embed_dim * 3, bias=bias)
        else:
            self.q_proj = nn.Linear(query_embed_dim, attn_embed_dim, bias=bias)
            self.k_proj = nn.Linear(key_embed_dim, attn_embed_dim, bias=bias)
            self.v_proj = nn.Linear(value_embed_dim, attn_embed_dim, bias=bias)

        self.out_proj = nn.Linear(attn_embed_dim, output_embed_dim, bias=bias)
        self.attn_head_embed_dim = attn_embed_dim // n_heads


        self.dropout = dropout
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): query of shape (N, L_q, q,k_embed_dim)
            key (torch.Tensor): key of shape (N, L_kv, q,k_embed_dim)
            value (torch.Tensor): value of shape (N, L_kv, value_embed_dim)
            attn_mask (torch.Tensor, optional): attention mask of shape (N, L_q, L_kv) to pass to SDPA. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, query_embed_dim)
        """
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                query, key, value = torch.chunk(self.packed_proj(query), 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )
        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        query = query.unflatten(-1, [self.n_heads, self.attn_head_embed_dim]).transpose(1, 2)
        key = key.unflatten(-1, [self.n_heads, self.attn_head_embed_dim]).transpose(1, 2)
        value = value.unflatten(-1, [self.n_heads, self.attn_head_embed_dim]).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout,
            attn_mask=attn_mask,
            is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.out_proj(attn_output)

        return attn_output