import torch
import torch.nn as nn

class SDPA(nn.Module):
    """
    Computes multi-head attention. Scaled Dot Product Attention

    Args:
        query_embed_dim (int): Size of embedding dim for query
        key_embed_dim (int): Size of embedding dim for key
        value_embed_dim (int): Size of embedding dim for value
        attn_embed_dim (int): Total embedding dim of combined heads post input projection. Each head
            has dim attn_embed_dim // nheads
        output_embed_dim (int): Size of embedding dim for output
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        query_embed_dim: int,
        key_embed_dim: int,
        value_embed_dim: int,
        attn_embed_dim: int,
        output_embed_dim: int,
        nheads: int,
        dropout: float = 0.0,
        bias=True
    ):
        super().__init__()

        if attn_embed_dim % nheads != 0:
            raise ValueError("attn_embed_dim must be divisible by nheads")

        self.nheads = nheads
        self.dropout = dropout
        self._qkv_same_embed_dim = query_embed_dim == key_embed_dim and query_embed_dim == value_embed_dim

        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(query_embed_dim, attn_embed_dim * 3, bias=bias)
        else:
            self.q_proj = nn.Linear(query_embed_dim, attn_embed_dim, bias=bias)
            self.k_proj = nn.Linear(key_embed_dim, attn_embed_dim, bias=bias)
            self.v_proj = nn.Linear(value_embed_dim, attn_embed_dim, bias=bias)

        self.out_proj = nn.Linear(attn_embed_dim, output_embed_dim, bias=bias)
        self.E_head = attn_embed_dim // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask=None,
        is_causal=False,
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
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
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


        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout, is_causal=is_causal
        )

        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.out_proj(attn_output)

        return attn_output

class Attention(nn.Module):
    def __init__(
            self,
            attention,
            d_model,
            n_heads,
            d_keys=None,
            d_values=None
        ):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = SDPA()
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn