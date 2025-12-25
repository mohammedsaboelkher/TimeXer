import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
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

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class PatchEmbedding(nn.Module):
    """
    Patches a time series and embeds it
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

        if ((seq_len - patch_len) / (self.patch_len - self.patch_overlap)) % 1 == 0:
            raise ValueError("((seq_len - patch_len) / (self.patch_len - self.patch_overlap)) % 1 == 0 must be true to create valid patches")

class VariateEmbedding(nn.Module):
    """
    Embeds exogenous variables
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

    def forward(self, x, x_mark):
        x = (x if n_vars > 1 else x[:, :, :-1]).permute(0, 2, 1)
        x = self.proj(x) if x_mark is None else self.proj(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)
        self.patch_len = patch_len
        self.patch_overlap = patch_overlap
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_vars = n_vars

        self.proj = nn.Linear(patch_len, d_model, bias=bias)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = (x if self.n_vars > 1 else x[:, :, -1].unsqueeze(-1)).permute(0, 2, 1).unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.patch_len - self.patch_overlap
        )

        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        x = self.proj(x) + self.positional_encoding(x)
        x = x.reshape(-1, self.n_vars, x.shape[-2], x.shape[-1])
        x = self.dropout(x)

        return x

class GlobalSeriesToken(nn.Module):
    """
    Generates a learnable global token for each series
    """
    def __init__(self, d_model: int, n_vars: int, dropout: float = 0.0):
        super().__init__()

        self.token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.token.repeat((x.shape[0], 1, 1, 1)))