import torch
from torch import nn, Tensor
import torchaudio
import torch.nn.functional as F
from einops import rearrange
import torch
from nanodrz.utils import make_padding_mask


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-5)


class MLP(nn.Module):
    def __init__(self, d_model: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.c_proj = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class FusedQueryAtt(nn.Module):

    def __init__(
        self,
        dmodel: int = 1024,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.query = nn.Linear(dmodel, dmodel)
        self.act = F.gelu

    def forward(self, x, lengths):
        B, T, _ = x.shape
        query = self.act((x[:, None])).repeat(1, T, 1, 1)
        padding_mask = ~make_padding_mask(lengths).to(x.device)
        padding_mask = (padding_mask[None].T * padding_mask[None]).permute(1, 0, 2)
        tril = torch.tril(torch.ones(B, T, T, device=x.device))
        nseq = (tril * padding_mask)[..., None] * query
        x = x + nseq.sum(2)
        return


class Attention(nn.Module):
    """
    Memory Efficient or Flash Attention
    """

    def __init__(
        self,
        dmodel: int = 1024,
        n_heads: int = 16,
        bias: bool = False,
        dropout: float = 0.0,
        causal: bool = True,
        flash: bool = True,
    ):
        super().__init__()
        assert dmodel % n_heads == 0
        self.qkv = nn.Linear(dmodel, 3 * dmodel, bias=bias)
        self.c_proj = nn.Linear(dmodel, dmodel, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_heads
        self.n_embd = dmodel
        self.flash = flash
        self.dropout = dropout
        self.causal = causal

    def forward(self, x, mask=None):
        B, T, C = x.size()

        einop = "B T (split heads hs) -> split B heads T hs"
        q, k, v = rearrange(self.qkv(x), einop, heads=self.n_head, split=3)

        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0,
            is_causal=self.causal,
            attn_mask=mask,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        causal: bool = True,
    ):
        super().__init__()
        self.attn_norm = LayerNorm(d_model, bias=bias)
        self.attn = FusedQueryAtt(d_model, n_heads, bias, dropout, causal)
        self.mlp_norm = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model, bias, dropout)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        lengths: Tensor = None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), mask=mask, lengths=lengths)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_layers: int,
        bias: bool,
        causal: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, bias, dropout, causal=causal)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = LayerNorm(d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        lengths: Tensor = None,
    ):
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, mask, lengths=lengths)

        return self.norm_f(x)


class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, d_model: int, theta=10000, max_seqlen: int = 8192):
        super().__init__()
        assert (d_model % 2) == 0
        self.weight = nn.Parameter(torch.ones(1) * d_model**-0.5)

        half_dim = d_model // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta**-freq_seq

        pos = torch.arange(max_seqlen)
        emb = torch.einsum("i, j -> i j", pos, inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        self.register_buffer("emb", emb, persistent=False)

    def forward(self, pos_ids: Tensor):
        emb = self.emb[pos_ids]
        return self.weight * emb


class WhisperConvs(nn.Module):
    def __init__(self, dmodel: int, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, dmodel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(dmodel, dmodel, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        return x.permute(0, 2, 1)
