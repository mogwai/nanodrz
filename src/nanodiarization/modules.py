import torch
from torch import nn, Tensor
import torch.nn.functional as F
from flash_attn.modules.mha import MHA
from memory_efficient_attention_pytorch import Attention as MemoryEfficientAttention
from flash_attn.utils.generation import InferenceParams
import torch


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


class Attention(nn.Module):
    """
    Memory Efficient or Flash Attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        causal: bool,
        layer_idx: int | None = None,
        rotary_emb_dim: int = 0,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn

        if use_flash_attn:
            self.attn = MHA(
                d_model,
                n_heads,
                cross_attn=False,
                causal=causal,
                dropout=dropout,
                use_flash_attn=False,
                layer_idx=layer_idx,
                qkv_proj_bias=bias,
                out_proj_bias=bias,
                rotary_emb_dim=rotary_emb_dim,
            )
        else:
            self.attn = MemoryEfficientAttention(
                dim=d_model,
                heads=n_heads,
                dropout=dropout,
                causal=True,
                memory_efficient=True,
            )

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        inference_params: InferenceParams | None = None,
        mask: Tensor = None,
    ):
        if self.use_flash_attn:
            return self.attn(
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                inference_params=inference_params,
            )
        else:
            return self.attn(x, mask=mask)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool,
        dropout: float,
        layer_idx: int | None = None,
        causal: bool = True,
        rotary_emb: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()

        rotary_emb_dim = d_model if rotary_emb else 0

        self.attn_norm = LayerNorm(d_model, bias=bias)
        self.attn = Attention(
            d_model,
            n_heads,
            bias,
            dropout,
            causal,
            layer_idx,
            rotary_emb_dim,
            use_flash_attn,
        )
        self.mlp_norm = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model, bias, dropout)

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        inference_params: InferenceParams | None = None,
        mask: Tensor = None,
    ) -> Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inference_params=inference_params,
        )
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
        dropout: float = 0.0
    ):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_heads, bias, dropout, layer_idx=layer_idx, causal=causal
                )
                for layer_idx in range(n_layers)
            ]
        )
        self.norm_f = LayerNorm(d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        mask: Tensor = None,
        inference_params: InferenceParams | None = None,
    ):
        x = self.drop(x)

        for block in self.blocks:
            x = block(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                inference_params=inference_params,
            )

        x = self.norm_f(x)  # (T_total, d_model) or (B, T, d_model)

        return x


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
