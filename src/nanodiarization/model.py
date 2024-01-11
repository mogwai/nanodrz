import torch
from torch.nn import Module
from torch import Tensor, nn
import torch.nn.functional as F
from flash_attn.bert_padding import pad_input, unpad_input
from torch.nn.utils.rnn import pad_sequence
import einops

from transformers import AutoTokenizer, T5EncoderModel
from .config import ModelConfig

import dac

class DiarizeGPT(Module):

    def __init__(self, config:ModelConfig):
        super().__init__()
        # Load 
        self.config = config
        name = "google/byt5-large"
        self.text_tokenizer = AutoTokenizer.from_pretrained(name)
        model = T5EncoderModel.from_pretrained(name)
        self.config.dmodel = model.d_model

        # Load DAC
        model_path = dac.utils.download(model_type="16khz")
        self.dac: dac.DAC = dac.DAC.load(model_path).eval()
        
        # self.audio_head = nn.Linear(config.dmodel, self.dac.codebook_size)
        # self.audio_embs = nn.ModuleList([nn.Linear(self.dac.codebook_size, config.dmodel) for _ in range(self.config.K)])
        
        self.diarize_emb = nn.Parameter(torch.zeros(config.dmodel))
        self.bos_idx = self.text_tokenizer.vocab_size + 1
        self.eos_idx = self.text_tokenizer.vocab_size
        self.text_vocab_size = self.text_tokenizer.vocab_size + 1
        self.text_head = nn.Linear(config.dmodel, self.text_vocab_size)
        self.text_emb = model.shared
        self.text_emb_projection = nn.Linear(self.text_emb.embedding_dim, config.dmodel)
        self.audio_proj = nn.Linear(1024, config.dmodel)

        # Positional Encoding        
        self.audio_pos_emb = ScaledSinusoidalEmbedding(config.dmodel)
        self.text_pos_emb = ScaledSinusoidalEmbedding(config.dmodel)

        self.decoder = Decoder(
            d_model=config.dmodel,
            n_heads=config.nheads,
            n_layers=config.layers,
            bias=config.bias,
            dropout=config.dropout,
        )

    
    def forward(self, audio:Tensor, labels:str, text_lengths: Tensor, audio_lengths: Tensor):
        B = audio.shape[0]

        with torch.no_grad():
            audio = self.dac.preprocess(audio)
            audio = self.dac.encode(audio)[0]
        
        audio = self.audio_proj(audio)
        src = t
        text_tokens = self.text_tokenizer.batch_encode_plus(
            labels,  
            return_tensors="pt",
            padding="longest",
        )

        bos_embs = self.text_emb(self.bos_idx)
        text_embs = self.text_emb(text_tokens)
        eos_emb = self.text_emb(self.eos_idx)


        audio

        embs = []
        for b in range(B):
            embs.append( 
                torch.cat((
                    bos_embs,
                    audio[b, : audio_lengths[b]], 
                    self.diarize_emb, 
                    text_embs[b, : text_lengths[b]], 
                    eos_emb
                    ), 
                dim=0
                )
            )
        
        seqlens = text_lengths + audio_lengths
        max_seqlen = audio.shape[1] + labels.shape[-1] 

        cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        latents = self.decoder(embs, cu_seqlens, max_seqlen)
        
        x = torch.split(latents, seqlens.tolist())

        text_latents = []
        for b in range(B):
            text_latents.append(x[b][1+audio_lengths[b]+1:])

        text_pred = pad_sequence(text_latents, batch_first=True)  # (B, S, d_model)
        text_logits = self.text 
        
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
    ):
        super().__init__()

        rotary_emb_dim = d_model if rotary_emb else 0

        self.attn_norm = LayerNorm(d_model, bias=bias)
        self.attn = MHA(
            d_model,
            n_heads,
            cross_attn=False,
            causal=causal,
            dropout=dropout,
            use_flash_attn=True,
            layer_idx=layer_idx,
            qkv_proj_bias=bias,
            out_proj_bias=bias,
            rotary_emb_dim=rotary_emb_dim,
        )

        self.mlp_norm = LayerNorm(d_model, bias=bias)
        self.mlp = MLP(d_model, bias, dropout)

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
        inference_params: InferenceParams | None = None,
    ) -> Tensor:
        x = x + self.attn(
            self.attn_norm(x), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, inference_params=inference_params
        )
        x = x + self.mlp(self.mlp_norm(x))
        return x



class Decoder(nn.Module):
    def __init__(
        self, *, d_model: int, n_heads: int, n_layers: int, bias: bool, causal: bool = True, dropout: float = 0.0
    ):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, bias, dropout, layer_idx=layer_idx, causal=causal)
                for layer_idx in range(n_layers)
            ]
        )
        self.norm_f = LayerNorm(d_model, bias=bias)

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Tensor | None = None,
        max_seqlen: int | None = None,
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

class ScaledSinusoidalEmbedding(Module):
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

if __name__ == "__main__":
    import random
    # TODO Example Inference
    config = ModelConfig()
    model = DiarizeGPT(config)
    K = 4
    B = 3  
    # Load Audio 

    # Load Example Labels
    codes = [] 
    lengths = []
    for i in range(B):
        lengths.append(random.randint(86*3, 86*30))
        codes.append(torch.randint(0, 1023, (K, lengths[-1])))
    
    text = "12.1,2,A\n2"
    model()