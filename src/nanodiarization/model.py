import torch
from torch.nn import Module
from torch import Tensor, nn
import torch.nn.functional as F
from flash_attn.modules.mha import MHA
from flash_attn.bert_padding import pad_input, unpad_input
from torch.nn.utils.rnn import pad_sequence
from flash_attn.utils.generation import InferenceParams

from transformers import AutoTokenizer, T5EncoderModel
from .config import ModelConfig

import dac

class DiarizeGPT(Module):

    def __init__(self, config:ModelConfig):
        super().__init__()
        self.config = config
        # Using the byt5 tokenizer model
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        # Try and use the weights to init this model
        self.text_emb = T5EncoderModel.from_pretrained(config.tokenizer_model).shared

        # Load DAC
        model_path = dac.utils.download(model_type=config.dac_model)
        self.dac: dac.DAC = dac.DAC.load(model_path).eval()
    
        self.start_diarize_emb = nn.Parameter(torch.zeros(config.dmodel))
        self.bos_idx = self.text_tokenizer.vocab_size + 1
        self.eos_idx = self.text_tokenizer.vocab_size
        self.text_vocab_size = self.text_tokenizer.vocab_size + 1
        self.text_head = nn.Linear(config.dmodel, self.text_vocab_size)
        self.text_emb_projection = nn.Linear(self.text_emb.embedding_dim, config.dmodel)
        self.audio_proj = nn.Linear(self.dac.codebook_size, config.dmodel)

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
        self.freeze_components()

    
    def forward(self, audio:Tensor, labels:str, text_lengths: Tensor, audio_lengths: Tensor):
        B = audio.shape[0]

        with torch.no_grad():
            audio = self.dac.preprocess(audio)
            audio = self.dac.encode(audio)[0]
        
        audio = self.audio_proj(audio) + self.audio_pos_emb(torch.arange(audio.shape[-1]))
        breakpoint()
        text_tokens = self.text_tokenizer.batch_encode_plus(
            labels,  
            return_tensors="pt",
            padding="longest",
        )

        bos_embs = self.text_emb(self.bos_idx)
        text_embs = self.text_emb(text_tokens)
        text_embs = text_embs + self.text_pos_emb(torch.arange(text_embs.shape[-1]))
        eos_emb = self.text_emb(self.eos_idx)

        embs = []
        for b in range(B):
            emb = torch.cat(
                (
                    bos_embs,
                    audio[b, : audio_lengths[b]], 
                    self.diarize_emb, 
                    text_embs[b, : text_lengths[b]],
                ),
                dim=0
            )
            emb = emb + self.audio_pos_emb(torch.arange(emb.shape[-1]))
            embs.append(emb)
        
        seqlens = text_lengths + audio_lengths
        max_seqlen = audio.shape[1] + labels.shape[-1] 

        cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
        latents = self.decoder(embs, cu_seqlens, max_seqlen)
        
        x = torch.split(latents, seqlens.tolist())

        text_latents = []
        for b in range(B):
            text_latents.append(x[b][1+audio_lengths[b]+1:])

        text_pred = pad_sequence(text_latents, batch_first=True)
        text_logits = self.text_head(text_latents)
        loss = F.cross_entropy(text_logits, text_tokens, ignore_index=self.eos_idx)
        return loss
    
    def freeze_components(self):
        # Freeze DAC
        [p.requires_grad_(False) for p in self.dac.parameters()]

    def train(self, mode: bool = True):
        res = super().train(mode)
        self.freeze_components()
        return res

    def configure_optimizers(self, *, weight_decay: float, lr: float, betas: tuple[float, float]):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)

        return optimizer


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
        


def main():
    config = ModelConfig()
    model = DiarizeGPT(config)
    from nanodiarization.data import gather_speakers_from_folder, artificial_diarisation_sample
    B = 2
    speakers = gather_speakers_from_folder("/home/harry/storj/data/LibriTTS/test-clean/", lambda x: x.split("/")[-3])
    
    audios = []
    labels = []
    for i in range(B):
        audio, label = artificial_diarisation_sample(speakers, max_secs=30)
        audios.append(audio)
        labels.append("\n".join([",".join(l) for l in labels]))

    audio_lengths = [a.shape[-1] for a in audios]
    label_lengths = [len(l) for l in labels]
    model(audio, labels)


if __name__ == "__main__":
    main()