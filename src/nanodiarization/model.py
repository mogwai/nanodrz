import torch
from torch.nn import Module
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, T5EncoderModel
from .config import ModelConfig
from .modules import ScaledSinusoidalEmbedding, Decoder
from .utils import to_device, make_padding_mask

import dac


class DiarizeGPT(Module):
    """
    Decoder Only

    [DAC Z Latents {dmodel}] -> [ByT5 Text Encoder(Start, Duration, Label)]
    
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Get us a good starting point to save some time
        self.text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_model)
        self.text_emb = T5EncoderModel.from_pretrained(config.tokenizer_model).shared

        # Load DAC
        model_path = dac.utils.download(model_type=config.dac_model)
        self.dac: dac.DAC = dac.DAC.load(model_path).eval()

        self.start_diarize_emb = nn.Parameter(torch.zeros(config.dmodel))

        self.eos_idx = self.text_tokenizer.eos_token_id
        self.pad_idx = self.text_tokenizer.pad_token_id

        self.text_head = nn.Linear(config.dmodel, self.text_tokenizer.vocab_size)
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

        # We want to init only these modules and leave the rest
        self.init_mod_weights = [
            self.decoder,
            self.text_head,
            self.text_emb_projection,
            self.audio_proj,
            self.audio_pos_emb,
            self.text_pos_emb,
        ]

        for w in self.init_mod_weights:
            w.apply(self._init_weights)

        self._freeze_components()

    def _freeze_components(self):
        # Freeze DAC
        [p.requires_grad_(False) for p in self.dac.parameters()]
        [p.requires_grad_(False) for p in self.text_emb.parameters()]

    def train(self, mode: bool = True):
        res = super().train(mode)
        self._freeze_components()
        return res

    def configure_optimizers(
        self, *, weight_decay: float, lr: float, betas: tuple[float, float]
    ):
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        audio: Tensor,
        labels: Tensor,
        audio_lengths: Tensor,
        label_lengths: Tensor,
    ):
        B = audio.shape[0]

        # DAC Z Latent Reduction factor
        audio_lengths = audio_lengths // 320

        with torch.no_grad():
            audio = self.dac.encode(audio)[0]
            audio = rearrange(audio, "B L T -> B T L")

        audio = self.audio_proj(audio) + self.audio_pos_emb(
            torch.arange(audio.shape[1])
        )
        text_embs = self.text_emb(labels)
        text_embs = self.text_emb_projection(text_embs) + self.text_pos_emb(
            torch.arange(text_embs.shape[1])
        )

        embs = []
        for b in range(B):
            emb = torch.cat(
                (
                    audio[b, : audio_lengths[b]],
                    self.start_diarize_emb[None],
                    # We're predicting up to eos token (which is included in the sequence) 
                    text_embs[b, : label_lengths[b] - 1],
                ),
                dim=0,
            )
            embs.append(emb)

        # + 1 for diarization emb
        seqlens = audio_lengths + label_lengths
        if self.config.use_flash_attn:
            embs = torch.cat(embs, dim=1)
            max_seqlen = audio.shape[1] + labels.shape[1]
            cu_seqlens = F.pad(seqlens.cumsum(0, dtype=torch.int32), (1, 0), value=0)
            x = self.decoder(embs, cu_seqlens, max_seqlen)
        else:
            embs = pad_sequence(embs, batch_first=True)
            mask = ~make_padding_mask(seqlens)
            x = self.decoder(embs, mask=mask)

        text_latents = []

        for b in range(B):
            text_latents.append(
                x[b, audio_lengths[b] : audio_lengths[b] + label_lengths[b]]
            )

        text_latents = pad_sequence(text_latents, batch_first=True)
        text_logits = self.text_head(text_latents).permute(0, 2, 1)
        loss = F.cross_entropy(text_logits, labels, ignore_index=self.pad_idx)

        return {"loss": loss}


def main():
    """
    Performs a quick training step
    """
    from nanodiarization.data import (
        gather_speakers_from_folder,
        artificial_diarisation_sample,
    )
    from nanodiarization import download

    device_type = "cuda"

    config = ModelConfig()
    model = DiarizeGPT(config).cuda()
    B = 2
    folder = download.dl_libritts_clean()
    speakers = gather_speakers_from_folder(folder, lambda x: x.split("/")[-3])

    audios = []
    labels = []
    for i in range(B):
        audio, label = artificial_diarisation_sample(speakers, max_secs=30, sr=16000)
        audio = model.dac.preprocess(audio, model.dac.sample_rate)
        audios.append(rearrange(audio, "c s -> s c"))
        labels.append("\n".join([",".join([str(x) for x in l]) for l in label]))

    text_tokens = model.text_tokenizer.batch_encode_plus(
        labels,
        return_tensors="pt",
        padding="longest",
    )

    # Preprocess the audio here and get the lengths
    label_lengths = text_tokens["attention_mask"].sum(dim=-1)
    label_lengths = to_device(label_lengths, "cuda")
    text_tokens = text_tokens["input_ids"].cuda()

    audio_lengths = torch.tensor([a.shape[0] for a in audios]) // 320
    audio_lengths = to_device(audio_lengths, "cuda")

    audios = pad_sequence(audios, batch_first=True)
    audios = rearrange(audios, "B S C -> B C S")
    audios = to_device(audios, "cuda")

    dtype = torch.float16

    with torch.autocast(enabled=True, device_type=device_type, dtype=dtype):
        print(model(audios, text_tokens, audio_lengths, label_lengths))


if __name__ == "__main__":
    main()
