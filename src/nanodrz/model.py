import torch
from torch.nn import Module
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from nanodrz.config import ModelConfig, Config
from nanodrz.modules import ScaledSinusoidalEmbedding, Decoder
from nanodrz.utils import to_device, make_padding_mask
from nanodrz import utils

import dac


class DiarizeGPT(Module):
    """
    Decoder Only

    [DAC Z Latents {dmodel}] -> [Quantized Start Sec, Quantized End Sec, Label

    """

    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        modelcfg = config.model
        dmodel = modelcfg.dmodel

        # Let's predict at least 8 speakers
        self.num_classes = 8

        # From the Coordinate Qunatization notebook , we saw that we need 288 ~50ms errors on average
        # Round to the nearest power of 2
        self.num_embs = 512

        # 2 for eos and pad 1 for
        self.num_time_tokens = self.num_embs - (2 + self.num_classes)

        self.eos_idx = 1
        self.pad_idx = 0
        self.text_emb = nn.Embedding(self.num_embs, dmodel)

        self.text_head = nn.Linear(dmodel, self.num_embs)

        # Load DAC
        model_path = dac.utils.download(model_type=modelcfg.dac_model)
        self.dac: dac.DAC = dac.DAC.load(model_path).eval()
        
        # We're not in need of this as we're not converting back to audio.
        del self.dac.decoder

        self.start_diarize_emb = nn.Parameter(torch.zeros(dmodel))

        self.audio_proj = nn.Linear(self.dac.latent_dim, dmodel)

        # Positional Encoding
        self.audio_pos_emb = ScaledSinusoidalEmbedding(dmodel)
        self.text_pos_emb = ScaledSinusoidalEmbedding(dmodel)

        # This is to embed the position of the time in the token representing the audio
        self.time_pos_emb = ScaledSinusoidalEmbedding(dmodel)

        self.decoder = Decoder(
            d_model=dmodel,
            n_heads=modelcfg.nheads,
            n_layers=modelcfg.layers,
            bias=modelcfg.bias,
            dropout=modelcfg.dropout,
        )

        # We want to init only these modules and leave the rest
        self.init_mod_weights = [
            self.decoder,
            self.text_head,
            self.audio_pos_emb,
            self.text_pos_emb,
            self.audio_proj,
        ]

        for w in self.init_mod_weights:
            w.apply(self._init_weights)

        self._freeze_components()

    def _freeze_components(self):
        # Freeze DAC
        [p.requires_grad_(False) for p in self.dac.parameters()]

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

        audio = self.audio_proj(audio)
        text_embs = self.text_emb(labels)

        # view of the start and end tokens for each triplet in the sequence of coords
        # [start, end, label, start, end, label, ...]
        # [
        #   [start, end],
        #   [start, end]
        #   ...
        # ]
        time_boundaries = labels.view(B, -1, 3)[:, :, :2]
        # Add concept of time to the time boundary tokens
        rearrange(text_embs, "B (b s) L -> B b s L", s=3)[:, :, :2].add_(
            self.time_pos_emb(time_boundaries)
        )
        text_embs = text_embs + self.text_pos_emb(torch.arange(text_embs.shape[1]))

        audio = audio + self.audio_pos_emb(torch.arange(audio.shape[1]))

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

        embs = pad_sequence(embs, batch_first=True)
        x = self.decoder(embs)

        text_latents = [x[b, audio_lengths[b] : audio_lengths[b] + label_lengths[b]] for b in range(B)]
        text_latents = pad_sequence(text_latents, batch_first=True)
        text_logits = self.text_head(text_latents).permute(0, 2, 1)
        loss = F.cross_entropy(text_logits, labels, ignore_index=self.pad_idx)
        return loss

    @torch.inference_mode()
    def generate(
        self,
        audio: Tensor,
        temperature=0.8,
        max_steps=100,
        top_k: int = 10,
        top_p: float = 0.0,
    ):
        with torch.no_grad():
            audio = self.dac.encode(audio[None])[0]
            audio = rearrange(audio, "B L T -> B T L")

        audio = self.audio_proj(audio)

        audio = audio + self.audio_pos_emb(torch.arange(audio.shape[1]))

        emb = torch.cat((audio, self.start_diarize_emb[None][None]), dim=1)

        tokens = []
        for _ in range(max_steps):
            latents = self.decoder(emb)[:, [-1], :]
            logits = self.text_head(latents)
            logits[..., 0] = -1e9

            # if repetition_penalty != 1.0:
            #     begin = max(0, offset - repetition_penalty_window)

            #     score = torch.gather(logits, -1, sequence[[0], :, begin:offset])
            #     # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            #     score = torch.where(score <= 0.0, score * repetition_penalty, score / repetition_penalty)
            #     logits = logits.scatter(-1, sequence[[0], :, begin:offset], score)

            probs = F.softmax(logits / temperature, dim=-1)
            eos_probs = probs[..., self.eos_idx]
            if torch.any(eos_probs > 0.1):
                print(f"{eos_probs=} greater than threshold - early stopping")
                break

            if top_k is not None and top_k > 0:
                next_token = utils.sample_top_k(probs, top_k)
            elif top_p is not None and top_p > 0.0:
                next_token = utils.sample_top_p(probs, top_p)
            else:
                next_token = utils.multinomial(probs, num_samples=1)

            next_token = next_token.flatten()
            tokens.append(next_token.item())

            next_emb = self.text_emb_projection(self.text_emb(next_token))[None]
            emb = torch.cat((emb, next_emb), dim=1)

        return self.text_tokenizer.decode(torch.tensor(tokens))

    @staticmethod
    def from_pretrained(ckpt: str | dict):
        """
        {
            "config": config.model_dump(),
            "step": step,
            "model": {
                k: v
                for k, v in model.state_dict().items()
                if not k.startswith("dac.")
            },
            "optimizer": optimizer.state_dict(),
        }
        """
        if type(ckpt) is str:
            ckpt = torch.load(ckpt)

        config = Config(**ckpt["config"])
        model = DiarizeGPT(config)
        model.load_state_dict(ckpt["model"], strict=False)
        return model


def main():
    """
    Performs a quick training step
    """
    from nanodrz.data import (
        artificial_diarisation_sample,
        libritts_test,
    )

    device_type = "cuda"

    model = ModelConfig()
    config = DiarizeGPT(config).cuda()
    B = 2
    speakers = libritts_test()

    audios = []
    labels = []

    for _ in range(B):
        audio, label = artificial_diarisation_sample(speakers, max_secs=30, sr=16000)
        audio = model.dac.preprocess(audio, config.model.dac.sample_rate)
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
