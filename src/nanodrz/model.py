import torch
from torch.nn import Module
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

from nanodrz.config import ModelConfig, Config
from nanodrz.modules import ScaledSinusoidalEmbedding, Decoder, WhisperConvs
from nanodrz.utils import to_device, make_padding_mask, mel_spec
from nanodrz import utils
from functools import partial

import torch


def quantize(time, max_seconds, num_time_tokens):
    return time / max_seconds * num_time_tokens + 2


def dequantize(time, max_secs, num_time_tokens):
    return (time - 2) / num_time_tokens * max_secs


def enc_label(label, num_embs):
    return num_embs - 1 - (ord(label) - ord("A"))


def dec_label(label, num_embs):
    return chr(ord("A") + (num_embs - (label + 1)).item())


def check_valid_label(token_so_far) -> bool:
    triplets = token_so_far.split(3).tolist()
    last_label = triplets[-1]
    start, end, cls = last_label

    # same class
    triplets = filter(lambda x: x[2] == cls, triplets)

    for lab in triplets[:-1]:
        # Not intersecting.
        if (start > lab[0] and start > lab[1]) or end < lab[0] or lab[1]:
            continue


class DiarizeGPT(Module):
    """
    Decoder Only

    [DAC Z Latents {dmodel}] -> [Quantized Start Sec, Quantized End Sec, Label

    """

    def __init__(self, config: Config = Config()):
        super().__init__()

        self.config = config
        modelcfg = config.model
        datacfg = config.data
        dmodel = modelcfg.dmodel

        # Let's predict at least 8 speakers
        self.num_classes = config.data.num_speakers

        # From the Coordinate Qunatization notebook , we saw that we need 288 ~50ms errors on average
        # Round to the nearest power of 2
        self.num_embs = modelcfg.num_embs

        # 2 for eos and pad 1 for
        self.num_time_tokens = self.num_embs - (2 + self.num_classes)

        self.eos_idx = 1
        self.pad_idx = 0
        self.text_emb = nn.Embedding(self.num_embs, dmodel)

        self.text_head = nn.Linear(dmodel, self.num_embs, bias=modelcfg.bias)

        self.start_diarize_emb = nn.Parameter(torch.zeros(dmodel))
        torch.nn.init.normal_(self.start_diarize_emb, mean=0.0, std=0.02)

        self.init_mod_weights = []

        # Determine the encoding o
        if modelcfg.audio_encode == "dac":
            self.audio_proj = nn.Linear(self.dac.latent_dim, dmodel)
            self.init_mod_weights += [self.audio_proj]
        elif modelcfg.audio_encode == "dac-codes":
            self.whispconv = WhisperConvs(
                dmodel, self.dac.codebook_dim * self.dac.n_codebooks
            )
            self.init_mod_weights += [self.whispconv]
        elif modelcfg.audio_encode == "mel":

            self.whispconv = WhisperConvs(
                dmodel,
                datacfg.n_mels,
            )
            self.mel = partial(mel_spec, n_mels=datacfg.n_mels, sr=modelcfg.sample_rate)
            self.init_mod_weights += [self.whispconv]

        # Positional Encoding
        self.audio_pos_emb = ScaledSinusoidalEmbedding(dmodel)
        self.text_pos_emb = ScaledSinusoidalEmbedding(dmodel)

        # This is to embed the position of the time in the token representing the audio
        if modelcfg.use_time_pos:
            self.time_pos_emb = ScaledSinusoidalEmbedding(dmodel)

        self.decoder = Decoder(
            d_model=dmodel,
            n_heads=modelcfg.nheads,
            n_layers=modelcfg.layers,
            bias=modelcfg.bias,
            dropout=modelcfg.dropout,
        )

        # We want to init only these modules and leave the rest
        self.init_mod_weights += [
            self.decoder,
            self.text_head,
            self.audio_pos_emb,
            self.text_pos_emb,
        ]

        for w in self.init_mod_weights:
            w.apply(self._init_weights)

        self._freeze_components()

    def _freeze_components(self):
        # Freeze DAC
        if hasattr(self, "dac"):
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
        modelcfg = self.config.model
        data = self.config.data
        len_coef = 1

        if modelcfg.audio_encode == "dac":
            # DAC Z Latent Reduction factor
            audio_lengths = audio_lengths // 320
            len_coef /= 320
            with torch.no_grad():
                audio = self.dac.encode(audio)[0]
                audio = rearrange(audio, "B L T -> B T L")

            audio = self.audio_proj(audio)

        elif modelcfg.audio_encode == "dac-codes":
            with torch.no_grad():
                codes = self.dac.encode(audio)[1]
                audio = [
                    self.dac.quantizer.quantizers[i].codebook(codes[:, i])
                    for i in range(len(self.dac.quantizer.quantizers))
                ]
                audio = torch.cat(audio, dim=-1)
                # Dac length reduction
                audio_lengths = audio_lengths // 320
                len_coef /= 320

            audio = self.whispconv(audio.permute(0, 2, 1))
            # Whisper length reduction
            audio_lengths = audio_lengths // 2
            len_coef /= 2

        elif modelcfg.audio_encode == "mel":
            len_coef /= 256
            audio = self.whispconv(audio)
            len_coef /= 2
            audio_lengths = audio_lengths // 2

        audio = audio + self.audio_pos_emb(torch.arange(audio.shape[1]))

        text_embs = self.text_emb(labels) + self.text_pos_emb(
            torch.arange(labels.shape[1])
        )

        # view of the start and end tokens for each triplet in the sequence of coords
        # [start, end, label, start, end, label, ...]
        # [
        #   [start, end],
        #   [start, end]
        #   ...
        # ]
        if modelcfg.use_time_pos:
            time_boundaries = labels.view(B, -1, 3)[:, :, :2]
            # Add concept of time to the time boundary tokens
            rearrange(text_embs, "B (b s) L -> B b s L", s=3)[:, :, :2].add_(
                self.time_pos_emb(time_boundaries)
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

        embs = pad_sequence(embs, batch_first=True)

        # Fixed size for torch.compile
        # max audio len sequence + max labels + drz_cmd
        max_seq_len = int(data.max_secs * modelcfg.sample_rate * len_coef) + 30 * 3 + 1
        embs = torch.nn.functional.pad(
            embs, (0, 0, 0, max(0, max_seq_len - embs.size(1)), 0, 0)
        )
        x = self.decoder(embs)

        text_latents = [
            x[b, audio_lengths[b] : audio_lengths[b] + label_lengths[b]]
            for b in range(B)
        ]

        text_latents = pad_sequence(text_latents, batch_first=True)
        text_logits = self.text_head(text_latents).permute(0, 2, 1)
        return F.cross_entropy(text_logits, labels, ignore_index=self.pad_idx)

    @torch.inference_mode()
    def generate(
        self,
        audio: Tensor,
        prefix_labels: Tensor = None,
        temperature=0.8,
        # Total number of labels * 3
        # Must be a multiple of 3
        max_steps=3 * 10,
        top_k: int = 100,
        top_p: float = 0.0,
    ):
        cfg = self.config.model

        if len(audio.shape) == 1:
            audio = audio[None]

        if len(audio.shape) == 2:
            audio = audio[None]

        if cfg.audio_encode == "dac":
            with torch.no_grad():
                audio = self.dac.encode(audio)[0]
                audio = rearrange(audio, "B L T -> B T L")

            audio = self.audio_proj(audio)

        elif cfg.audio_encode == "dac-codes":
            codes = self.dac.encode(audio)[1]

            audio = [
                self.dac.quantizer.quantizers[i].codebook(codes[:, i])
                for i in range(len(self.dac.quantizer.quantizers))
            ]
            audio = torch.cat(audio, dim=-1)
            audio = self.whispconv(audio)

        elif cfg.audio_encode == "mel":
            # Turn into a mel spectrogram
            if audio.shape[1] == 1:
                audio = self.mel(audio[0])
            audio = self.whispconv(audio)

        audio = audio + self.audio_pos_emb(torch.arange(audio.shape[1]))

        emb = torch.cat((audio, self.start_diarize_emb[None][None]), dim=1)

        if prefix_labels is not None:
            prefix_emb = self.text_emb(prefix_labels)[None]
            emb = torch.cat((emb, prefix_emb), dim=1)

        tokens = torch.zeros(0, dtype=torch.long).cuda()

        for step in range(max_steps):
            latents = self.decoder(emb)[:, [-1], :]
            logits = self.text_head(latents)

            min_value = torch.finfo(logits.dtype).min

            # We never want to predict the padding token
            logits[..., 0] = min_value

            # We know that we're predicting start, end, label
            # So we can make eos and label no predictable
            if step % 3 != 0 or step == 0:
                # Prevent EOS
                logits[..., 1] = min_value

            class_pred_step = (step - 2) % 3 == 0

            # Class prediction steps
            if class_pred_step:
                logits[..., : -self.num_classes] = min_value
            else:
                # Prevent class prediction
                logits[..., -self.num_classes :] = min_value

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

            next_token = next_token.flatten().long()

            # If we're at a class prediction step, prevent overlapping
            tokens = torch.cat([tokens, next_token])

            next_emb = self.text_emb(next_token)[None] + self.text_pos_emb(step)

            # Add time embedding
            if not class_pred_step and cfg.use_time_pos:
                next_emb = next_emb + self.time_pos_emb(next_token)

            emb = torch.cat((emb, next_emb), dim=1)

        # If it isn't, remove the last prediction?
        assert tokens.shape[-1] % 3 == 0

        # Convert these tokens into labels
        nlabels = []

        for start, end, label in tokens.split(3):
            # Unquantize the start and end times
            start = dequantize(start, self.config.data.max_secs, self.num_time_tokens)
            end = dequantize(end, self.config.data.max_secs, self.num_time_tokens)
            label = dec_label(label, self.num_embs)
            nlabels.append([start.item(), end.item(), label])

        return nlabels

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
