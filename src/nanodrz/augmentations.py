import torchaudio.transforms as tfms
import random
import torch
from torch.nn import Module, ModuleList
from nanodrz.utils import cache, CACHE_DIR
from os.path import join

class AdjustSpeed(Module):
    def __init__(self, sr=16000, max_seconds=30, max_speed=1.2, min_speed=0.8):
        super().__init__()
        self.sr = sr
        self.max_seconds = max_seconds
        self.min_speed = min_speed
        self.max_speed = max_speed

    def forward(self, audio):
        secs = audio.shape[-1] / self.sr
        max_slow = secs / self.max_seconds
        speed_factor = random.uniform(max(max_slow, self.min_speed), self.max_speed)
        return tfms.Speed(self.sr, speed_factor)(audio)[0]


class SinVol(Module):

    def __init__(self, sr=16000):
        super().__init__()
        self.sr = sr

    def forward(self, audio):
        frequency = torch.rand(1) * 0.4 + 0.1
        amplitude = torch.rand(1) * 1.0 + 0.5

        t = torch.arange(audio.shape[-1]) / self.sr
        sine_wave = torch.sin(2 * torch.tensor(torch.pi) * frequency * t) * amplitude

        audio_with_sine = torch.mul(audio, sine_wave)
        return audio_with_sine.clamp(-1, 1)


class RandPitchShift(Module):

    def __init__(self, sr=16000):
        super().__init__()
        self.pitch_shifts = ModuleList(tfms.PitchShift(sr, i) for i in [-1, 1])
        # [p.initialize_parameters() for p in self.pitch_shifts]

    def forward(self, audio):
        shift = random.choice(self.pitch_shifts)
        return shift(audio)


class AddNoise(Module):

    def forward(self, audio):
        audio = audio + torch.rand_like(audio) * random.uniform(0.01, 0.2)
        return audio.clamp(-1, 1)

@torch.inference_mode()
def denoise(denoiser, audio, sr):    
    audio = audio.sum(dim=0, keepdim=True)
    B = 40
    denoiser = denoiser.cuda()
    wav = audio.split(B*denoiser.sample_rate, dim=1)
    denoised = []
    for w in wav:
        denoised.append(denoiser(w.cuda()))
    denoiser = denoiser.cpu()
    denoised = torch.cat(denoised, dim=-1)
    return denoised


def build_augmentations(augmentations: list[tuple[Module, float]]):
    def _inner(audio):
        for aug, chance in augmentations:
            if random.random() < chance:
                audio = aug(audio)

        return audio

    return _inner
