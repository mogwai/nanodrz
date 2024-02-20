from glob import glob
from nanodrz.augmentations import denoise
import torchaudio
import os
from os.path import basename, exists, join
from tqdm import tqdm

from denoiser import pretrained
import torchaudio
from nanodrz.utils import resample
import torch
from dataclasses import dataclass
from nanodrz.utils import sha256, CACHE_DIR
from denoiser import pretrained
import torch
import torchaudio

torch.cuda.set_device("cuda:0")
denoiser = pretrained.dns64().cuda().eval()



@torch.inference_mode()
def denoise(denoiser, audio):
    audio = audio.sum(dim=0, keepdim=True)
    # audio = resample(sr, denoiser.sample_rate, audio)
    B = 40
    denoiser = denoiser.cuda()
    wav = audio.split(B * denoiser.sample_rate, dim=1)
    denoised = []
    for w in wav:
        denoised.append(denoiser(w.cuda()))
    denoised = torch.cat(denoised, dim=-1)
    return denoised


files = glob("/home/harry/.cache/nanodrz/**/*.flac", recursive=True)

denoiser = pretrained.dns64().cuda().eval()


@dataclass
class Utterance:
    file: str
    speaker: str
    segements: list[tuple[int, int]] = []


# Voice Activity Detection
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
(get_speech_timestamps, _, read_audio, *_) = utils

with torch.inference_mode():
    nfiles = []
    for source in tqdm(files, desc="denoise", leave=False):
        fsplit = f.split("/")
        fsplit[-1] = fsplit[-1].replace(".flac", "_d.wav")
        destination = "/".join(fsplit)

        if exists(destination):
            continue

        audio, sr = torchaudio.load(source)
        assert sr == 16000

        out = denoise(audio.cuda()).cpu()[0]
        torchaudio.save(destination, out, 16000)
        nfiles.append(destination)

    del denoiser

    files = nfiles
    nfiles = []

    os.makedirs(join(CACHE_DIR, "utts"), exist_ok=True)
    
    for f in tqdm(files, desc="vad"):
        hsh = sha256(f)
        p = join(CACHE_DIR, "utts", hsh)
        
        if exists(p):
            continue

        audio, sr = torchaudio.load(f)
        speech_timestamps = get_speech_timestamps(
            audio.cpu(), model, sampling_rate=sr
        )
        utt = Utterance(file=f, speaker=f.split("/")[-3])
        utt.segements = [(s["start"], s["end"]) for s in speech_timestamps]
        torch.save(utt, p)
