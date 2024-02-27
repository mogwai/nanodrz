"""
This script is basically to preprocess a set of audio files contain single speakers. 
Each wav file should have a single speaker in it and a way to indentify that speaker.

This will denoise the audio, removing background sounds and then determine the boundaries of speech
These labels will be used later to pick out sections of speech to be pulled together.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from glob import glob
from os.path import basename, exists, join

import torch
import torchaudio
from denoiser import pretrained
from tqdm import tqdm

from nanodrz.augmentations import denoise
from nanodrz.utils import CACHE_DIR, sha256
from nanodrz import data

torch.cuda.set_device("cuda:0")
denoiser = pretrained.dns64().cuda().eval()

NUM_COPIES = 8

# Downloads some data
data.librilight_small()


@torch.inference_mode()
def denoise(denoiser, audio, B=40):
    """
    Splits up long audio files into batches to be denoised on the gpu faster
    """
    audio = audio.sum(dim=0, keepdim=True)
    denoiser = denoiser.cuda()
    wav = audio.split(B * denoiser.sample_rate, dim=1)
    denoised = []
    for w in wav:
        denoised.append(denoiser(w.cuda()))
    denoised = torch.cat(denoised, dim=-1)
    return denoised


# Files should be all the things you want to preprocess

# Denoised wavs will be saved in the same location with _d.wav so be careful not
# match them here.
files = glob("/home/harry/.cache/nanodrz/**/*.flac", recursive=True)

denoiser = pretrained.dns64().cuda().eval()

_, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True, onnx=False
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

vad_models = dict()


def init_model():
    pid = multiprocessing.current_process().pid
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    vad_models[pid] = model


@torch.inference_mode()
def vad_save(f):
    pid = multiprocessing.current_process().pid
    hsh = sha256(f)
    p = join(CACHE_DIR, "utts", hsh)
    if exists(p):
        return p
    audio, sr = torchaudio.load(f)
    vad = vad_models[pid]
    speech_timestamps = get_speech_timestamps(audio, vad, sampling_rate=sr)
    utt = {}
    utt["file"] = f
    utt["speaker"] = f.split("/")[-3]
    utt["segments"] = [(s["start"], s["end"]) for s in speech_timestamps]
    torch.save(utt, p)
    return p


nfiles = []
for source in tqdm(files, desc="denoise"):
    fsplit = source.split("/")
    fsplit[-1] = fsplit[-1].replace(".flac", "_d.wav")
    destination = "/".join(fsplit)
    nfiles.append(destination)

    if exists(destination):
        continue

    audio, sr = torchaudio.load(source)
    assert sr == 16000

    out = denoise(denoiser, audio.cuda()).cpu()[0]
    torchaudio.save(destination, out, 16000)

del denoiser

os.makedirs(join(CACHE_DIR, "utts"), exist_ok=True)

futures = []

with ProcessPoolExecutor(max_workers=os.cpu_count(), initializer=init_model) as ex:
    for i in nfiles:
        futures.append(ex.submit(vad_save, i))

for finished in as_completed(futures):
    print(finished.result())
