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
from dataclasses import dataclass, field
from nanodrz.utils import sha256, CACHE_DIR, multimap
from denoiser import pretrained

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

torch.set_num_threads(1)
import torchaudio

torch.cuda.set_device("cuda:0")
denoiser = pretrained.dns64().cuda().eval()

NUM_COPIES = 8


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
    segements: list[tuple[int, int]] = field(default_factory=lambda: [])


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
    utt = Utterance(file=f, speaker=f.split("/")[-3])
    utt.segements = [(s["start"], s["end"]) for s in speech_timestamps]
    torch.save(utt, p)
    print(p)
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
