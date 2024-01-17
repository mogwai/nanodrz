"""
In this file there are various strategies to collate utterances from single speakers
to create diarisation targets.
"""
import torch
from dataclasses import dataclass
import glob
import torchaudio
import os
from os.path import expanduser
import numpy as np
import random
import itertools
import time

from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import IterableDataset
from nanodrz import download

from nanodrz.utils import get_file_duration, resample
from nanodrz.logger import logger

@dataclass
class Utterance:
    file_url: str | None =  None
    seconds: float | None = None
    sr: int | None = None


@dataclass
class Speaker:
    # Audio samples
    name: str|None = None
    # Change this to a list of files
    utts: list[Utterance]|None  = None

    def __repr__(self):
        return self.name


class GeneratorIterableDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)


def collate_fn(batch):
    t1 = time.perf_counter()
    audios = [b[0] for b in batch]
    audio_lengths = [a.shape[-1] for a in audios]
    labels = [b[1][0] for b in batch]
    label_lengths = [l.shape[-1] for l in labels]
    audios = pad_sequence([a.permute(1, 0) for a in audios], batch_first=True).permute(
        0, 2, 1
    )
    labels = pad_sequence(labels, batch_first=True)
    return {
        "audio": audios,
        "labels": labels,
        "audio_lengths": torch.tensor(audio_lengths),
        "label_lengths": torch.tensor(label_lengths),
    }


def artificial_drz_generator(
    speakers: list[Speaker],
    model: torch.nn.Module,
    max_secs=30,
    min_secs=10,
    sr=16000,
    interrupt_sec_mean=0.2,
    interrupt_var=0.1,
    num_speakers=4,
):
    while True:
        t1 = time.perf_counter()
        audio, label = artificial_diarisation_sample(
            speakers,
            max_secs=max_secs,
            min_seconds=min_secs,
            sr=sr,
            interrupt_sec_mean=interrupt_sec_mean,
            interrupt_var=interrupt_var,
            num_speakers=num_speakers,
        )
        audio = model.dac.preprocess(audio, model.dac.sample_rate)
        label = "\n".join([",".join([str(x) for x in l]) for l in label])
        label = model.text_tokenizer.encode(
            label,
            return_tensors="pt",
            padding="longest",
        )
        logger.debug(f"{audio.shape[-1] / model.dac.sample_rate}s of audio in {time.perf_counter() - t1}s")
        yield audio, label


def artificial_diarisation_sample(
    speakers: list[Speaker],
    max_secs=30,
    min_seconds=7.5,
    interrupt_sec_mean=0.2,
    interrupt_var=0.1,
    num_speakers=4,
    sr=16000,
):
    audio = torch.zeros(1, 0)
    names, labels = [], []

    cur_speakers = random.sample(speakers, k=random.randint(2, num_speakers))
    seconds = random.uniform(min_seconds, max_secs)

    last_speaker = None
    # While we're still less than the target secs
    while audio.shape[-1] / sr < seconds:
        # Pick a random speaker
        speaker = random.choice(cur_speakers)

        if speaker.name == last_speaker:
            continue

        last_speaker = speaker.name

        # Pick a random sample
        random_sample_file = random.choice(speaker.utts).file_url
        random_sample, ssr = torchaudio.load(random_sample_file)
        random_sample = random_sample.sum(dim=0)[None]
        random_sample = resample(ssr, sr, random_sample)

        # Beginning
        if audio.shape[-1] == 0:
            audio = random_sample
            start_label = 0
        else:
            # Choose interrupt section with mean interrupt_sec_mean and var interrupt_var
            interrupt_dur = np.random.normal(interrupt_sec_mean, interrupt_var)
            interrupt_dur = min(audio.shape[-1] / sr, interrupt_dur)
            start_label = audio.shape[-1] / sr - interrupt_dur


            try: 
                audio = torch.cat(
                    (
                        audio,
                        torch.zeros(1, random_sample.shape[-1] - int(interrupt_dur * sr)),
                    ),
                    dim=-1,
                )
                audio[:, -random_sample.shape[-1] :] += random_sample
            except Exception as e:
                breakpoint()
        
        if speaker.name not in names:
            i = len(names)
            names.append(speaker.name)
        else:
            i = names.index(speaker.name)

        name_label = chr(ord("A") + i)

        labels.append((start_label, random_sample.shape[-1] / sr, name_label))

    return audio, labels

def gather_speakers_from_folder(
    folder: str,
    retrieve_speaker: callable,
    exts: list[str] = ["wav", "opus", "mp3"],
    file_filters: list[callable] = [],
):
    """
    Retrieves all the audio files from a specific directory recursively.

    param file_filters: callable returning true for files that don't pass
    look at min_duration
    """

    folder = expanduser(folder)
    wav_files = itertools.chain(
        *[glob.glob(folder + f"/**/*.{ext}", recursive=True) for ext in exts]
    )
    speakers: list[Speaker] = []

    for file in wav_files:
        # Extract the speaker name from the file path
        speaker_name = retrieve_speaker(file)
        utt = Utterance()
        utt.file_url = file
        # info = torchaudio.info(file)
        # utt.sr = info.sample_rate
        # utt.seconds = info.num_frames / info.sample_rate

        stop = False
        for check in file_filters:
            if not check(utt):
                stop = True
                break
        if stop:
            continue

        # Check if the speaker object already exists
        speaker = None
        for s in speakers:
            if s.name == speaker_name:
                speaker = s
                break

        # If the speaker object doesn't exist, create a new one
        if speaker is None:
            speaker = Speaker()
            speaker.name = speaker_name
            speakers.append(speaker)
            speaker.utts = []

        speaker.utts.append(utt)

    return speakers

def min_duration(min_secs:int=.1) -> callable:
    return lambda utt: utt.seconds > min_secs

def libritts_test()-> list[Speaker]:
    folder = download.dl_libritts_test()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("_")[0],
        # file_filters=[min_duration()]
    )

def libritts_dev() -> list[Speaker]:
    folder = download.dl_libritts_dev()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("_")[0],
        # file_filters=[min_duration()]
    )
