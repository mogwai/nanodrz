import glob
import itertools
import os
import random
from dataclasses import dataclass
from os.path import expanduser

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from nanodrz import download
from nanodrz.model import DiarizeGPT
from nanodrz.utils import resample, find_nonsilence_chunks


@dataclass
class Utterance:
    file_url: str | None = None
    seconds: float | None = None
    sr: int | None = None


@dataclass
class Speaker:
    # Audio samples
    name: str | None = None
    # Change this to a list of files
    utts: list[Utterance] | None = None

    def __repr__(self):
        return self.name


class GeneratorIterableDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)


def collate_fn(model: DiarizeGPT) -> callable:
    cfg = model.config

    def _collate(batch):
        audios = [b[0] for b in batch]
        audio_lengths = torch.tensor([a.shape[-1] for a in audios])

        if cfg.model.audio_encode == "mel":
            audios = [model.mel(a)[0] for a in audios]
            audio_lengths = audio_lengths // cfg.data.hop_length

        labels = [b[1] for b in batch]

        Q = model.config.data.max_secs / model.num_time_tokens

        truth = [[l.copy() for l in b] for b in labels]

        for b in labels:
            for l in b:
                l[1] = round(l[1] / Q) + 2
                l[0] = round(l[0] / Q) + 2
                l[2] = model.num_embs - 1 - (ord(l[2]) - ord("A"))

        audios = pad_sequence([a.permute(1, 0) for a in audios], batch_first=True)
        audios = audios.permute(0, 2, 1)
        labels = [torch.tensor(l).flatten().long() for l in labels]
        label_lengths = torch.tensor([len(b) for b in labels])
        labels = pad_sequence(labels, batch_first=True)
        print(label_lengths)
        return {
            "audio": audios,
            "labels": labels,
            "truth": truth,
            "audio_lengths": audio_lengths,
            "label_lengths": label_lengths,
        }

    return _collate


# Datasets to download:


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


def libritts_test() -> list[Speaker]:
    folder = download.dl_libritts_test()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("_")[0],
    )


def libritts_dev() -> list[Speaker]:
    folder = download.dl_libritts_dev()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("_")[0],
    )


def librilight_small() -> list[Speaker]:
    folder = download.dl_libri_light_small()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("/")[-3],
    )


def librilight_medium() -> list[Speaker]:
    folder = download.dl_libri_light_medium()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("/")[-3],
    )


def librilight_large() -> list[Speaker]:
    folder = download.dl_libri_light_large()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("/")[-3],
    )


def artificial_drz_generator(
    model: torch.nn.Module,
    speakers: list[Speaker] = libritts_test(),
    sr=16000,
    max_secs=30,
    **kwargs,
):
    while True:
        audio, label = artificial_diarisation_sample(
            speakers,
            sr=sr,
            max_secs=max_secs,
            **kwargs,
        )

        if hasattr(model, "dac"):
            audio = model.dac.preprocess(audio, sr)

        if audio.shape[-1] / sr > max_secs:
            continue

        yield audio, label


def artificial_diarisation_sample(
    speakers: list[Speaker] = libritts_test(),
    max_secs=30,
    min_secs=7.5,
    interrupt_max=0.2,
    silence_max=0.2,
    num_speakers=4,
    sr=16000,
    **kwargs,
):
    audio = torch.zeros(1, 0)
    names, labels = [], []

    cur_speakers = random.sample(speakers, k=random.randint(2, num_speakers))
    seconds = random.uniform(min_secs, max_secs - 1)

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
        random_sample = resample(ssr, sr, random_sample)
        random_sample = random.choice(find_nonsilence_chunks(random_sample, sr)[0])

        if (audio.shape[-1] + random_sample.shape[-1]) / sr > seconds:
            break

        random_sample = random_sample.sum(dim=0)[None]

        int_range = min(
            interrupt_max, audio.shape[-1] / sr, random_sample.shape[-1] / sr
        )

        cut_point = int(random.uniform(-int_range, silence_max) * sr)
        start_label = audio.shape[-1] / sr + cut_point / sr

        padding = torch.zeros(1, random_sample.shape[-1] + cut_point)
        audio = torch.cat((audio, padding), dim=-1)
        audio[:, -random_sample.shape[-1] :] += random_sample

        if speaker.name not in names:
            i = len(names)
            names.append(speaker.name)
        else:
            i = names.index(speaker.name)

        name_label = chr(ord("A") + i)

        if start_label > 60:
            breakpoint()

        labels.append(
            [start_label, start_label + random_sample.shape[-1] / sr, name_label]
        )

    return audio, labels


def min_duration(min_secs: int = 0.1) -> callable:
    return lambda utt: utt.seconds > min_secs
