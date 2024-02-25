from functools import partial
import glob
import itertools
import os
import random
from dataclasses import dataclass, field
import nanodrz.augmentations as augs

from tqdm import tqdm
from os.path import expanduser, join, basename

import torch
import torchaudio

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset


from nanodrz import download
from nanodrz.model import DiarizeGPT
from nanodrz.constants import CACHE_DIR
from nanodrz.utils import resample, find_nonsilence_chunks, multimap
from nanodrz import format_conversions as formats
from tqdm import tqdm
from dataclasses import field



class GeneratorIterableDataset(IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return iter(self.generator)


class DiarizationDataset(IterableDataset):
    def __init__(self, folder, sr=16000, max_secs=30, min_seconds=10):
        self.sr = sr
        self.rttm_files = glob.glob(os.path.join(folder, "*.rttm"))
        self.min_secs = min_seconds
        self.max_secs = max_secs

    def __iter__(self):
        while True:
            i = random.choice(self.rttm_files)
            wav = i.replace(".rttm", ".wav")
            wav, sr = torchaudio.load(wav)
            wav = resample(sr, self.sr, wav)

            with open(i, "r") as file:
                rttm = file.read()

            labels = formats.convert_rttm(rttm)

            # Make sure we're in order
            labels.sort(key=lambda x: x[0])

            # We need to cut the audio to fit in our max seconds
            secs = wav.shape[-1] / self.sr
            duration = random.uniform(self.min_secs, min(self.max_secs, secs))
            assert secs > self.min_secs

            start = 0
            end = 0

            start = random.uniform(0, secs - duration)
            end = start + duration
            wav = wav[:, int(start * self.sr) : int(end * self.sr)]

            labels = list(
                filter(
                    lambda x: (x[0] > start and x[0] < end)
                    or (x[1] > start and x[1] < end)
                    or (x[0] < start and x[1] > end),
                    labels,
                )
            )

            # Clip the labels
            labels = [[max(l[0], start), min(l[1], end), l[2]] for l in labels]
            # Adjust timings
            labels = [[l[0] - start, l[1] - start, l[2]] for l in labels]

            # PIX2Seq said this was best
            random.shuffle(labels)

            yield wav, labels

    def __len__(self):
        return len(self.rttm_files)


def collate_fn(model: DiarizeGPT) -> callable:
    cfg = model.config
    dcfg = model.config.data
    sr = model.config.model.sample_rate

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

        return {
            "audio": audios,
            "labels": labels,
            "truth": truth,
            "audio_lengths": audio_lengths,
            "label_lengths": label_lengths,
        }

    return _collate


def gather_speakers_from_folder(
    folder: str,
    retrieve_speaker: callable,
    exts: list[str] = ["wav", "opus", "mp3", "flac"],
    split_silence: bool = True,
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
    speakers: dict[str, Speaker] = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for file in tqdm(list(wav_files), desc=folder, leave=False):
        # Extract the speaker name from the file path
        speaker_name = retrieve_speaker(file)
        speaker = None

        if speaker_name in speakers:
            speaker = speakers[speaker_name]
        else:
            speaker = Speaker()
            speaker.name = speaker_name
            speaker.utts = []
            speakers[speaker_name] = speaker

        # Seperate into smaller files
        if split_silence:
            chunk_files = find_nonsilence_chunks(file, device=device)
        else:
            chunk_files = [file]

        speakers[speaker_name].utts += chunk_files

    for s in speakers.values():
        s.utts = list(enumerate(s.utts))

    return list(speakers.values())


def libritts_test() -> list:
    folder = download.dl_libritts_test()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("_")[0],
    )


def libritts_dev(split_silence=True) -> list:
    folder = download.dl_libritts_dev()
    return gather_speakers_from_folder(
        folder,
        lambda x: os.path.basename(x).split("_")[0],
        split_silence=split_silence,
    )


def librilight_small(split_silence=True) -> list:
    folder = download.dl_libri_light_small()
    return gather_speakers_from_folder(
        folder,
        lambda x: x.split("/")[-3],
        split_silence=split_silence,
    )


def librilight_medium() -> list:
    folder = download.dl_libri_light_medium()
    return gather_speakers_from_folder(
        folder,
        lambda x: x.split("/")[-3],
    )


def librilight_large() -> list:
    folder = download.dl_libri_light_large()
    return gather_speakers_from_folder(
        folder,
        lambda x: x.split("/")[-3],
    )


def voxconverse_dev(
    sr=16000, max_seconds: int = 30, max_speakers=3
) -> DiarizationDataset:
    folder = download.dl_voxconverse_dev()
    ds = DiarizationDataset(folder, sr, max_seconds)

    delete = []

    for i in tqdm(range(len(ds)), desc="Filtering voxconverse", leave=False):
        labels = set([l[-1] for l in ds[i][1]])
        if len(labels) > max_speakers:
            delete.append(i)

    for i in sorted(delete, reverse=True):
        del ds.rttm_files[i]

    return ds


def artificial_drz_generator(
    model: torch.nn.Module,
    speakers: dict = None,
    sr=16000,
    max_secs=30,
    **kwargs,
):
    if speakers is None:
        speakers = libritts_dev()
    while True:
        audio, label = artificial_diarisation_sample(
            speakers,
            sr=sr,
            max_secs=max_secs,
            **kwargs,
        )

        # This can pad to over the max duration
        if hasattr(model, "dac"):
            audio = model.dac.preprocess(audio, sr)

        if audio.shape[-1] < 1 or audio.shape[-1] / sr > max_secs:
            continue

        yield audio, label


def artificial_diarisation_sample(
    speakers: dict[str] = None,
    max_secs=30,
    min_secs=7.5,
    interrupt_max=1,
    silence_max=0.2,
    num_speakers=3,
    sr=16000,
    **kwargs,
):
    keys = list(speakers.keys())
    audio = torch.zeros(1, 0)
    names, labels = [], []

    cur_speakers = random.sample(keys, k=random.randint(2, num_speakers))
    seconds = random.uniform(min_secs, max_secs - 1)

    last_speaker = None

    for i in range(20):
        # Pick a random speaker
        speaker: str = random.choice(cur_speakers)

        cur_len = audio.shape[-1] / sr

        if seconds - cur_len < 1:
            break

        if speaker == last_speaker:
            continue

        inter = -interrupt_max
        if i == 0:
            inter = 0

        intpad = random.uniform(max(-cur_len + 1, inter), silence_max)
        # We might not have a sample of this length or
        max_sample_len = seconds - cur_len + intpad

        max_sample_len = max(max_sample_len, 1)

        utts = speakers[speaker]
        max_sample_len = min(max([x.length for x in utts]), max_sample_len * sr)

        # What is our smallest segment potential
        sample_len = int(random.uniform(0.5 * sr, max_sample_len))

        # Pick a utterance that is as long as this or longer
        utts = list(filter(lambda x: x.length > sample_len, speakers[speaker]))
        utt = random.choice(utts)

        # Choose a random start point that gives space for the length
        # pick a segment starting with speech
        starts, ends = list(zip(*utt.segments))
        starts = [s for s in starts if s < (utt.length - sample_len)]
        start = random.choice(starts)

        try:
            end = [e for e in ends if e < (start + sample_len) and e > start][-1]
        except Exception as e:
            continue

        sample = torchaudio.load(
            utt.file,
            backend="soundfile",
        )[
            0
        ][:, start:end]

        padding = torch.zeros(1, max(0, sample.shape[-1] + int(intpad * sr)))
        audio = torch.cat((audio, padding), dim=-1)
        audio[:, -sample.shape[-1] :] += sample

        # Derive the labels from the segment labels
        nlabels = [s for s in utt.segments if s[0] >= start and s[1] <= end]

        if speaker not in names:
            i = len(names)
            names.append(speaker)
        else:
            i = names.index(speaker)

        name_label = chr(ord("A") + i)

        # Need to at the cut points and minus the start
        labels += [
            [
                # Length - start +/- intpad - relative_start
                (cur_len * sr + s[0] + intpad * sr - start) / sr,
                (cur_len * sr + s[1] + intpad * sr - start) / sr,
                name_label,
            ]
            for s in nlabels
        ]
        last_speaker = speaker
    audio = audio.clamp(-1, 1)
    return audio, labels
