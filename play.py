import torch
from torch.utils.data import IterableDataset, DataLoader
from itertools import chain
from nanodrz import data
from nanodrz.data import GeneratorIterableDataset
from nanodrz.model import DiarizeGPT
from tqdm.contrib.concurrent import thread_map
from multiprocessing import Pool
from tqdm import tqdm
from nanodrz.data import Speaker, Utterance
import itertools
from os.path import basename, expanduser
import glob
from nanodrz.utils import find_nonsilence_chunks

# model = DiarizeGPT()

# ds = GeneratorIterableDataset(data.artificial_drz_generator(model))

# ds2 = data.voxconverse_dev()
# for i in range(len(ds2)):
#     x = ds2[i


def multimap(items: list, func: callable, workers=4, desc=None) -> list:
    """
    Quick and dirty multiprocessing that will return the result of func if it returns None
    """
    results = thread_map(
        func, items, leave=False, desc=desc, max_workers=workers, total=len(items)
    )
    return list(filter(lambda x: x is not None, results))


def add(x):
    return x + 1


folder = data.download.dl_libritts_dev()


def gather_speakers_from_folder(
    folder: str,
    retrieve_speaker: callable,
    exts: list[str] = ["wav", "opus", "mp3", "flac"],
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
    speakers: dict[str, Speaker] = {}

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
        chunk_files = find_nonsilence_chunks(file)

        for c in chunk_files:
            utt = Utterance()
            utt.file_url = c
            speakers[speaker_name].utts.append(utt)

    for s in speakers.values():
        s.utts = list(enumerate(s.utts))

    return speakers.values()


speakers = data.libritts_dev()
multimap(list(range(20000)), add)[:10]
