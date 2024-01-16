import logging
import math
import random
import hashlib
from typing import Union

import numpy as np
from torch import Tensor
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from torchaudio.transforms import Resample
import torch.nn as nn
import torch.nn.functional as F


import hashlib
from typing import Union
from torch import Tensor

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, nongrad=False):
    return sum(p.numel() for p in model.parameters() if nongrad or p.requires_grad)


def get_git_commit() -> str:
    """
    https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    """
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except subprocess.CalledProcessError:
        commit = "unknown"

    return commit


def get_git_repo() -> str:
    try:
        repo = (
            subprocess.check_output(["git", "remote", "get-url", "origin"])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        repo = "unknown"

    return repo


def get_git_branch() -> str:
    try:
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        branch = "unknown"

    return branch


def might_have_uncommitted_changes():
    try:
        msg = subprocess.check_output(["git", "status", "-s"]).decode().strip()
    except subprocess.CalledProcessError:
        msg = ""

    return len(msg) > 0


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / world_size
    else:
        rt = rt // world_size
    return rt


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_padding_mask(lengths: Tensor) -> Tensor:
    T_max = lengths.max()
    B = lengths.size(0)

    expanded_lengths = torch.arange(T_max).expand(B, T_max).to(lengths)

    return expanded_lengths >= lengths.unsqueeze(1)


def count_nans(x: Tensor):
    nan_mask = torch.isnan(x)
    num_nans = torch.sum(nan_mask).item()
    return num_nans


def prob_mask_like(shape, prob: float, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def multinomial(input: Tensor, num_samples: int, replacement=False, *, generator=None):
    """torch.multinomial with arbitrary number of dimensions, and number of candidates on the last dimension.

    Args:
        input (Tensor): The input tensor containing probabilities.
        num_samples (int): Number of samples to draw.
        replacement (bool): Whether to draw with replacement or not.
    Keywords args:
        generator (torch.Generator): A pseudorandom number generator for sampling.
    Returns:
        Tensor: Last dimension contains num_samples indices
            sampled from the multinomial probability distribution
            located in the last dimension of tensor input.
    """
    input_ = input.reshape(-1, input.shape[-1])
    output_ = torch.multinomial(
        input_, num_samples=num_samples, replacement=replacement, generator=generator
    )
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def apply_repetition_penalty(
    prev_ids: Tensor, next_logits: Tensor, repetition_penalty: float = 1.0
):
    score = torch.gather(next_logits, -1, prev_ids)
    score = torch.where(
        score < 0, score * repetition_penalty, score / repetition_penalty
    )
    # NOTE(james) we used to do this inplace i.e. `next_logits.scatter_(-1, prev_ids, score)` but that seemed to cause weird cuda issues...
    next_logits = torch.scatter(next_logits, -1, prev_ids, score)
    return next_logits


def sample_top_k(probs: Tensor, k: int) -> Tensor:
    """Sample next token from top K values along the last dimension of the input probs tensor.

    Args:
        probs (Tensor): Input probabilities with token candidates on the last dimension.
        k (int): The k in “top-k”.
    Returns:
        Tensor: Sampled tokens.
    """
    top_k_value, _ = torch.topk(probs, k, dim=-1)
    min_value_top_k = top_k_value[..., [-1]]
    probs *= (probs >= min_value_top_k).float()
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs, num_samples=1)
    return next_token


def sample_top_p(probs: Tensor, p: float) -> Tensor:
    """Sample next token from top P probabilities along the last dimension of the input probs tensor.

    Args:
        probs (Tensor): Input probabilities with token candidates on the last dimension.
        p (int): The p in “top-p”.
    Returns:
        Tensor: Sampled tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort *= (~mask).float()
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


def sha256(b: Union[float, list, Tensor, str, bytes, np.ndarray]):
    if isinstance(b, (int, list, float)):
        b = str(b)
    if isinstance(b, Tensor):
        b = b.cpu().numpy()
    if isinstance(b, np.ndarray):
        b = b.tostring()
    if type(b) == str:
        b = b.encode()
    if type(b) == bytes:
        return hashlib.sha256(b).hexdigest()
    else:
        raise Exception("Not implemented a method to handle {0}".format(type(b)))


def play(audio: [Tensor, np.ndarray], sr=44100, autoplay=True):
    from IPython.display import display, Audio

    audio = audio.flatten()
    if audio.dim() < 2:
        audio = audio[None]
    # Sum Channels
    if audio.shape[0] > 1:
        audio = audio.sum(dim=0)

    display(Audio(audio.cpu().detach(), rate=sr, autoplay=autoplay))


def find_nonsilence_chunks(
    audio: Tensor, sr: int, silence_threshold=0.01, min_silence_len=0.2, min_chunk_len=1
):
    """
    Finds and returns non-silence chunks in the given audio.

    Args:
        audio (Tensor): The audio waveform.
        sr (int): The sample rate of the audio.
        silence_threshold (float, optional): The threshold below which audio is considered as silence. Defaults to 0.01.
        min_silence_len (float, optional): The minimum duration of silence to be considered as a separate chunk. Defaults to 0.2.
        min_chunk_len (float, optional): The minimum duration of a non-silence chunk. Defaults to 1.

    Returns:
        List[Tensor]: A list of non-silence chunks.
        List[Tuple[int, int]]: A list of tuples representing the start and end indexes of silence segments.
    """
    # Add min_silence_len+1 silence to the end of the audio
    audio = torch.cat([audio, torch.zeros(1, int(sr * min_silence_len) + 1)], dim=-1)
    amplitude = torch.abs(audio)
    is_silence = amplitude < silence_threshold
    silent_frames = is_silence.all(dim=0)

    silence_indexes = []
    start_idx = 0

    for idx, is_silent in enumerate(silent_frames):
        if is_silent and start_idx == -1:
            start_idx = idx
        elif not is_silent and start_idx != -1:
            if (idx - start_idx) / sr >= min_silence_len:
                silence_indexes.append((start_idx, idx))
            start_idx = -1

    if (
        start_idx is not None
        and (len(silent_frames) - start_idx) / sr >= min_silence_len
    ):
        silence_indexes.append((start_idx, len(silent_frames)))

    chunks = []
    cur_chunk = torch.zeros(1, 0)
    cur_idx = 0

    for b, e in silence_indexes:
        cur_chunk = torch.cat([cur_chunk, audio[:, cur_idx:b]], dim=-1)
        if cur_chunk.shape[-1] > sr * min_chunk_len:
            cur_idx = e
            chunks.append(cur_chunk)
            cur_chunk = torch.zeros(1, 0)
        else:
            cur_idx = b

    if cur_chunk.shape[-1] != 0:
        chunks.append(cur_chunk)

    return chunks, silence_indexes


def to_device(obj: [nn.Module, Tensor, list, dict], targets: str | list[str]):
    """
    Takes obj and iterates through the keys putting them on the `device`
    """
    if isinstance(obj, Tensor) or isinstance(obj, nn.Module):
        return obj.to(targets)
    elif isinstance(obj, (list, tuple)):
        return [to_device(item, targets) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_device(value, targets) for key, value in obj.items()}
    else:
        raise Exception("Must be called with list, dict, Tensor or ")


def visualise_annotation(labels: list):
    from pyannote.core import Annotation, Segment
    from IPython.display import display

    annotation = Annotation()
    for l in labels:
        annotation[Segment(l[0], l[0] + l[1])] = l[2]
    display(annotation)


RESAMPLERS = {}


def resample(source: int, target: int, audio: Tensor):
    """Maintains classes globally for resampling"""
    global RESAMPLERS

    # Check resampler
    if source not in RESAMPLERS:
        RESAMPLERS[source] = {}
    if target not in RESAMPLERS[source]:
        RESAMPLERS[source][target] = Resample(source, target)

    return RESAMPLERS[source][target](audio)


def get_file_duration(file: str):
    """Returns the duration in seconds of the given file"""
    info = torchaudio.info(file)
    return info.num_frames / info.sample_rate
