import hashlib
import random
import subprocess
from typing import Union

from librosa.filters import mel as librosa_mel_fn
import math

import numpy as np
import torch
import torch.distributed as dist
import hashlib
import os
import pickle
from functools import wraps
import torch.nn as nn
import torchaudio
from torch import Tensor
from nanodrz.format_conversions import labels_to_annotation
from torchaudio.transforms import Resample
import torch.nn.functional as F

from nanodrz.constants import CACHE_DIR


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


def get_random_states():
    # Get the state of torch, numpy and random libraries random state
    torch_state = torch.get_rng_state()
    np_state = np.random.get_state()
    random_state = random.getstate()
    return torch_state, np_state, random_state


def set_random_states(torch_state, np_state, random_state):
    """Set the state of the libraries"""
    torch.set_rng_state(torch_state)
    np.random.set_state(np_state)
    random.setstate(random_state)


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


def play(audio: [Tensor, np.ndarray, str], sr=16000, autoplay=True):
    from IPython.display import Audio, display

    if type(audio) is str:
        audio, sr = torchaudio.load(audio)

    assert audio.numel() > 100, "play() needs a non empty audio array"

    audio = audio.flatten()
    if audio.dim() < 2:
        audio = audio[None]

    # Sum Channels
    if audio.shape[0] > 1:
        audio = audio.sum(dim=0)

    display(Audio(audio.cpu().detach(), rate=sr, autoplay=autoplay, normalize=False))


def to_device(obj: [nn.Module, Tensor, list, dict], targets: str | list[str]):
    """
    Takes obj and iterates through the keys putting them on the `device`
    """
    if isinstance(obj, (float, int, str)):
        return obj
    elif isinstance(obj, Tensor) or isinstance(obj, nn.Module):
        return obj.to(targets)
    elif isinstance(obj, (list, tuple)):
        return [to_device(item, targets) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_device(value, targets) for key, value in obj.items()}
    else:
        raise Exception("Must be called with list, dict, Tensor or ")


def visualise_annotation(labels: list):
    from IPython.display import display

    annotation = labels_to_annotation(labels)
    display(annotation)


RESAMPLERS = {}


def resample(source: int, target: int, audio: Tensor):
    """Maintains classes globally for resampling"""
    global RESAMPLERS
    if source == target:
        return audio

    # Check resampler
    if source not in RESAMPLERS:
        RESAMPLERS[source] = {}
    if target not in RESAMPLERS[source]:
        RESAMPLERS[source][target] = Resample(source, target)

    return RESAMPLERS[source][target](audio)


def get_file_duration(file: str):
    """Returns the duration in seconds of the given file"""
    info = torchaudio.info(file)
    duration = info.num_frames / info.sample_rate
    return duration


def hash_arguments(args, kwargs):
    arguments = list(args) + list(kwargs.keys()) + list(kwargs.values())
    return "".join([sha256(b) for b in arguments])


def cache(location=".cache") -> callable:
    def inner_function(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            os.makedirs(location, exist_ok=True)
            s = hash_arguments(args, kwargs)
            key = f.__name__ + s
            # Hash the args correctly
            fname = sha256(key)
            fname = os.path.join(location, fname)
            if os.path.exists(fname):
                with open(fname, "rb") as fl:
                    return pickle.load(fl)
            ret = f(*args, **kwargs)
            with open(fname, "wb") as fl:
                pickle.dump(ret, fl)
            return ret

        return wrapper

    return inner_function


def contains_non_silence(audio, sr=16000, min_duration=0.2, threshold=0.1) -> bool:
    audio = audio < threshold
    kernel_size = int(min_duration * sr)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = torch.ones(1, kernel_size)

    out = F.conv1d(audio, kernel, paddig=kernel_size // 2)
    return (out == kernel_size).any()


@cache(os.path.join(CACHE_DIR, "find_nonsilence"))
def find_nonsilence_chunks(
    audio_file: str,
    silence_threshold=0.01,
    min_silence_len=0.2,
    min_duration=1,
    device="cpu",
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
    audio, sr = torchaudio.load(audio_file)
    audio = resample(sr, 16000, audio)
    sr = 16000
    chunks = find_nonsilence_chunks_vtrz(
        audio.to(device),
        silence_threshold,
        min_silence_len,
        sr,
        device=device,
        min_duration=min_duration,
    )

    bn = os.path.basename(audio_file)
    ext = "." + bn.split(".")[-1]
    bn = bn.replace(ext, "")
    chunk_paths = []
    os.makedirs(os.path.join(CACHE_DIR, "chunks"), exist_ok=True)

    for i, c in enumerate(chunks):
        f = bn + "_" + str(i) + ext
        p = os.path.join(CACHE_DIR, "chunks", f)
        chunk_paths.append(p)
        torchaudio.save(p, c.cpu(), sr)

    return chunk_paths


@torch.jit.script
def find_nonsilence_chunks_vtrz(
    audio: torch.Tensor,
    silence_threshold: float = 0.02,
    min_silence_len: float = 0.3,
    sr: int = 16000,
    min_duration: int = 4,
    device: torch.device = "cuda",
    chunk_size: int = 60 * 16000,
) -> list[torch.Tensor]:
    if audio.shape[-1] < sr * min_duration:
        return [audio]

    if chunk_size is None:
        chunk_size = sr * 60

    if audio.shape[-1] > chunk_size:
        # Pad the audio shape to be divisible by chunk_size
        padding_length = chunk_size - (audio.shape[-1] % chunk_size)
        audio = torch.cat(
            (audio, torch.zeros(1, padding_length, device=device)), dim=-1
        )
    else:
        chunk_size = audio.shape[-1]

    silence = torch.abs(audio) < silence_threshold
    silence = silence.float()

    kernel_size = int(min_silence_len * sr)

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = torch.ones(1, 1, kernel_size, device=device)

    out = []

    cur_chunk = torch.zeros(1, 0, device=device)

    for i, chunk in enumerate(silence.chunk(silence.shape[-1] // chunk_size, dim=-1)):
        silence_padding = torch.ones(1, kernel_size, device=device)
        chunk = torch.cat((silence_padding, chunk, silence_padding), dim=-1)
        conv_output = F.conv1d(chunk[None], kernel, stride=1)
        idxs = (conv_output == kernel_size).int().flatten()
        switches = idxs[:-1] - idxs[1:]
        starts = switches.eq(1).nonzero().flatten()
        ends = switches.eq(-1).nonzero().flatten()

        shift = i * chunk_size

        for s, e in zip(starts, ends):
            c = audio[:, shift + s : shift + e]
            cur_chunk = torch.cat((cur_chunk, c), dim=-1)
            if (cur_chunk.shape[-1] / sr) > min_duration:
                out.append(cur_chunk)
                cur_chunk = torch.zeros(1, 0, device=device)

    return out


def load_what_you_can(checkpoint: dict, model: nn.Module):
    """
    This method takes a checkpoint and loads as many weights from it as possible:

    If they are the same shape, there's nothing to do

    Will load the smallest shape otherwise.
    """
    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint

    for name, param in checkpoint_state_dict.items():
        if name not in model_state_dict:
            print(f"Ignoring parameter '{name}' because it is not found in the model")
            continue

        model_state = model_state_dict[name]
        mshape = model_state.shape
        pshape = param.shape

        if pshape == mshape:
            model_state.copy_(param)
            continue

        if len(pshape) != len(mshape):
            # Completely different shapes so probably unwise to merge
            continue

        min_shape = [
            min(param.shape[i], model_state.shape[i]) for i in range(len(param.shape))
        ]
        print(name, "model:", mshape, "chkpt:", pshape, "loading:", min_shape)
        idxs = torch.meshgrid(*[torch.arange(s) for s in min_shape])
        model_state[tuple(idxs)].copy_(param[tuple(idxs)])

    return model.load_state_dict(model_state_dict)


def mel_spec(
    audio: Tensor,
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_size: int = 256,
    win_size: int = 1024,
    fmin: int = 0,
    fmax: int = 8000,
) -> Tensor:
    mel_basis: dict[int, Tensor] = {}
    hann_window = {}
    center = False

    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(audio.device)] = (
            torch.from_numpy(mel).float().to(audio.device)
        )
        hann_window[str(audio.device)] = torch.hann_window(
            win_size, device=audio.device
        )

    y = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )

    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    # dynamic_range_compression
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec


def dictdiff(maindict: dict, changeddict: dict) -> dict:
    """
    Returns the difference of two dicts
    """
    assert type(maindict) == dict
    assert type(changeddict) == dict
    from copy import deepcopy

    changeddict = deepcopy(changeddict)

    for k in maindict.keys():
        if k not in changeddict:
            continue

        if type(maindict[k]) == dict:
            # Recursive dict difference had no changes either
            dif = dictdiff(maindict[k], changeddict[k])
            changeddict[k] = dif
            if dif == {}:
                del changeddict[k]
        elif maindict[k] == changeddict[k] or (
            maindict[k] is None and not changeddict[k]
        ):
            del changeddict[k]

    return changeddict


def dict_to_strs(d, prefix="") -> str:
    s = []

    for k, v in d.items():
        if type(v) is dict:
            s += dict_to_strs(v, prefix + k + ".")
        else:
            s += [f"{prefix + k}={v}"]

    return s if prefix != "" else "|".join(s)


def human_readable_time(secs: int) -> str:
    days = secs // (24 * 3600)
    hours = (secs % (24 * 3600)) // 3600
    minutes = (secs % 3600) // 60
    seconds = secs % 60
    return f"{days}d {hours}h {minutes}m {seconds}s"


def human_readable_number(n: int) -> str:
    """Converts long numbers to shorthand:
    1123 -> 1K
    123123 -> 123K
    """
    n = float(n)
    millnames = ["", "K", "M"]
    millidx = max(
        0,
        min(
            len(millnames) - 1, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
        ),
    )

    return "{:.0f}{}".format(n / 10 ** (3 * millidx), millnames[millidx])


def multimap(items: list, func: callable, workers=4, desc=None) -> list:
    """
    Quick and dirty multiprocessing that will return the result of func if it returns None
    """
    from tqdm.contrib.concurrent import process_map

    results = process_map(
        func, items, leave=False, desc=desc, max_workers=workers, total=len(items)
    )
    return list(filter(lambda x: x is not None, results))
