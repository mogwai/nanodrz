import logging
import math
import random
import hashlib
from typing import Union

import numpy as np
from torch import Tensor
import subprocess
from typing import Callable, TypeVar, overload

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


import hashlib
from typing import Union
from torch import Tensor

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


T = TypeVar("T")


@overload
def tree_map(fn: Callable, x: list[T]) -> list[T]:
    ...


@overload
def tree_map(fn: Callable, x: tuple[T]) -> tuple[T]:
    ...


@overload
def tree_map(fn: Callable, x: dict[str, T]) -> dict[str, T]:
    ...


@overload
def tree_map(fn: Callable, x: T) -> T:
    ...


def tree_map(fn: Callable, x):
    if isinstance(x, list):
        x = [tree_map(fn, xi) for xi in x]
    elif isinstance(x, tuple):
        x = (tree_map(fn, xi) for xi in x)
    elif isinstance(x, dict):
        x = {k: tree_map(fn, v) for k, v in x.items()}
    elif isinstance(x, Tensor):
        x = fn(x)
    return x


def to_device(x: T, device) -> T:
    return tree_map(lambda t: t.to(device, non_blocking=True), x)


def make_infinite_epochs(dl):
    while True:
        yield from dl


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
        repo = subprocess.check_output(["git", "remote", "get-url", "origin"]).decode().strip()
    except subprocess.CalledProcessError:
        repo = "unknown"

    return repo


def might_have_uncommitted_changes():
    try:
        msg = subprocess.check_output(["git", "status", "-s"]).decode().strip()
    except subprocess.CalledProcessError:
        msg = ""

    return len(msg) > 0


def warmup_then_linear_decay(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float) -> float:
    if step <= warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > total_steps:
        return min_lr
    else:
        return max_lr - max_lr / (total_steps - warmup_steps) * (step - warmup_steps)


def warmup_then_cosine_decay(step: int, warmup_steps: int, total_steps: int, min_lr: float, max_lr: float):
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > total_steps:
        return min_lr
    else:
        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (max_lr - min_lr)


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if rt.is_floating_point():
        rt = rt / world_size
    else:
        rt = rt // world_size
    return rt


def fig_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def tensor_to_numpy_plot(x: Tensor) -> np.ndarray:
    x = x.float().cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 8))
    _ = ax.matshow(x, interpolation="none")
    plt.tight_layout()

    fig.canvas.draw()
    data = fig_to_numpy(fig)
    plt.close()
    return data


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def mask_out_after_eos_id(t: Tensor, eos_id: int, value: int = -100, keep_eos: bool = True) -> Tensor:
    eos_mask = (t == eos_id).float()

    if keep_eos:
        eos_mask = F.pad(eos_mask, (1, -1))

    after_eos_mask = eos_mask.cumsum(dim=-1) > 0

    return t.masked_fill(after_eos_mask, value)


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
    output_ = torch.multinomial(input_, num_samples=num_samples, replacement=replacement, generator=generator)
    output = output_.reshape(*list(input.shape[:-1]), -1)
    return output


def apply_repetition_penalty(prev_ids: Tensor, next_logits: Tensor, repetition_penalty: float = 1.0):
    score = torch.gather(next_logits, -1, prev_ids)
    score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
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


def play(audio:[Tensor, np.ndarray], sr=44100, autoplay=True):
    from IPython.display import display, Audio
    audio = audio.flatten()
    # Sum Channels
    if audio.shape[0] > 1:
        audio = audio.sum(dim=0)
    print(audio.shape)
    display(Audio(audio.cpu().detach(), rate=sr, autoplay=autoplay))