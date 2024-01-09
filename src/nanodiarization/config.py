import getpass
import os
from typing import Literal

import click
import yaml
from pydantic import BaseModel

from .constants import CACHE_DIR, GCS_BUCKET_NAME
from .utils import get_git_commit, get_git_repo


class FooConfig(BaseModel):
    kind: Literal["foo"] = "foo"

class ModelConfig(BaseModel):    
    layers: int = 12
    dmodel: int = 1024
    nheads: int = 16
    dropout: float = 0.0
    bias: bool = False
    max_seqlen: int = 8192
    # Number of codebooks to use
    K: int = 4

class DataConfig(BaseModel):
    val_size: int = 5_000
    num_workers: int = 4
    min_audio_duration: float = 0.5
    max_audio_duration: float = 60.0

class TrainConfig(BaseModel):
    total_steps: int = 1_000_000

    batch_size: int = 8
    gradient_accumulation_steps: int = 1

    # AdamW
    min_lr: float = 1e-5
    max_lr: float | None = None
    weight_decay: float = 0.00
    betas: list[float] = [0.9, 0.95]  # YAML and tuples...

    max_grad_norm: float = 1.0
    log_batch_grad: float = 10.0
    log_batch_grad_steps: int = 1000

    lr_schedule: str = "warmup_then_cosine_decay"
    lr_warmup_steps: int = 800

    log_every: int = 10
    do_val: bool = True
    val_every: int = 10_000

    checkpoint_every: int = 10_000
    checkpoint: str | None = None  # checkpoint to restore from
    continue_from_checkpoint: bool = True  # use the optimizer from the checkpoint

    amp_dtype: str = "bfloat16"

    profile: bool = False  # switch on pytorch profiling

    watch: bool = False  # switch on wandb.watch
    watch_every: int = 1000

    seq_len_warmup_steps: int | None = None

    @property
    def is_resuming(self) -> bool:
        return self.checkpoint is not None


class Config(BaseModel):
    data: DataConfig 
    model: ModelConfig
    train: TrainConfig = TrainConfig()
    seed: int = 42

    name: str | None = None  # memorable name for the run e.g. gptts-small-ref-enc
    notes: str | None = None
    user: str | None = None
    repo: str | None = None
    commit: str | None = None

    run_dir: str | None = None
    local_run_dir: str | None = None


def load_config(config_path: str, edit: bool) -> Config:
    assert os.getenv("WANDB_API_KEY") is not None, "Please make sure you have set your `WANDB_API_KEY`"

    with open(config_path, encoding="utf-8") as f:
        config = Config(**yaml.safe_load(f))

    if config.user is None:
        config.user = getpass.getuser()

    if config.repo is None and config.commit is None:
        config.repo = get_git_repo()
        config.commit = get_git_commit()

    if config.name is None:
        config.name = config.model.kind

    local_run_dir = os.path.join(CACHE_DIR, "gs", d)
    config.local_run_dir = local_run_dir
    os.makedirs(config.local_run_dir, exist_ok=True)

    if edit:
        edited = click.edit(yaml.dump(config.dict()))
        if edited is not None:
            config = Config(**yaml.safe_load(edited))

    return config
