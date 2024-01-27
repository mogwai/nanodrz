import getpass
import os
import time
from typing import Literal

import click
import yaml
from pydantic import BaseModel

from .constants import RUN_DIR
from .utils import get_git_commit, get_git_branch


class ModelConfig(BaseModel):
    layers: int = 8
    dmodel: int = 1024
    nheads: int = 16
    dropout: float = 0
    bias: bool = False
    audio_encode: str = "dac"
    dac_model: str = "16khz"
    sample_rate: int = 16000
    # Turn this on this if you have an ampere GPU
    use_flash_attn: bool = False
    use_time_pos: bool = True
    num_embs: int = 512

class DataConfig(BaseModel):
    num_workers: int = 8
    max_secs: float = 30.0
    min_secs: float = 10.0
    interrupt_max: float = 0.1
    silence_max: float = 1
    num_speakers: int = 8


class TrainConfig(BaseModel):
    total_steps: int = 1_000_000

    # If this is set to none, then the batch size will be determine automatically
    batch_size: int = None
    # How many steps to do the forward before computing backward
    grad_acc_steps: int = 21

    # AdamW
    min_lr: float = 1e-5
    max_lr: float = 5e-5
    weight_decay: float = 0.00
    betas: list[float] = [0.9, 0.95]  # YAML and tuples...

    max_grad_norm: float = 1.0
    log_batch_grad: float = 10.0
    log_batch_grad_steps: int = 1000

    lr_schedule: str = "warmup_then_cosine_decay"
    lr_warmup_steps: int = 800

    do_val: bool = False
    val_every: int = 10_000

    checkpoint_every: int = 500
    checkpoint: str | None = None
    continue_from_checkpoint: bool = True

    amp_dtype: str | None = None
    torch_profile: bool = False
    wandb_watch: bool = False
    log_every: int = 1
    watch_every: int = 1000

    @property
    def is_resuming(self) -> bool:
        return self.checkpoint is not None


class Config(BaseModel):
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    seed: int = 42

    name: str | None = "nanodrz"
    commit: str | None = None
    branch: str | None = None
    run_dir: str | None = None


def load_config(config: str | Config, edit: bool) -> Config:
    assert (
        os.getenv("WANDB_API_KEY") is not None
    ), "Please make sure you have set your `WANDB_API_KEY`"

    if type(config) is str:
        with open(config, encoding="utf-8") as f:
            config = Config(**yaml.safe_load(f))

    if config.branch is None and config.commit is None:
        config.branch = get_git_branch()
        config.commit = get_git_commit()

    config.run_dir = os.path.join(RUN_DIR, config.name, str(int(time.time())))
    os.makedirs(config.run_dir, exist_ok=True)

    if edit:
        edited = click.edit(yaml.dump(config.dict()))
        if edited is not None:
            config = Config(**yaml.safe_load(edited))

    return config
