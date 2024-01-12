import getpass
import os
from typing import Literal

import click
import yaml
from pydantic import BaseModel

from .constants import CACHE_DIR
from .utils import get_git_commit, get_git_repo


class ModelConfig(BaseModel):    
    layers: int = 12
    dmodel: int = 1024
    nheads: int = 16
    dropout: float = 0.0
    bias: bool = False
    max_seqlen: int = 8192
    # Number of codebooks to use
    K: int = 4
    tokenizer_model:str = "google/byt5-small"
    dac_model:str = "16khz"
    sample_rate: int = 16000
    
class DataConfig(BaseModel):
    num_workers: int = 4
    min_audio_duration: float = 0.5
    max_audio_duration: float = 30.0

class TrainConfig(BaseModel):
    total_steps: int = 1_000_000

    batch_size: int = 8
    # How many steps to do the forward before computing backward
    grad_acc_steps: int = 1

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

    do_val: bool = False
    val_every: int = 10_000

    checkpoint_every: int = 10_000
    checkpoint: str | None = None
    continue_from_checkpoint: bool = True

    amp_dtype: str = "bfloat16"
    torch_profile: bool = False
    wandb_watch: bool = False
    log_every: int = 50
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

    name: str | None = None
    commit: str | None = None
    branch: str | None = None
    run_dir: str | None = None


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

    local_run_dir = os.path.join(CACHE_DIR, config.name)
    config.local_run_dir = local_run_dir
    os.makedirs(config.local_run_dir, exist_ok=True)

    if edit:
        edited = click.edit(yaml.dump(config.dict()))
        if edited is not None:
            config = Config(**yaml.safe_load(edited))

    return config
