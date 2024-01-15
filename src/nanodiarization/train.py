import logging
import os
import time
from contextlib import nullcontext

import click
import torch
import torch.multiprocessing as mp
import wandb
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, T5EncoderModel

from torch.utils.data import DataLoader, IterableDataset

from nanodiarization.config import Config, load_config, ModelConfig
from nanodiarization.optim import (
    warmup_then_constant,
    warmup_then_cosine_decay,
    warmup_then_inv_sqrt_decay,
    warmup_then_linear_decay,
)
from nanodiarization.model import DiarizeGPT as Model
from nanodiarization.utils import (
    count_parameters,
    might_have_uncommitted_changes,
    reduce_tensor,
    seed_all,
    to_device,
)

from nanodiarization.data import (
    gather_speakers_from_folder,
    artificial_diarisation_sample,
    collate_fn,
)
from nanodiarization import download

from nanodiarization.constants import CACHE_DIR, RUN_DIR

from .data import artificial_drz_generator, GeneratorIterableDataset
import time

logger = logging.getLogger(__name__)


def train(rank: int, world_size: int, config: Config):
    init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    start_time = time.time()

    train = config.train

    if train.checkpoint is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = ModelConfig(**checkpoint["config"]["model"])
    else:
        model_config: ModelConfig = config.model

    data = config.data

    assert config.run_dir is not None

    is_main_process = rank == 0

    folder = download.dl_libritts_clean()
    speakers = gather_speakers_from_folder(folder, lambda x: x.split("/")[-3])

    if is_main_process:
        wandb.init(
            project="nano-diarization", config=config.dict(), settings=wandb.Settings()
        )

    seed_all(config.seed)

    device_type = "cuda"
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    B = train.batch_size

    model = Model(model_config)
    model.cuda(rank)

    ds = GeneratorIterableDataset(artificial_drz_generator(speakers, model))
    train_dl = DataLoader(
        ds, batch_size=B, collate_fn=collate_fn, num_workers=data.num_workers
    )
    val_dl = train_dl

    betas = tuple(train.betas)
    optimizer = model.configure_optimizers(
        weight_decay=train.weight_decay, lr=train.min_lr, betas=betas
    )

    step = 0

    if train.checkpoint is not None:
        checkpoint_path = train.checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{rank}")
        _ = model.load_state_dict(checkpoint["model"], strict=False)

        if train.continue_from_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = checkpoint["step"]

        del checkpoint
        torch.cuda.empty_cache()

    if is_main_process and train.wandb_watch:
        wandb.watch(model, log="all", log_freq=train.watch_every)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if is_main_process:
        logger.info(f"{count_parameters(model) / 1_000_000:.2f}M parameters")

    dtype = torch.bfloat16 if train.amp_dtype == "bfloat16" else torch.float16

    max_lr = train.max_lr or 10.0 * train.min_lr
    gradient_accumulation_steps = train.grad_acc_steps

    train_dl_iter = iter(train_dl)
    batch = next(train_dl_iter)
    batch = to_device(batch, device)

    def wandb_log(*args, **kwargs):
        if is_main_process:
            wandb.log(*args, **kwargs)

    if is_main_process:
        logger.info(f"Took {time.time() - start_time:.2f}s to hit training loop")

    if train.lr_schedule == "warmup_then_linear_decay":
        get_lr = warmup_then_linear_decay
    elif train.lr_schedule == "warmup_then_cosine_decay":
        get_lr = warmup_then_cosine_decay
    elif train.lr_schedule == "warmup_then_inv_sqrt_decay":
        get_lr = warmup_then_inv_sqrt_decay
    else:
        get_lr = warmup_then_constant

    wait, warmup, active = 5, 5, 5
    steps = wait + warmup + active if train.torch_profile else train.total_steps

    prof = (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, repeat=1
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiles"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,  # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False,  # only for torchscript models atm
        )
        if train.torch_profile
        else nullcontext()
    )

    codebook_losses = []

    loss = 0.0
    codebook_losses = None
    hours_seen = 0.0

    with prof:
        while step < steps:
            t1 = time.perf_counter()
            lr = get_lr(
                step, train.lr_warmup_steps, train.total_steps, train.min_lr, max_lr
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            for micro_step in range(gradient_accumulation_steps):
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )

                with torch.amp.autocast(
                    enabled=True, device_type=device_type, dtype=dtype
                ):
                    out = model(**batch)
                    loss = out["loss"]
                    loss = loss / gradient_accumulation_steps

                batch = next(train_dl_iter)
                batch = to_device(batch, device)
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), train.max_grad_norm
            )

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            t2 = time.perf_counter()

            if train.torch_profile:
                prof.step()

            if step % train.log_every == 0:
                metrics = {
                    "train/loss": gradient_accumulation_steps * loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr,
                    "train/batch_duration": t2 - t1,
                }

                wandb_log(
                    metrics,
                    step=step,
                )

            if train.do_val and step % train.val_every == 0:
                model.eval()
                val_dl.seed(config.seed)

                val_loss = torch.tensor(0.0, device=device)
                val_steps = 0

                val_start = time.perf_counter()

                with torch.no_grad():
                    for val_batch in val_dl:
                        val_batch = to_device(val_batch, device)

                        with torch.amp.autocast(
                            enabled=True, device_type=device_type, dtype=dtype
                        ):
                            out = model(**val_batch)
                            loss = out["loss"]
                            val_loss = val_loss + loss

                        val_steps += 1

                val_loss = reduce_tensor(val_loss, world_size) / val_steps

                val_end = time.perf_counter()

                val_duration = val_end - val_start

                logger.info(
                    f"{device=} val took {val_duration:.2f}s for {val_steps} steps"
                )

                wandb_log(
                    {"val/loss": val_loss.item(), "val/duration": val_duration},
                    step=step,
                )

                model.train()

            if step % train.checkpoint_every == 0 and is_main_process:
                checkpoint = {
                    "config": config.dict(),
                    "step": step,
                    "model": {
                        k: v
                        for k, v in model.state_dict().items()
                        if not k.startswith("dac.")
                    },
                    "optimizer": optimizer.state_dict(),
                }
                filename = f"{model_config.kind}-{step:07}.pt"

                file_url = os.path.join(CACHE_DIR, config.run_dir, filename)

                torch.save(checkpoint, file_url)

    destroy_process_group()


@click.command()
@click.argument("config", type=str, default=None, required=False)
@click.option("--edit", is_flag=True, help="Edit the config before running")
@click.option(
    "--dev",
    is_flag=True,
    help="Run in dev mode (small subset of the dataset, small batch size, small number of epochs, regular checkpoints/eval etc.)",
)
@click.option(
    "--profile",
    is_flag=True,
)
@click.option(
    "--watch",
    is_flag=True,
)
def main(config: str, edit: bool, dev: bool, profile: bool, watch: bool):
    if config is None:
        config = Config()
    
    config = load_config(config, edit)

    config.train.torch_profile = config.train.torch_profile or profile
    config.train.wandb_watch = config.train.wandb_watch or watch

    if dev:
        print("Running in dev mode (smaller dataset, batch size, fewer epochs, etc.)")
        config.train.val_every = 100
        config.train.total_steps = 1000
        config.train.checkpoint_every = 100
        config.train.grad_acc_steps = 1
        config.data.num_workers = 0

    if config.train.is_resuming:
        print(f"Resuming from {config.train.checkpoint}. Incrementing seed.")
        # TODO save the seed state in the checkpoint
        config.seed += 1

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "17778"

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            train,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )
    else:
        train(0, world_size, config)


if __name__ == "__main__":
    main()
