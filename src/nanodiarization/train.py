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
from transformers import AutoTokenizer, T5EncoderModel, UMT5EncoderModel

from nanodiarization.config import Config, load_config
from nanodiarization.optim import (
    warmup_then_constant,
    warmup_then_cosine_decay,
    warmup_then_inv_sqrt_decay,
    warmup_then_linear_decay,
)
from nanodiarization.utils import (
    count_parameters,
    might_have_uncommitted_changes,
    reduce_tensor,
    seed_all,
    to_device,
)

from .data import build_dl

logger = logging.getLogger(__name__)


def train(rank: int, world_size: int, config: Config):
    init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
    )

    start_time = time.time()

    train_config = config.train

    if train_config.checkpoint is not None:
        checkpoint_path = gcs.download(url=train_config.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = SpeechcraftConfig(**checkpoint["config"]["model"])
    else:
        model_config: SpeechcraftConfig = config.model

    data_config = config.data

    assert config.run_dir is not None
    assert config.local_run_dir is not None

    is_main_process = rank == 0

    if is_main_process:
        wandb.init(project="nano-diarization", config=config.dict(), settings=wandb.Settings())

    seed_all(config.seed)

    device_type = "cuda"
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    B = train_config.batch_size

    model = Model(model_config)
    model.cuda(rank)

    betas = tuple(train_config.betas)
    optimizer = model.configure_optimizers(weight_decay=train_config.weight_decay, lr=train_config.min_lr, betas=betas)

    step = 0

    if train_config.checkpoint is not None:
        checkpoint_path = gcs.download(url=train_config.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{rank}")
        _ = model.load_state_dict(checkpoint["model"], strict=False)

        if train_config.continue_from_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = checkpoint["step"]

        del checkpoint
        torch.cuda.empty_cache()

    if is_main_process and train_config.watch:
        wandb.watch(model, log="all", log_freq=train_config.watch_every)

    if 
    model = DDP(model, device_ids=[rank])

    train_dl = build_dl(
        # TODO 
    )

    val_dl = build_dl(
        # TODO
    )

    if is_main_process:
        logger.info(f"{count_parameters(model) / 1_000_000:.2f}M parameters")

    dtype = torch.bfloat16 if train_config.amp_dtype == "bfloat16" else torch.float16

    max_lr = train_config.max_lr or 10.0 * train_config.min_lr
    gradient_accumulation_steps = train_config.gradient_accumulation_steps

    train_dl_iter = iter(train_dl)
    batch = next(train_dl_iter)
    batch = to_device(batch, device)

    def wandb_log(*args, **kwargs):
        if is_main_process:
            wandb.log(*args, **kwargs)

    if is_main_process:
        logger.info(f"Took {time.time() - start_time:.2f}s to hit training loop")

    if train_config.lr_schedule == "warmup_then_linear_decay":
        get_lr = warmup_then_linear_decay
    elif train_config.lr_schedule == "warmup_then_cosine_decay":
        get_lr = warmup_then_cosine_decay
    elif train_config.lr_schedule == "warmup_then_inv_sqrt_decay":
        get_lr = warmup_then_inv_sqrt_decay
    else:
        get_lr = warmup_then_constant

    wait, warmup, active = 5, 5, 5
    steps = wait + warmup + active if train_config.profile else train_config.total_steps

    prof = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiles"),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,  # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False,  # only for torchscript models atm
        )
        if train_config.profile
        else nullcontext()
    )

    codebook_losses = []

    loss = 0.0
    codebook_losses = None
    seq_len = 0
    hours_seen = 0.0

    with prof:
        while step < steps:
            t1 = time.perf_counter()
            lr = get_lr(step, train_config.lr_warmup_steps, train_config.total_steps, train_config.min_lr, max_lr)

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            for micro_step in range(gradient_accumulation_steps):
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1

                with torch.amp.autocast(enabled=True, device_type=device_type, dtype=dtype):
                    out = model(**batch)
                    loss = out["loss"]
                    loss = loss / gradient_accumulation_steps

                    codebook_losses = out["codebook_losses"]
                    codebook_losses = [loss.item() / gradient_accumulation_steps for loss in codebook_losses]

                    seq_len = batch["audio_tokens"].size(-1)
                    hours_seen += batch["audio_tokens_lengths"].sum() / 86.1 / 60 / 60

                batch = next(train_dl_iter)
                batch = to_device(batch, device)
                loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            t2 = time.perf_counter()

            if train_config.profile:
                prof.step()

            if step % train_config.log_every == 0:
                metrics = {
                    "train/loss": gradient_accumulation_steps * loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr,
                    "train/seq_len": seq_len,
                    "train/batch_duration": t2 - t1,
                    "train/hours_seen": hours_seen.item() * world_size,
                }

                for k, codebook_loss in enumerate(codebook_losses):
                    metrics[f"train/loss_{k}"] = gradient_accumulation_steps * codebook_loss

                wandb_log(
                    metrics,
                    step=step,
                )

            if train_config.do_val and step % train_config.val_every == 0:
                model.eval()
                val_dl.seed(config.seed)

                val_loss = torch.tensor(0.0, device=device)
                val_steps = 0

                val_start = time.perf_counter()

                with torch.no_grad():
                    for val_batch in val_dl:
                        val_batch = to_device(val_batch, device)

                        with torch.amp.autocast(enabled=True, device_type=device_type, dtype=dtype):
                            out = model(**val_batch)
                            loss = out["loss"]
                            val_loss = val_loss + loss

                        val_steps += 1

                val_loss = reduce_tensor(val_loss, world_size) / val_steps

                val_end = time.perf_counter()

                val_duration = val_end - val_start

                logger.info(f"{device=} val took {val_duration:.2f}s for {val_steps} steps")

                wandb_log({"val/loss": val_loss.item(), "val/duration": val_duration}, step=step)

                model.train()

            if step % train_config.checkpoint_every == 0 and is_main_process:
                checkpoint = {
                    "config": config.dict(),
                    "step": step,
                    "model": {k: v for k, v in model.module.state_dict().items() if not k.startswith("text_encoder.")},
                    "optimizer": optimizer.state_dict(),
                }
                filename = f"{model_config.kind}-{step:07}.pt"
                file_url = os.path.join(config.run_dir, filename)
                local_path = gcs.get_local_cache_path(file_url)

                torch.save(checkpoint, local_path)

                latest_checkpoint_url = os.path.join(config.run_dir, f"{model_config.kind}-latest.pt")

                # TODO Convert to save to disk
                gcs.upload(local_path, url=file_url)
                gcs.upload(local_path, url=latest_checkpoint_url, overwrite=True)

                logger.info(f"Saved checkpoint to {file_url} {latest_checkpoint_url}")

    train_dl.shutdown()
    val_dl.shutdown()
    destroy_process_group()


@click.command()
@click.argument("config_path", type=str)
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
def main(config_path: str, edit: bool, dev: bool, profile: bool, watch: bool):
    if might_have_uncommitted_changes() and not dev:
        print(
            "You might have uncommited changes and you're not running in dev mode. Please commit your changes and try again."
        )
        return

    config = load_config(config_path, edit)

    config.train.profile = config.train.profile or profile
    config.train.watch = config.train.watch or watch

    if dev:
        print("Running in dev mode (smaller dataset, batch size, fewer epochs, etc.)")
        config.train.val_every = 100
        config.train.total_steps = 1000
        config.train.checkpoint_every = 100
        config.train.gradient_accumulation_steps = 1
        config.data.val_size = 1_000

    TextEncoderCls =T5EncoderModel
    _ = AutoTokenizer.from_pretrained(config.model.text_encoder)
    _ = TextEncoderCls.from_pretrained(config.model.text_encoder)

    if config.train.is_resuming:
        print(f"Resuming from {config.train.checkpoint}. Incrementing seed.")
        config.seed += 1
        _ = gcs.download(url=config.train.checkpoint)

    # TODO(james) torchrun sets a lot of this stuff for you
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "17778"

    world_size = torch.cuda.device_count()

    mp.spawn(
        train,
        args=(world_size, config),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
