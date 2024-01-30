import os
import time
from contextlib import nullcontext

import click
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

import wandb
from nanodrz import data, utils, download
from nanodrz.config import Config, load_config
from nanodrz.data import GeneratorIterableDataset, collate_fn
from nanodrz.model import DiarizeGPT as Model
from nanodrz import optim, download
from nanodrz.utils import count_parameters, reduce_tensor, seed_all, to_device
from pyannote.metrics.diarization import DiarizationErrorRate
from nanodrz import format_conversions as format


def train(rank: int, world_size: int, config: Config, dev: bool = False):
    if world_size > 1:
        init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
        )

    torch.cuda.set_device(rank)

    start_time = time.time()

    train = config.train

    datacfg = config.data

    assert config.run_dir is not None

    is_main_process = rank == 0
    B = train.batch_size

    if is_main_process:
        wandb.init(
            project="nano-diarization",
            config=config.model_dump(),
            settings=wandb.Settings(),
            name="dev" if dev else None,
        )

    # Get Speakers
    speakers = []
    for ds in datacfg.synth_datasets:
        speakers += getattr(data, ds)()

    print(
        f"Speakers: {len(speakers)} Effective BS: {B*world_size*train.grad_acc_steps}"
    )

    seed_all(config.seed)

    device_type = "cuda"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()

    model = Model(config).cuda(rank)

    ds = GeneratorIterableDataset(
        data.artificial_drz_generator(
            model,
            speakers,
            model.config.model.sample_rate,
            **datacfg.model_dump(),
        )
    )

    train_dl = DataLoader(
        ds,
        batch_size=B,
        collate_fn=collate_fn(model),
        num_workers=datacfg.num_workers,
        pin_memory=True,
        persistent_workers=datacfg.num_workers > 0,
    )

    # We're not doing val yet
    val_dl = train_dl

    betas = tuple(train.betas)
    optimizer = model.configure_optimizers(
        weight_decay=train.weight_decay, lr=train.min_lr, betas=betas
    )

    step = 0
    hours_seen = 0.0

    if train.checkpoint is not None:
        checkpoint_path = train.checkpoint
        if ":" in checkpoint_path:
            checkpoint_path = download.dl_scp_file(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{rank}")
        utils.load_what_you_can(checkpoint["model"], model)
        hours_seen = checkpoint["step"]

        if train.continue_from_checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            step = checkpoint["step"]

        del checkpoint
        torch.cuda.empty_cache()

    # if not dev:
    #     model = torch.compile(model)
    if is_main_process and train.wandb_watch:
        wandb.watch(model, log="all", log_freq=train.watch_every)

    if world_size > 1:
        model = DDP(model, device_ids=[rank])

    if is_main_process:
        print(f"{count_parameters(model) / 1_000_000:.2f}M parameters")

    max_lr = train.max_lr or 10.0 * train.min_lr
    gradient_accumulation_steps = train.grad_acc_steps

    train_dl_iter = iter(train_dl)

    batch = to_device(next(train_dl_iter), device)

    def wandb_log(*args, **kwargs):
        if is_main_process:
            wandb.log(*args, **kwargs)

    get_lr = getattr(optim, train.lr_schedule)

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

    loss = 0.0
    losses = []
    running = True

    if is_main_process:
        print(f"Took {time.time() - start_time:.2f}s to hit training")

    with prof:
        while step < steps and running:
            t1 = time.perf_counter()
            lr = get_lr(
                step, train.lr_warmup_steps, train.total_steps, train.min_lr, max_lr
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            for micro_step in range(gradient_accumulation_steps):
                hours_seen += (
                    batch["audio_lengths"].sum()
                    / config.model.sample_rate
                    / 60
                    / 60
                    / 60
                )
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )

                with torch.amp.autocast(
                    enabled=True, device_type=device_type, dtype=dtype
                ), torch.backends.cuda.sdp_kernel(**train.flash.model_dump()):
                    del batch["truth"]
                    loss = model(**batch)
                    loss = loss / gradient_accumulation_steps

                if torch.isnan(loss):
                    wandb.alert("Nan Detection", "Stopping!")
                    running = False
                    continue

                loss.backward()
                batch = to_device(next(train_dl_iter), device)

            if is_main_process and step == 0:
                print(
                    f"Took {time.time() - start_time:.2f}s to finish first step training loop"
                )

            grad_norm = clip_grad_norm_(model.parameters(), train.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            t2 = time.perf_counter()

            if train.torch_profile:
                prof.step()

            if step % train.log_every == 0:
                loss = loss.item()
                losses.append(loss)
                loss_slope = optim.calculate_smoothed_slope(
                    losses,
                    regression_win=train.regression_win,
                    smoothing_constant=train.regression_smoothing,
                )

                if loss_slope > 1e-3 and steps > 400:
                    wandb.alert("Explosion Warning", "Check loss graph")

                metrics = {
                    "train/loss": gradient_accumulation_steps * loss,
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr,
                    "train/batch_duration": t2 - t1,
                    "hours_seen": hours_seen,
                    "train/loss_slope": loss_slope,
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

                print(f"{device=} val took {val_duration:.2f}s for {val_steps} steps")

                wandb_log(
                    {"val/loss": val_loss.item(), "val/duration": val_duration},
                    step=step,
                )

                model.train()

            if step % train.checkpoint_every == 0 and is_main_process:
                # Evaluation
                batch = next(train_dl_iter)
                batch = to_device(batch, "cuda")
                der = 0
                distance_mean = 0

                for i, audio in enumerate(batch["audio"]):
                    # For now we stop the model generating too far
                    labels = model.generate(audio, max_steps=len(batch["truth"][i]) * 3)
                    truth = batch["truth"][i]
                    labels_annotation = format.labels_to_annotation(labels)
                    truth_annotation = format.labels_to_annotation(truth)
                    der += DiarizationErrorRate()(truth_annotation, labels_annotation)

                    # Sort the lists by start
                    truth.sort(key=lambda x: x[0])
                    labels.sort(key=lambda x: x[0])

                    # Calculate the absolute distance between starts and ends
                    distances = []
                    for i in range(len(truth)):
                        truth_start, truth_end, _ = truth[i]
                        labels_start, labels_end, _ = labels[i]
                        distance = abs(truth_start - labels_start) + abs(
                            truth_end - labels_end
                        )
                        distances.append(distance)

                    # Print the distances
                    distance_mean += sum(distances) / len(distances)

                der = B
                distance_mean /= B

                wandb_log(
                    {
                        "eval/DER": der,
                        "eval/distance_mean": distance_mean,
                    },
                    step=step,
                )

                checkpoint = {
                    "config": config.model_dump(),
                    "step": step,
                    "hours_seen": hours_seen,
                    "model": {
                        k: v
                        for k, v in model.state_dict().items()
                        if not k.startswith("dac.")
                    },
                    "optimizer": optimizer.state_dict(),
                }
                file_url = os.path.join(config.run_dir, f"{step:07}.pt")
                print(f"Saved checkpoint to {file_url}")
                torch.save(checkpoint, file_url)

    if world_size > 1:
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

    config: Config = load_config(config, edit)

    config.train.torch_profile = config.train.torch_profile or profile
    config.train.wandb_watch = config.train.wandb_watch or watch

    if dev:
        print("Running in dev mode (smaller dataset, batch size, fewer epochs, etc.)")
        config.train.val_every = 1
        config.train.batch_size = 2
        config.train.total_steps = 2
        config.train.checkpoint_every = 1
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
            args=(world_size, config, dev),
            nprocs=world_size,
            join=True,
        )
    else:
        train(0, world_size, config, dev)


if __name__ == "__main__":
    main()
