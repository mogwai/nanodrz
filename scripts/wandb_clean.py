"""
Cleans up test runs and dev runs
"""

import wandb
from wandb.wandb_run import Run

# Replace 'entity/project' with your specific entity and project name
entity_project = "harrycblum/nano-diarization"


api = wandb.Api()
runs: list[Run] = api.runs(entity_project)

for run in runs:
    if run.state == "running":
        continue

    if run.name == "dev":
        run.delete()
        continue

    if not len(run.summary.keys()):
        run.delete()
        continue

    if "_wandb" not in run.summary:
        run.delete()
        continue

    info = run.summary
    # print(run.summary)
    runtime = run.summary.get("_wandb").get("runtime")

    if "train/loss" not in info:
        run.delete()
        continue

    if not runtime or runtime < 60:
        run.delete()
        continue

    if "_step" in info and info["_step"] < 500:
        print(info["_step"])
        run.delete()
        continue
