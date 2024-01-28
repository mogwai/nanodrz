import wandb
from wandb.wandb_run import Run

# Replace 'entity/project' with your specific entity and project name
entity_project = "harrycblum/nano-diarization"

api = wandb.Api()
runs:list[Run] = api.runs(entity_project)

for run in runs:
    if not len(run.summary.keys()):
        run.delete()

    if "_wandb" in run.summary:
        runtime = run.summary.get("_wandb").get("runtime")

        if runtime and runtime > 600:
            continue

        if run.name != "dev":
            continue
        print(run.name)
        run.delete()
